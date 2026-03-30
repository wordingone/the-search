"""
Step 1345 — MLP+Adam baseline RHAE distribution across same 5 draws as 1344.
Leo mail 3825, 2026-03-29.

Same MLP architecture as 1337. Same 5 game draws as 1344 (seeds 13440-13444).
Replace TP with Adam (lr=1e-3). Everything else identical.

Question: does Adam produce RHAE > 0 where TP produced zero?
Credit depth test: Adam uses global gradient vs TP's local targets.

Decision tree:
  Adam RHAE > 0 on ≥2/5 draws → credit depth IS the gap.
  Adam RHAE = 0 on all 5    → not credit depth, games too hard for any 2K-step substrate.
  Adam RHAE > 0 on 1/5      → ambiguous, need more draws.

Constitutional note: R2-VIOLATING (Adam). Reference experiment, same precedent as
CNN+Adam at step 1324. Not a direction — a measurement.

Budget: ~7.5 min (30 episodes × ~15s avg). Within threshold.
"""
import sys, os, time, json, logging, hashlib
from collections import deque

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_masked import (
    select_games, seal_mapping, label_filename,
    masked_game_list,
    compute_progress_speedup, format_speedup,
    speedup_for_chain, compute_rhae_try2, write_experiment_results,
    ARC_OPTIMAL_STEPS_PROXY,
)

STEP        = 1345
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

N_DRAWS     = 5
# CRITICAL: same seeds as 1344 for direct comparison
DRAW_SEEDS  = [13440, 13441, 13442, 13443, 13444]

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1345')

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN     = 512
_ADAM_LR    = 1e-3          # Adam lr (spec: 1e-3)
_BUFFER_MAX = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
LOSS_CHECKPOINTS = [500, 1000, 2000]


# ---------------------------------------------------------------------------
# Obs encoding (same as 1344)
# ---------------------------------------------------------------------------

def _is_arc_obs(obs_arr):
    arr = np.asarray(obs_arr, dtype=np.float32)
    return arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] == 64 and arr.shape[2] == 64


def _obs_to_one_hot(obs_arr):
    frame = np.round(obs_arr).astype(np.int32).squeeze(0)
    frame = np.clip(frame, 0, 15)
    one_hot = np.zeros((16, 64, 64), dtype=np.bool_)
    for c in range(16):
        one_hot[c] = (frame == c)
    return one_hot


def _encode_obs_mlp(obs_arr):
    arr = np.asarray(obs_arr, dtype=np.float32)
    if _is_arc_obs(arr):
        return _obs_to_one_hot(arr).astype(np.float32).flatten()
    return arr.flatten()


def _action_type_vec(action, n_types=6):
    vec = np.zeros(6, dtype=np.float32)
    vec[int(action) % n_types] = 1.0
    return vec


# ---------------------------------------------------------------------------
# MLP model (same as 1344)
# ---------------------------------------------------------------------------

class MlpModel(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=_HIDDEN):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden)
        self.fc1         = nn.Linear(hidden, hidden)
        self.fc2         = nn.Linear(hidden, hidden)
        self.fc3         = nn.Linear(hidden, hidden)
        self.action_head = nn.Linear(hidden, n_actions)
        self.pred_head   = nn.Linear(hidden + 6, hidden)
        self.dropout     = nn.Dropout(0.2)

    def forward(self, x):
        h0 = F.relu(self.input_proj(x))
        h1 = F.relu(self.fc1(h0))
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        h3 = F.relu(self.fc3(h2))
        return self.action_head(h3), h3


# ---------------------------------------------------------------------------
# MLP+Adam substrate
# ---------------------------------------------------------------------------

class MlpAdamSubstrate:
    """MLP + Adam (lr=1e-3) + entropy action selection.

    Same architecture as 1344 MlpTpSubstrate but with Adam end-to-end backprop
    instead of target propagation. R2-violating (Adam uses external gradient).
    Reference experiment only.
    """

    def __init__(self, n_actions):
        self.n_actions      = n_actions
        self._rng           = np.random.RandomState(42)
        self._model         = None
        self._optimizer     = None

        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_enc      = None
        self._train_counter = 0
        self._step          = 0
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

    def _init_model(self, input_dim):
        self._model     = MlpModel(input_dim, self.n_actions).to(_DEVICE)
        # Adam trains ALL parameters via backprop through prediction loss
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=_ADAM_LR)

    def reset_for_try2(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def process(self, obs_arr):
        self._step += 1
        enc = _encode_obs_mlp(obs_arr)
        if self._model is None:
            self._init_model(enc.shape[0])
        self._prev_enc = enc
        tensor = torch.from_numpy(enc).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits, _ = self._model(tensor)
        probs = torch.softmax(logits.squeeze(0).cpu(), dim=-1)
        return int(torch.multinomial(probs, 1).item()) % self.n_actions

    def update_after_step(self, obs_next, action, reward):
        if self._prev_enc is None or self._model is None:
            return
        enc_next = _encode_obs_mlp(obs_next)
        at_vec   = _action_type_vec(action)
        h = hashlib.md5(self._prev_enc.tobytes() +
                        np.array([action], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({'state':           self._prev_enc.copy(),
                                 'action_type_vec': at_vec,
                                 'next_state':      enc_next.copy()})
        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            loss = self._adam_train_step()
            if loss is not None:
                self._recent_losses.append(loss)
                for ck in self._pred_loss_at:
                    if self._pred_loss_at[ck] is None and self._step >= ck:
                        self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

    def on_level_transition(self):
        self._prev_enc = None

    def _adam_train_step(self):
        n   = len(self._buffer); buf = list(self._buffer)
        idx = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf[i] for i in idx]

        states      = torch.from_numpy(np.stack([b['state'].astype(np.float32)
                                                 for b in batch])).to(_DEVICE)
        next_states = torch.from_numpy(np.stack([b['next_state'].astype(np.float32)
                                                 for b in batch])).to(_DEVICE)
        action_vecs = torch.from_numpy(np.stack([b['action_type_vec']
                                                 for b in batch])).to(_DEVICE)

        # Target: stop gradient on next-state encoding (standard self-sup learning)
        with torch.no_grad():
            _, target_h3_next = self._model(next_states)

        # Prediction: full backprop through encoder
        _, h3       = self._model(states)
        pred_in     = torch.cat([h3, action_vecs], dim=1)
        pred_next   = self._model.pred_head(pred_in)
        loss        = F.mse_loss(pred_next, target_h3_next)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return float(loss.item())

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(500)
        l_late  = self._pred_loss_at.get(2000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None


# ---------------------------------------------------------------------------
# Game factory
# ---------------------------------------------------------------------------

def make_game(game_name):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        return mbpp_game.make(gn)
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def get_optimal_steps(game_name, seed):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        problem_idx = int(seed) % mbpp_game.N_EVAL_PROBLEMS
        solver = mbpp_game.compute_solver_steps(problem_idx)
        return solver.get(1)
    return ARC_OPTIMAL_STEPS_PROXY


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, seed, max_steps):
    obs            = env.reset(seed=seed)
    steps          = 0
    level          = 0
    progress_count = 0
    steps_to_first_progress = None
    t_start        = time.time()
    fresh_episode  = True

    while steps < max_steps:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action  = substrate.process(obs_arr) % n_actions
        obs_next, reward, done, info = env.step(action)
        steps += 1
        if obs_next is not None:
            substrate.update_after_step(obs_next, action, reward)
        if fresh_episode:
            fresh_episode = False
            obs = obs_next
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            progress_count += 1
            if steps_to_first_progress is None:
                steps_to_first_progress = steps
            level = cl
            substrate.on_level_transition()
        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            level = 0
        else:
            obs = obs_next

    elapsed = time.time() - t_start
    return steps_to_first_progress, progress_count, round(elapsed, 2)


# ---------------------------------------------------------------------------
# Single draw runner
# ---------------------------------------------------------------------------

def run_draw(draw_idx, draw_seed):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    print(f"  Draw {draw_idx} (seed={draw_seed}): {masked_game_list(game_labels)}")
    draw_results   = []
    try2_progress  = {}
    optimal_steps_d = {}

    for game_name, label in zip(games, game_labels.values()):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = 4103

        substrate = MlpAdamSubstrate(n_actions=n_actions)

        p1, _, t1 = run_episode(env, substrate, n_actions, seed=0,        max_steps=TRY1_STEPS)
        substrate.reset_for_try2()
        p2, _, t2 = run_episode(env, substrate, n_actions, seed=TRY2_SEED, max_steps=TRY2_STEPS)

        speedup = compute_progress_speedup(p1, p2)
        opt     = get_optimal_steps(game_name, TRY2_SEED)
        eff_sq  = 0.0
        if p2 is not None and opt is not None and opt > 0:
            eff    = min(1.0, opt / p2)
            eff_sq = round(eff ** 2, 6)

        cr = substrate.get_compression_ratio()
        try2_progress[label]   = p2
        optimal_steps_d[label] = opt

        result = {
            'draw': draw_idx, 'label': label, 'game': game_name,
            'p1': p1, 'p2': p2, 'speedup': speedup,
            'eff_sq': eff_sq, 'optimal_steps': opt, 'cr': cr,
            't1': t1, 't2': t2,
        }
        draw_results.append(result)

        print(f"    {label}: speedup={format_speedup(speedup)}  eff²={eff_sq}  cr={cr}  ({t1+t2:.1f}s)")

        out_path = os.path.join(draw_dir, label_filename(label, STEP))
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, default=str) + '\n')

    rhae     = compute_rhae_try2(try2_progress, optimal_steps_d)
    chain_sp = sum(speedup_for_chain(r['speedup']) for r in draw_results) / len(draw_results)
    print(f"    → RHAE(try2) = {rhae:.6f}  chain_speedup={chain_sp:.4f}")
    return draw_results, rhae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — MLP+Adam: {N_DRAWS} draws. Same games as 1344 (seeds {DRAW_SEEDS}).")
    print(f"Device: {_DEVICE}")
    print(f"Protocol: {N_DRAWS} draws × (MBPP + 2 ARC) × 2 tries = {N_DRAWS*3*2} episodes")
    print(f"Adam lr={_ADAM_LR}. Entropy action selection. 2K steps/try.")
    print()

    # Tier 1: timing on first draw's first ARC game
    games0, labels0 = select_games(seed=DRAW_SEEDS[0])
    first_arc = next((g for g in games0 if not g.lower().startswith('mbpp')), None)
    if first_arc:
        first_label = labels0[first_arc]
        print(f"Tier 1: timing on {first_label} (100 steps, Adam)...")
        env_t = make_game(first_arc)
        try:
            na_t = int(env_t.n_actions)
        except AttributeError:
            na_t = 4103
        sub_t  = MlpAdamSubstrate(n_actions=na_t)
        t0     = time.time()
        obs_t  = env_t.reset(seed=0)
        for _ in range(100):
            a = sub_t.process(np.asarray(obs_t, dtype=np.float32)) % na_t
            obs_t, _, done_t, _ = env_t.step(a)
            if done_t or obs_t is None:
                obs_t = env_t.reset(seed=0)
        elapsed_100 = time.time() - t0
        est_ep      = elapsed_100 / 100 * 2000
        est_total   = est_ep * N_DRAWS * 3 * 2
        print(f"  100 steps: {elapsed_100:.1f}s → est per episode: {est_ep:.0f}s → est total: {est_total:.0f}s")
        if est_total > 900:
            print(f"  WARNING: est {est_total:.0f}s > 15-min budget.")
        print()

    all_results   = []
    rhae_per_draw = []

    for draw_idx in range(N_DRAWS):
        draw_results, draw_rhae = run_draw(draw_idx, DRAW_SEEDS[draw_idx])
        all_results.extend(draw_results)
        rhae_per_draw.append(draw_rhae)
        print()

    # -------------------------------------------------------------------------
    # Aggregate report
    # -------------------------------------------------------------------------
    chain_mean    = sum(rhae_per_draw) / len(rhae_per_draw)
    nonzero_draws = sum(1 for r in rhae_per_draw if r is not None and r > 0)
    nonzero_games = sum(1 for r in all_results if r['eff_sq'] > 0)
    total_games   = len(all_results)

    all_eff_sq = [r['eff_sq'] for r in all_results]

    # Compression trajectory: mean cr per draw
    cr_by_draw = {}
    for r in all_results:
        d = r['draw']
        if r['cr'] is not None:
            cr_by_draw.setdefault(d, []).append(r['cr'])
    cr_means = {d: round(sum(vs)/len(vs), 4) for d, vs in cr_by_draw.items()}

    print("=" * 80)
    print(f"STEP {STEP} — RESULT (MLP+Adam RHAE distribution, same games as 1344)")
    print()
    print(f"  RHAE per draw:  {[f'{r:.6f}' for r in rhae_per_draw]}")
    print(f"  Chain mean:     {chain_mean:.6f}")
    print(f"  Non-zero draws: {nonzero_draws}/{N_DRAWS}")
    print(f"  Non-zero games: {nonzero_games}/{total_games}")
    print()
    print(f"  Per-game efficiency² distribution:")
    print(f"    max  = {max(all_eff_sq):.6f}")
    print(f"    mean = {sum(all_eff_sq)/len(all_eff_sq):.6f}")
    print(f"    non-zero count = {nonzero_games}/{total_games}")
    print()
    print(f"  Compression ratio (loss_2000/loss_500) per draw:")
    for d in sorted(cr_means):
        print(f"    Draw {d}: cr_mean={cr_means[d]}")
    print()
    print("  Games with progress:")
    for r in all_results:
        if r['eff_sq'] > 0:
            print(f"    Draw {r['draw']} {r['label']}: eff²={r['eff_sq']}  p2={r['p2']}  opt={r['optimal_steps']}")
    print()

    # 1344 comparison
    print("  MLP+TP baseline (1344, same seeds):")
    print("    RHAE per draw: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]")
    print("    Chain mean: 0.000000 | Non-zero draws: 0/5 | Non-zero games: 0/15")
    print()

    print("ASSESSMENT (vs 1344 MLP+TP):")
    if nonzero_draws >= 2:
        print(f"  >>> CREDIT DEPTH: Adam RHAE > 0 on {nonzero_draws}/5 draws.")
        print(f"  >>> Chain mean={chain_mean:.6f}. TP can't produce features Adam can.")
        print(f"  >>> Next: find R2-compliant method with deeper credit (EP, deeper TP).")
    elif nonzero_draws == 1:
        print(f"  >>> AMBIGUOUS: Only 1/5 draws non-zero (chain mean={chain_mean:.6f}).")
        print(f"  >>> Need more draws to distinguish credit depth from luck.")
    else:
        print(f"  >>> NOT CREDIT: Adam RHAE=0 on all 5 draws (same as TP).")
        print(f"  >>> These games are too hard for any 2K-step substrate.")
        print(f"  >>> Gap is budget or game difficulty, not update rule.")
    print("=" * 80)

    # Save summary
    summary = {
        'step':           STEP,
        'n_draws':        N_DRAWS,
        'draw_seeds':     DRAW_SEEDS,
        'rhae_per_draw':  rhae_per_draw,
        'chain_mean_rhae': chain_mean,
        'nonzero_draws':  nonzero_draws,
        'nonzero_games':  nonzero_games,
        'total_games':    total_games,
        'all_eff_sq':     all_eff_sq,
        'cr_per_draw':    cr_means,
        'baseline_1344':  {'rhae_per_draw': [0]*5, 'chain_mean': 0.0, 'nonzero_draws': 0},
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'diagnostics.json'), 'w') as f:
        json.dump({'step': STEP, 'results': all_results}, f, indent=2, default=str)


if __name__ == '__main__':
    main()
