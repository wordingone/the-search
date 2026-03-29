"""
Step 1348 — Prior knowledge through multi-game childhood.
Leo mail 3832, 2026-03-29.

Contingency: 1346 RHAE=0 AND 1347 RHAE=0 → met.

The substrate has zero prior knowledge. 14+ experiments (1344-1347) confirm:
no substrate can reach L1 on these games starting from scratch.

Hypothesis: train on many games BEFORE evaluation → accumulated cross-game features
(shared representations of ARC structure) → better on novel test game.

Childhood: 10 random ARC games × 500 steps + 3 MBPP × 500 steps (one-time).
Evaluation: 5 draws × 3 games × 2 tries using childhood-trained weights.
Fresh control: use 1344 results (RHAE=0 on all draws, no re-run).

Constitutional audit:
  R0: Childhood IS the dynamics overwriting initialization. R0 satisfied by definition.
  R1: Self-supervised prediction during childhood. No external reward. ✓
  R2: TP local targets during childhood. Same mechanism. ✓

Budget: ~3.3 min childhood + ~7.5 min eval = ~10.8 min. Within threshold.
"""
import sys, os, time, json, logging, hashlib, copy
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

STEP        = 1348
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 340

N_DRAWS      = 5
DRAW_SEEDS   = [13440, 13441, 13442, 13443, 13444]

# Childhood parameters
CHILDHOOD_STEPS    = 500      # steps per childhood game
N_CHILDHOOD_ARC    = 10       # ARC games for childhood
N_CHILDHOOD_MBPP   = 3        # MBPP problems for childhood
CHILDHOOD_BASE_SEED = 100     # seed range for childhood game selection (far from eval seeds)

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1348')

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN     = 512
_LR         = 1e-4
_TP_LR      = 0.01
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
# MLP model
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

    def forward_all_layers(self, x):
        h0 = F.relu(self.input_proj(x))
        h1 = F.relu(self.fc1(h0))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return h3, h2, h1, h0


# ---------------------------------------------------------------------------
# MLP-TP substrate (same as 1344) with snapshot/restore support
# ---------------------------------------------------------------------------

class MlpTpSubstrate:
    def __init__(self, n_actions):
        self.n_actions      = n_actions
        self._rng           = np.random.RandomState(42)
        self._model         = None
        self._g             = None
        self._opt_pred      = None
        self._opt_g         = None
        self._opt_f         = None

        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_enc      = None
        self._train_counter = 0
        self._step          = 0
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

    def _init_model(self, input_dim):
        self._model = MlpModel(input_dim, self.n_actions).to(_DEVICE)
        self._g = nn.ModuleDict({
            'g3': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g2': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g1': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
        })
        self._opt_pred = torch.optim.Adam(self._model.pred_head.parameters(), lr=_LR)
        self._opt_g    = {k: torch.optim.Adam(v.parameters(), lr=_LR)
                         for k, v in self._g.items()}
        self._opt_f    = {
            'f_proj': torch.optim.Adam(self._model.input_proj.parameters(), lr=_LR),
            'f1':     torch.optim.Adam(self._model.fc1.parameters(), lr=_LR),
            'f2':     torch.optim.Adam(self._model.fc2.parameters(), lr=_LR),
            'f3':     torch.optim.Adam(self._model.fc3.parameters(), lr=_LR),
        }

    def reset_for_try2(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def snapshot(self):
        """Snapshot model weights + buffer after childhood training."""
        if self._model is None:
            return None
        return {
            'model':         copy.deepcopy(self._model.state_dict()),
            'g':             copy.deepcopy(self._g.state_dict()),
            'buffer':        copy.deepcopy(self._buffer),
            'buffer_hashes': copy.deepcopy(self._buffer_hashes),
            'train_counter': self._train_counter,
        }

    def restore_from_snapshot(self, snap):
        """Restore to childhood weights + buffer. Reset episode state."""
        if snap is None or self._model is None:
            return
        self._model.load_state_dict(snap['model'])
        self._g.load_state_dict(snap['g'])
        self._buffer        = copy.deepcopy(snap['buffer'])
        self._buffer_hashes = copy.deepcopy(snap['buffer_hashes'])
        self._train_counter = snap['train_counter']
        # Reset episode state
        self._step          = 0
        self._prev_enc      = None
        self._recent_losses.clear()
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

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
            loss = self._tp_train_step()
            if loss is not None:
                self._recent_losses.append(loss)
                for ck in self._pred_loss_at:
                    if self._pred_loss_at[ck] is None and self._step >= ck:
                        self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

    def on_level_transition(self):
        self._prev_enc = None

    def _tp_train_step(self):
        n   = len(self._buffer); buf = list(self._buffer)
        idx = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf[i] for i in idx]

        states      = torch.from_numpy(np.stack([b['state'].astype(np.float32)
                                                 for b in batch])).to(_DEVICE)
        next_states = torch.from_numpy(np.stack([b['next_state'].astype(np.float32)
                                                 for b in batch])).to(_DEVICE)
        action_vecs = torch.from_numpy(np.stack([b['action_type_vec']
                                                 for b in batch])).to(_DEVICE)

        with torch.no_grad():
            h3, h2, h1, h0 = self._model.forward_all_layers(states)
            _, target_h3_next = self._model(next_states)
            pred_in   = torch.cat([h3, action_vecs], dim=1)
            pred_next = self._model.pred_head(pred_in)
            pred_err  = pred_next - target_h3_next
            pred_loss = float((pred_err ** 2).mean())
            target_h3 = h3 - _TP_LR * pred_err
            target_h2 = self._g['g3'](target_h3)
            target_h1 = self._g['g2'](target_h2)
            target_h0 = self._g['g1'](target_h1)

        with torch.enable_grad():
            pred_n2  = self._model.pred_head(
                torch.cat([h3.detach(), action_vecs.detach()], dim=1))
            loss_pred = F.mse_loss(pred_n2, target_h3_next.detach())
            self._opt_pred.zero_grad(); loss_pred.backward(); self._opt_pred.step()

        for gk, h_in, h_out in [('g3', h3, h2), ('g2', h2, h1), ('g1', h1, h0)]:
            with torch.enable_grad():
                hr = self._g[gk](h_in.detach())
                lg = F.mse_loss(hr, h_out.detach())
                self._opt_g[gk].zero_grad(); lg.backward(); self._opt_g[gk].step()

        for fk, layer, x_in, target in [
            ('f3',     self._model.fc3,        h2,     target_h3),
            ('f2',     self._model.fc2,        h1,     target_h2),
            ('f1',     self._model.fc1,        h0,     target_h1),
            ('f_proj', self._model.input_proj, states, target_h0),
        ]:
            with torch.enable_grad():
                hf = F.relu(layer(x_in.detach()))
                lf = F.mse_loss(hf, target.detach())
                self._opt_f[fk].zero_grad(); lf.backward(); self._opt_f[fk].step()

        return pred_loss

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
# Episode runners
# ---------------------------------------------------------------------------

def run_childhood_episode(env, substrate, n_actions, seed, steps):
    """Run a childhood game episode. No progress tracking — just train."""
    obs   = env.reset(seed=seed)
    step  = 0
    fresh = True
    t0    = time.time()
    while step < steps:
        if time.time() - t0 > 60:  # 1 min max per childhood game
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh = True
            continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action  = substrate.process(obs_arr) % n_actions
        obs_next, reward, done, info = env.step(action)
        step += 1
        if obs_next is not None:
            substrate.update_after_step(obs_next, action, reward)
        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh = True
        else:
            obs = obs_next if obs_next is not None else obs
    return step, round(time.time() - t0, 2)


def run_eval_episode(env, substrate, n_actions, seed, max_steps):
    """Standard evaluation episode (same as 1344)."""
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
# Childhood game selection
# ---------------------------------------------------------------------------

def select_childhood_arc_games():
    """Select N_CHILDHOOD_ARC ARC games for childhood (deterministic)."""
    games = []
    seed  = CHILDHOOD_BASE_SEED
    while len(games) < N_CHILDHOOD_ARC:
        cands, _ = select_games(seed=seed)
        for g in cands:
            if not g.lower().startswith('mbpp') and len(games) < N_CHILDHOOD_ARC:
                games.append(g)
        seed += 1
    return games


# ---------------------------------------------------------------------------
# Childhood training phase
# ---------------------------------------------------------------------------

def run_childhood_phase():
    """Train on N_CHILDHOOD_ARC ARC games + N_CHILDHOOD_MBPP MBPP problems.
    Returns: (arc_snapshot, mbpp_snapshot, stats)

    arc_snapshot: snapshot of ARC substrate after childhood (65536-dim)
    mbpp_snapshot: snapshot of MBPP substrate after childhood (variable-dim)
    """
    print("  [Childhood] ARC phase:")
    arc_games = select_childhood_arc_games()
    arc_sub   = MlpTpSubstrate(n_actions=4103)

    for i, game_name in enumerate(arc_games):
        env = make_game(game_name)
        steps_done, elapsed = run_childhood_episode(env, arc_sub, 4103,
                                                     seed=i, steps=CHILDHOOD_STEPS)
        print(f"    ARC game {i+1}/{N_CHILDHOOD_ARC}: {steps_done} steps in {elapsed:.1f}s"
              f"  buf={len(arc_sub._buffer)}")

    arc_snap = arc_sub.snapshot()
    print(f"  [Childhood] ARC done. Buffer size: {len(arc_sub._buffer)}")

    print("  [Childhood] MBPP phase:")
    mbpp_sub  = None
    mbpp_snap = None
    for i in range(N_CHILDHOOD_MBPP):
        game_name = f'mbpp_{i}'
        try:
            env = make_game(game_name)
        except Exception:
            env = make_game('mbpp')
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = 128
        if mbpp_sub is None:
            mbpp_sub = MlpTpSubstrate(n_actions=n_actions)
        steps_done, elapsed = run_childhood_episode(env, mbpp_sub, n_actions,
                                                     seed=i, steps=CHILDHOOD_STEPS)
        print(f"    MBPP prob {i+1}/{N_CHILDHOOD_MBPP}: {steps_done} steps in {elapsed:.1f}s"
              f"  buf={len(mbpp_sub._buffer)}")

    if mbpp_sub is not None:
        mbpp_snap = mbpp_sub.snapshot()
    print(f"  [Childhood] MBPP done.")

    return arc_snap, mbpp_snap


# ---------------------------------------------------------------------------
# Evaluation draw runner
# ---------------------------------------------------------------------------

def run_draw(draw_idx, draw_seed, arc_snap, mbpp_snap):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    print(f"  Draw {draw_idx} (seed={draw_seed}): {masked_game_list(game_labels)}")
    draw_results    = []
    try2_progress   = {}
    optimal_steps_d = {}

    for game_name, label in zip(games, game_labels.values()):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = 4103

        is_mbpp = game_name.lower().startswith('mbpp')

        # Create substrate with childhood weights (or fresh if snapshot unavailable)
        substrate = MlpTpSubstrate(n_actions=n_actions)

        # Warm-start: run one obs through process() to init model, then restore snapshot
        # We need to init the model first (requires seeing an obs), THEN restore
        init_obs = env.reset(seed=0)
        if init_obs is not None:
            init_enc = _encode_obs_mlp(np.asarray(init_obs, dtype=np.float32))
            substrate._init_model(init_enc.shape[0])

        # Now restore childhood snapshot if available and compatible
        snap = mbpp_snap if is_mbpp else arc_snap
        if snap is not None and substrate._model is not None:
            try:
                substrate.restore_from_snapshot(snap)
                print(f"    {label}: restored childhood weights (buf={len(substrate._buffer)})")
            except Exception as e:
                print(f"    {label}: snapshot restore failed ({e}), using fresh weights")

        # Run evaluation (try1 then try2 with restored snapshot for try2)
        # For try2: restore snapshot again (independent of try1's experience)
        p1, _, t1 = run_eval_episode(env, substrate, n_actions, seed=0, max_steps=TRY1_STEPS)

        # Restore for try2 (fresh childhood weights, independent of try1)
        substrate.reset_for_try2()
        if snap is not None and substrate._model is not None:
            try:
                # Only reset episode-level state; keep model weights from try1 end
                # (try2 continues learning from try1 — same as 1344 protocol)
                pass
            except Exception:
                pass

        p2, _, t2 = run_eval_episode(env, substrate, n_actions, seed=TRY2_SEED,
                                     max_steps=TRY2_STEPS)

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
    print(f"Step {STEP} — Childhood pre-training. {N_DRAWS} eval draws.")
    print(f"Device: {_DEVICE}")
    print(f"Childhood: {N_CHILDHOOD_ARC} ARC × {CHILDHOOD_STEPS} steps + "
          f"{N_CHILDHOOD_MBPP} MBPP × {CHILDHOOD_STEPS} steps = "
          f"{N_CHILDHOOD_ARC + N_CHILDHOOD_MBPP} games total")
    print(f"Evaluation: {N_DRAWS} draws × 3 games × 2 tries = {N_DRAWS*3*2} episodes")
    print(f"Draw seeds: {DRAW_SEEDS}")
    print()

    # Tier 1: timing on a childhood ARC game (100 steps)
    childhood_games = select_childhood_arc_games()
    print(f"Tier 1: timing on childhood ARC game 1 (100 steps)...")
    env_t = make_game(childhood_games[0])
    sub_t = MlpTpSubstrate(n_actions=4103)
    t0    = time.time()
    obs_t = env_t.reset(seed=0)
    for _ in range(100):
        if obs_t is None:
            obs_t = env_t.reset(seed=0)
            continue
        a = sub_t.process(np.asarray(obs_t, dtype=np.float32)) % 4103
        obs_t, _, done_t, _ = env_t.step(a)
        if done_t or obs_t is None:
            obs_t = env_t.reset(seed=0)
    elapsed_100 = time.time() - t0
    est_childhood = (elapsed_100 / 100) * CHILDHOOD_STEPS * (N_CHILDHOOD_ARC + N_CHILDHOOD_MBPP)
    est_eval      = (elapsed_100 / 100) * TRY1_STEPS * N_DRAWS * 3 * 2
    est_total     = est_childhood + est_eval
    print(f"  100 steps: {elapsed_100:.1f}s → childhood: {est_childhood:.0f}s → "
          f"eval: {est_eval:.0f}s → total: {est_total:.0f}s")
    if est_total > 900:
        print(f"  WARNING: est {est_total:.0f}s > 15-min budget.")
    print()

    # Childhood phase (one-time)
    print("CHILDHOOD PHASE (one-time, weights used for all eval draws):")
    t_childhood = time.time()
    arc_snap, mbpp_snap = run_childhood_phase()
    childhood_time = round(time.time() - t_childhood, 1)
    print(f"  Childhood complete in {childhood_time}s")
    print()

    # Evaluation phase
    all_results   = []
    rhae_per_draw = []

    for draw_idx in range(N_DRAWS):
        draw_results, draw_rhae = run_draw(draw_idx, DRAW_SEEDS[draw_idx],
                                           arc_snap, mbpp_snap)
        all_results.extend(draw_results)
        rhae_per_draw.append(draw_rhae)
        print()

    # Aggregate report
    chain_mean    = sum(rhae_per_draw) / len(rhae_per_draw)
    nonzero_draws = sum(1 for r in rhae_per_draw if r is not None and r > 0)
    nonzero_games = sum(1 for r in all_results if r['eff_sq'] > 0)
    all_eff_sq    = [r['eff_sq'] for r in all_results]

    print("=" * 80)
    print(f"STEP {STEP} — RESULT (Childhood pre-training vs Fresh control)")
    print()
    print(f"  CHILDHOOD RHAE per draw: {[f'{r:.6f}' for r in rhae_per_draw]}")
    print(f"  Chain mean:              {chain_mean:.6f}")
    print(f"  Non-zero draws:          {nonzero_draws}/{N_DRAWS}")
    print(f"  Non-zero games:          {nonzero_games}/15")
    print(f"  eff² max={max(all_eff_sq):.6f}  mean={sum(all_eff_sq)/len(all_eff_sq):.6f}")
    print()
    print("  Games with progress:")
    for r in all_results:
        if r['eff_sq'] > 0:
            print(f"    Draw {r['draw']} {r['label']}: eff²={r['eff_sq']}  p2={r['p2']}  cr={r['cr']}")
    print()
    print("  Fresh baseline (1344, same seeds):")
    print("    RHAE=[0,0,0,0,0] chain=0. Non-zero: 0/5 draws.")
    print()

    print("ASSESSMENT:")
    if nonzero_draws >= 2:
        print(f"  >>> CHILDHOOD SIGNAL: {nonzero_draws}/5 draws non-zero (chain={chain_mean:.6f})")
        print(f"  >>> Prior experience across games helps. First RHAE > 0 from non-trivial substrate.")
        print(f"  >>> Next: scale childhood (more games, more steps). OR combine with deliberation.")
    elif nonzero_draws == 1:
        print(f"  >>> POSSIBLE SIGNAL: 1/5 draws non-zero (chain={chain_mean:.6f})")
        print(f"  >>> Ambiguous. Need more draws or larger childhood phase.")
    else:
        print(f"  >>> NO CHILDHOOD SIGNAL: RHAE=0 same as fresh.")
        print(f"  >>> 10-game childhood doesn't build useful priors in 2K eval steps.")
        print(f"  >>> Options: (a) more childhood games, (b) different architecture,")
        print(f"  >>>          (c) the problem requires in-context adaptation, not prior knowledge.")
    print("=" * 80)

    summary = {
        'step':             STEP,
        'n_draws':          N_DRAWS,
        'draw_seeds':       DRAW_SEEDS,
        'n_childhood_arc':  N_CHILDHOOD_ARC,
        'n_childhood_mbpp': N_CHILDHOOD_MBPP,
        'childhood_steps':  CHILDHOOD_STEPS,
        'rhae_per_draw':    rhae_per_draw,
        'chain_mean':       chain_mean,
        'nonzero_draws':    nonzero_draws,
        'nonzero_games':    nonzero_games,
        'all_eff_sq':       all_eff_sq,
        'fresh_baseline':   {'rhae_per_draw': [0]*5, 'chain_mean': 0.0, 'step': 1344},
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'diagnostics.json'), 'w') as f:
        json.dump({'step': STEP, 'results': all_results}, f, indent=2, default=str)


if __name__ == '__main__':
    main()
