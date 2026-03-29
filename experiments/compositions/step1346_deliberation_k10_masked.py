"""
Step 1346 — Deliberation through action decimation (K=10 vs K=1 control).
Leo mail 3828, 2026-03-29.

1345 eliminated credit depth: Adam RHAE=0 same as TP. Bottleneck is NOT the optimizer.
Problem: 2000 random actions can't solve hard games.

Jun's insight: RHAE counts actions, not computation. Internal processing is free.
Act every K=10 steps → 200 actions, 2000 computation steps.
Model gets 10 TP training updates between actions → better-trained when each action taken.

Conditions:
  K=10: act every 10 steps. 200 actions. Deliberation.
  K=1:  act every step.  2000 actions. Control (=1344 baseline).

Same 5 draw seeds as 1344/1345 (13440-13444). Direct comparison.

Protocol: 5 draws × 3 games × 2 conditions × 2 tries = 60 episodes. ~15 min.

Constitutional audit:
  R0: deterministic init ✓ | R1: self-supervised prediction ✓ | R2: TP local targets ✓
  (K only affects action frequency, not update mechanism)
  R6: K=1 is deletion control ✓
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

STEP        = 1346
TRY1_STEPS  = 2000     # computation steps per try
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 340      # slightly more headroom for 60 episodes

N_DRAWS     = 5
DRAW_SEEDS  = [13440, 13441, 13442, 13443, 13444]  # same as 1344/1345

CONDITIONS  = [('k10', 10), ('k1', 1)]  # (name, K)

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1346')

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN     = 512
_LR         = 1e-4
_TP_LR      = 0.01
_BUFFER_MAX = 200_000
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

    def forward_all_layers(self, x):
        h0 = F.relu(self.input_proj(x))
        h1 = F.relu(self.fc1(h0))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return h3, h2, h1, h0


# ---------------------------------------------------------------------------
# MLP-TP substrate with internal_train support
# ---------------------------------------------------------------------------

class MlpTpSubstrateK:
    """MLP + TP + entropy action selection. Supports K-step action decimation.

    _step counts COMPUTATION steps (both action steps and internal steps).
    action_steps are tracked externally by the episode runner.
    internal_train() advances _step and runs TP training.
    process() advances _step and selects an action.
    """

    def __init__(self, n_actions):
        self.n_actions       = n_actions
        self._rng            = np.random.RandomState(42)
        self._model          = None
        self._g              = None
        self._opt_pred       = None
        self._opt_g          = None
        self._opt_f          = None

        self._buffer         = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes  = set()
        self._prev_enc       = None
        self._train_counter  = 0   # for K=1 frequency gating
        self._step           = 0   # computation steps (action + internal)
        self._recent_losses  = deque(maxlen=50)
        self._pred_loss_at   = {ck: None for ck in LOSS_CHECKPOINTS}

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

    def process(self, obs_arr):
        """Action step: encode obs, select action. Advances _step by 1."""
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
        """Called after env.step(). Adds transition to buffer. K=1 trains here."""
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
        # K=1 frequency-gated training
        self._train_counter += 1
        if self._train_counter % 5 == 0 and len(self._buffer) >= _BATCH_SIZE:
            loss = self._tp_train_step()
            if loss is not None:
                self._recent_losses.append(loss)
                self._check_checkpoints()

    def internal_train(self):
        """Internal processing step (K>1 only). Trains on buffer. Advances _step by 1."""
        self._step += 1
        if len(self._buffer) >= _BATCH_SIZE:
            loss = self._tp_train_step()
            if loss is not None:
                self._recent_losses.append(loss)
                self._check_checkpoints()

    def _check_checkpoints(self):
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
# Episode runner (K-aware)
# ---------------------------------------------------------------------------

def run_episode_k(env, substrate, n_actions, seed, max_comp_steps, K):
    """
    Run episode for max_comp_steps computation steps.
    K=1: act every step (2000 action steps from 2000 comp steps).
    K=10: act every 10 steps (200 action steps from 2000 comp steps).

    Returns: (action_steps_to_first_progress, progress_count, elapsed_s)
    RHAE denominator = action_steps_to_first_progress (NOT comp steps).
    """
    obs            = env.reset(seed=seed)
    comp_step      = 0   # total computation steps
    action_step    = 0   # action steps (RHAE denominator)
    level          = 0
    progress_count = 0
    steps_to_first_progress = None
    t_start        = time.time()
    fresh_episode  = True

    while comp_step < max_comp_steps:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            level = 0
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)

        if comp_step % K == 0:
            # Action step
            action   = substrate.process(obs_arr) % n_actions
            obs_next, reward, done, info = env.step(action)
            action_step += 1
            comp_step   += 1

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
                    steps_to_first_progress = action_step
                level = cl
                substrate.on_level_transition()

            if done:
                obs = env.reset(seed=seed)
                substrate.on_level_transition()
                fresh_episode = True
                level = 0
            else:
                obs = obs_next
        else:
            # Internal processing step: train on replay buffer, no env interaction
            substrate.internal_train()
            comp_step += 1

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

    # results keyed by condition
    all_results   = {cname: [] for cname, _ in CONDITIONS}
    rhae_per_cond = {}

    for cname, K in CONDITIONS:
        try2_progress  = {}
        optimal_steps_d = {}
        cond_results   = []

        for game_name, label in zip(games, game_labels.values()):
            env = make_game(game_name)
            try:
                n_actions = int(env.n_actions)
            except AttributeError:
                n_actions = 4103

            substrate = MlpTpSubstrateK(n_actions=n_actions)

            p1, _, t1 = run_episode_k(env, substrate, n_actions, seed=0,
                                      max_comp_steps=TRY1_STEPS, K=K)
            substrate.reset_for_try2()
            p2, _, t2 = run_episode_k(env, substrate, n_actions, seed=TRY2_SEED,
                                      max_comp_steps=TRY2_STEPS, K=K)

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
                'condition': cname, 'K': K,
                'p1': p1, 'p2': p2, 'speedup': speedup,
                'eff_sq': eff_sq, 'optimal_steps': opt, 'cr': cr,
                't1': t1, 't2': t2,
            }
            cond_results.append(result)

            action_budget = TRY1_STEPS // K
            print(f"    [{cname} K={K} budget={action_budget}] {label}: "
                  f"speedup={format_speedup(speedup)}  eff²={eff_sq}  cr={cr}  ({t1+t2:.1f}s)")

            out_path = os.path.join(draw_dir, f'{cname}_{label_filename(label, STEP)}')
            with open(out_path, 'a') as f:
                f.write(json.dumps(result, default=str) + '\n')

        rhae     = compute_rhae_try2(try2_progress, optimal_steps_d)
        chain_sp = sum(speedup_for_chain(r['speedup']) for r in cond_results) / len(cond_results)
        print(f"    → [{cname}] RHAE(try2) = {rhae:.6f}  chain_speedup={chain_sp:.4f}")

        all_results[cname]  = cond_results
        rhae_per_cond[cname] = rhae

    return all_results, rhae_per_cond


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — Deliberation K=10 vs K=1 control. {N_DRAWS} draws.")
    print(f"Device: {_DEVICE}")
    print(f"Protocol: {N_DRAWS} draws × 3 games × 2 conditions × 2 tries = {N_DRAWS*3*2*2} episodes")
    print(f"Conditions: {CONDITIONS}. Comp steps=2K/try. Draw seeds: {DRAW_SEEDS}")
    print()

    # Tier 1: timing on first draw's first ARC game, K=10
    games0, labels0 = select_games(seed=DRAW_SEEDS[0])
    first_arc = next((g for g in games0 if not g.lower().startswith('mbpp')), None)
    if first_arc:
        first_label = labels0[first_arc]
        print(f"Tier 1: timing on {first_label} (100 comp steps, K=10)...")
        env_t = make_game(first_arc)
        try:
            na_t = int(env_t.n_actions)
        except AttributeError:
            na_t = 4103
        sub_t = MlpTpSubstrateK(n_actions=na_t)
        t0    = time.time()
        obs_t = env_t.reset(seed=0)
        for cs in range(100):
            if obs_t is None:
                obs_t = env_t.reset(seed=0)
                continue
            if cs % 10 == 0:
                a = sub_t.process(np.asarray(obs_t, dtype=np.float32)) % na_t
                obs_t, _, done_t, _ = env_t.step(a)
                if done_t or obs_t is None:
                    obs_t = env_t.reset(seed=0)
            else:
                sub_t.internal_train()
        elapsed_100 = time.time() - t0
        est_ep      = elapsed_100 / 100 * 2000
        est_total   = est_ep * N_DRAWS * 3 * 2 * 2  # 2 conditions
        print(f"  100 comp steps: {elapsed_100:.1f}s → est per episode: {est_ep:.0f}s → est total: {est_total:.0f}s")
        if est_total > 1200:
            print(f"  WARNING: est {est_total:.0f}s > 20-min budget.")
        print()

    # Run all draws
    all_draws_results = {cname: [] for cname, _ in CONDITIONS}
    rhae_matrix = {cname: [] for cname, _ in CONDITIONS}

    for draw_idx in range(N_DRAWS):
        draw_results, draw_rhae = run_draw(draw_idx, DRAW_SEEDS[draw_idx])
        for cname, _ in CONDITIONS:
            all_draws_results[cname].extend(draw_results[cname])
            rhae_matrix[cname].append(draw_rhae[cname])
        print()

    # -------------------------------------------------------------------------
    # Aggregate report
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(f"STEP {STEP} — RESULT (Deliberation K=10 vs K=1 control)")
    print()

    for cname, K in CONDITIONS:
        results    = all_draws_results[cname]
        rhae_draws = rhae_matrix[cname]
        cm         = sum(rhae_draws) / len(rhae_draws)
        nz_draws   = sum(1 for r in rhae_draws if r is not None and r > 0)
        nz_games   = sum(1 for r in results if r['eff_sq'] > 0)
        all_eff_sq = [r['eff_sq'] for r in results]
        print(f"  [{cname} K={K}]")
        print(f"    RHAE per draw: {[f'{r:.6f}' for r in rhae_draws]}")
        print(f"    Chain mean:    {cm:.6f}")
        print(f"    Non-zero:      draws={nz_draws}/{N_DRAWS}  games={nz_games}/15")
        print(f"    eff² max={max(all_eff_sq):.6f}  mean={sum(all_eff_sq)/len(all_eff_sq):.6f}")
        if nz_games > 0:
            for r in results:
                if r['eff_sq'] > 0:
                    print(f"      Draw {r['draw']} {r['label']}: eff²={r['eff_sq']}  p2={r['p2']}")
        print()

    # Decision
    k10_nz = sum(1 for r in rhae_matrix['k10'] if r is not None and r > 0)
    k1_nz  = sum(1 for r in rhae_matrix['k1']  if r is not None and r > 0)
    k10_cm = sum(rhae_matrix['k10']) / N_DRAWS
    k1_cm  = sum(rhae_matrix['k1'])  / N_DRAWS

    print("ASSESSMENT:")
    if k10_nz >= 2:
        print(f"  >>> DELIBERATION SIGNAL: K=10 non-zero draws={k10_nz}/5 (chain mean={k10_cm:.6f})")
        print(f"  >>> K=1 baseline: {k1_nz}/5 draws (chain mean={k1_cm:.6f})")
        print(f"  >>> Action decimation helps. Fewer, better-timed actions > more random actions.")
        print(f"  >>> Next: try K=10 + model-based action selection (informed actions).")
    elif k10_nz == 1:
        print(f"  >>> AMBIGUOUS: K=10 only {k10_nz}/5 draws non-zero (chain mean={k10_cm:.6f})")
        print(f"  >>> Need more draws to confirm deliberation effect.")
    else:
        print(f"  >>> NO DELIBERATION SIGNAL: K=10 RHAE=0 same as K=1.")
        print(f"  >>> 200 random actions on these games also can't reach progress.")
        print(f"  >>> Next: need informed actions (not just fewer random ones).")
        print(f"  >>> Try K=10 + model-based selection (deliberation × curiosity).")
    print("=" * 80)

    # Save summary
    summary = {
        'step':        STEP,
        'n_draws':     N_DRAWS,
        'draw_seeds':  DRAW_SEEDS,
        'conditions':  {cname: {'K': K, 'rhae_per_draw': rhae_matrix[cname],
                                'chain_mean': sum(rhae_matrix[cname])/N_DRAWS,
                                'nonzero_draws': sum(1 for r in rhae_matrix[cname]
                                                    if r is not None and r > 0)}
                        for cname, K in CONDITIONS},
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'diagnostics.json'), 'w') as f:
        json.dump({'step': STEP, 'results': {
            cname: all_draws_results[cname] for cname, _ in CONDITIONS
        }}, f, indent=2, default=str)


if __name__ == '__main__':
    main()
