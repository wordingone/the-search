"""
Step 1392 — Dendritic spatial subunits + Izhikevich + calcium STDP.
Leo mail 3985/3987 (revision), 2026-03-30. Gate check: all 11 gates passed.

Architecture: 64 dendritic subunits, each owns one 8×8 spatial patch of the 64×64 ARC obs.
Each subunit: Izhikevich dynamics driven by (obs_delta, act_input).
- obs_delta: mean abs pixel change in this patch from previous obs
- act_input: 1.0 if click action targeted this patch, 0.0 otherwise (0 in MASKED cond)
STDP: click = pre-synaptic event (CA_PRE), subunit spike = post-synaptic (CA_POST).
BCM sliding threshold prevents runaway potentiation.

Why spatial locality bridges the semantic gap (Leo mail 3987):
- Interactive object at patch i: click→patch changes→act+obs co-fire→LTP→W[i] grows
- Animation at patch i: patch changes without click→obs fires alone→no LTP at W[i]
Global animations change many subunits; local interactions change ONE clicked subunit.

Mandatory diagnostic: SPIKE vs SPIKE-MASKED (3 draws, ~seconds).
- Compare per-subunit firing rate patterns after try1.
- pattern_diff = mean(|FR_SPIKE - FR_MASKED|) / (mean(FR_MASKED) + eps)
- Pass if pattern_diff > 0.05. Fail → DIAGNOSTIC_FAIL, experiment aborted.

Try1: random uniform actions (broadest STDP exploration).
Try2 SPIKE: softmax(W[64]) → click target patch → random (x,y) within patch.
Try2 RANDOM: random actions (control).

Seeds: 14410-14439. 30 draws.
"""

import os, sys, json, time, math
import numpy as np

sys.path.insert(0, 'B:/M/the-search/experiments/compositions')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')
sys.path.insert(0, 'B:/M/the-search')

from prism_masked import (
    select_games, seal_mapping, label_filename, det_weights,
    compute_progress_speedup, compute_rhae_try2,
    mask_result_row, ARC_OPTIMAL_STEPS_PROXY,
    masked_game_list,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STEP        = 1392
N_DRAWS     = 30
DRAW_SEEDS  = [14410 + i for i in range(N_DRAWS)]
N_DIAG_INIT = 3
TRY1_STEPS  = 4000
TRY2_STEPS  = 4000
TRY2_SEED   = 4
MAX_SECONDS = 280
TIER1_STEPS = 200

RESULTS_DIR     = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')
MLP_TP_BASELINE = 4.59e-5

# Spatial layout
N_SUBUNITS    = 64
PATCH_GRID    = 8    # patches per side
PATCH_SIZE    = 8    # pixels per patch side
ARC_SIZE      = 64   # ARC obs spatial size
MAX_N_ACTIONS = 4103  # 7 non-click + 64*64 click

# Izhikevich (regular spiking, Izhikevich 2003)
IZH_A = 0.02
IZH_B = 0.2
IZH_C = -65.0
IZH_D = 8.0
V_THRESH = 30.0

# Input scaling to Izhikevich current range (~5-15 for firing)
OBS_SCALE = 8.0   # obs_delta (0-1) → current
ACT_SCALE = 8.0   # act_input (0 or 1) → current

# STDP calcium (Shouval 2002 inspired)
CA_PRE   = 0.2
CA_POST  = 0.6
CA_DECAY = 0.95
W_LTP    = 0.01
W_LTD    = 0.005
W_MAX    = 20.0

# BCM sliding threshold (Bienenstock-Cooper-Munro 1982)
THETA_HIGH_INIT = 0.8
THETA_LOW       = 0.3
BCM_TARGET      = 2.0   # target spikes per 100 steps
BCM_ALPHA       = 0.3

# Action selection (try2)
TEMPERATURE = 3.0

# Diagnostic
ACTION_BLIND_THRESHOLD = 0.05  # pattern_diff > 5% → action-conditional

CONDITIONS = ['SPIKE', 'RANDOM']
DIAG_COND  = 'SPIKE-MASKED'


# ---------------------------------------------------------------------------
# Game helpers
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
# Obs / action helpers
# ---------------------------------------------------------------------------

def _is_arc_obs(arr):
    return arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] == 64 and arr.shape[2] == 64


def _extract_patches(obs_arr):
    """Extract mean normalized pixel value per 8×8 patch. Returns (64,) or None."""
    try:
        arr = np.asarray(obs_arr, dtype=np.float32)
        if not _is_arc_obs(arr):
            return None
        frame = arr.squeeze(0) / 15.0  # (64, 64), 0-1
        patches = np.zeros(N_SUBUNITS, dtype=np.float32)
        for i in range(N_SUBUNITS):
            r = (i // PATCH_GRID) * PATCH_SIZE
            c = (i % PATCH_GRID) * PATCH_SIZE
            patches[i] = frame[r:r+PATCH_SIZE, c:c+PATCH_SIZE].mean()
        return patches
    except Exception:
        return None


def _action_to_patch(action):
    """Return patch_idx (0-63) for click action, -1 for non-click."""
    if action < 7:
        return -1
    click_idx = action - 7
    x = click_idx % ARC_SIZE
    y = click_idx // ARC_SIZE
    patch_col = min(x // PATCH_SIZE, PATCH_GRID - 1)
    patch_row = min(y // PATCH_SIZE, PATCH_GRID - 1)
    return patch_row * PATCH_GRID + patch_col


def binomial_p_one_sided(wins, n):
    if n <= 0:
        return 1.0
    from math import comb
    p = 0.0
    for k in range(wins, n + 1):
        p += comb(n, k) * (0.5 ** n)
    return p


# ---------------------------------------------------------------------------
# Dendritic subunit
# ---------------------------------------------------------------------------

class DendriticSubunit:
    __slots__ = ['patch_idx', 'v', 'u', 'ca', 'w',
                 'spike_count', 'theta_h', '_recent_spikes', '_bcm_win']

    def __init__(self, patch_idx):
        self.patch_idx = patch_idx
        self.v = IZH_C
        self.u = IZH_B * IZH_C
        self.ca = 0.0
        self.w = 0.0
        self.spike_count = 0
        self.theta_h = THETA_HIGH_INIT
        self._recent_spikes = 0
        self._bcm_win = 0

    def reset_state(self):
        """Reset membrane/calcium for try2, keep learned W."""
        self.v = IZH_C
        self.u = IZH_B * IZH_C
        self.ca = 0.0
        self.spike_count = 0
        self._recent_spikes = 0
        self._bcm_win = 0

    def step(self, obs_delta, act_in):
        """One Euler step (dt=0.5ms×2 for stability). Returns True if fired."""
        I = float(obs_delta) * OBS_SCALE + float(act_in) * ACT_SCALE

        fired = False
        for _ in range(2):
            dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I) * 0.5
            du = (IZH_A * (IZH_B * self.v - self.u)) * 0.5
            self.v += dv
            self.u += du
            if self.v >= V_THRESH:
                self.v = IZH_C
                self.u += IZH_D
                fired = True
                break

        self.v = max(-100.0, min(35.0, self.v))  # safety clamp

        # STDP calcium dynamics
        if act_in > 0:
            self.ca += CA_PRE   # pre-synaptic: click on this patch
        if fired:
            self.ca += CA_POST  # post-synaptic: spike
            self.spike_count += 1
            self._recent_spikes += 1
        self.ca *= CA_DECAY

        # Weight update from calcium
        if self.ca > self.theta_h:
            self.w = min(W_MAX, self.w + W_LTP)
        elif self.ca > THETA_LOW:
            self.w = max(0.0, self.w - W_LTD)

        # BCM threshold update every 100 steps
        self._bcm_win += 1
        if self._bcm_win >= 100:
            avg_rate = self._recent_spikes / 100.0
            self.theta_h = max(0.3, min(1.5,
                0.5 + BCM_ALPHA * (avg_rate / max(BCM_TARGET, 1e-6))))
            self._recent_spikes = 0
            self._bcm_win = 0

        return fired


# ---------------------------------------------------------------------------
# Dendritic substrate
# ---------------------------------------------------------------------------

class DendriticSubstrate:
    def __init__(self, n_actions, mode='spike'):
        """
        mode: 'spike'        — full STDP + action-guided try2
              'spike-masked' — STDP but act_input zeroed (diagnostic)
              'random'       — random actions, no STDP (control)
        """
        self.n_actions = n_actions
        self.mode = mode
        self._masked = (mode == 'spike-masked')
        self._random = (mode == 'random')
        self._try2 = False
        self._rng = np.random.RandomState(0)
        self._prev_patches = None
        self._step = 0
        self._visit_count = np.zeros(n_actions, dtype=np.int32)
        self.subunits = [DendriticSubunit(i) for i in range(N_SUBUNITS)]

    def process(self, obs_arr):
        arr = np.asarray(obs_arr, dtype=np.float32)
        patches = _extract_patches(arr)
        if patches is not None and self._prev_patches is None:
            self._prev_patches = patches.copy()

        if self._random or not self._try2:
            action = int(self._rng.randint(self.n_actions))
        else:
            action = self._select_try2()

        self._visit_count[action % self.n_actions] += 1
        return action

    def _select_try2(self):
        """Softmax over STDP weights → click target patch."""
        w = np.array([s.w for s in self.subunits])
        if w.max() < 1e-8:
            return int(self._rng.randint(self.n_actions))
        w_s = w / TEMPERATURE
        w_e = np.exp(w_s - w_s.max())
        probs = w_e / w_e.sum()
        patch_idx = int(self._rng.choice(N_SUBUNITS, p=probs))
        r = patch_idx // PATCH_GRID
        c = patch_idx % PATCH_GRID
        x = c * PATCH_SIZE + self._rng.randint(PATCH_SIZE)
        y = r * PATCH_SIZE + self._rng.randint(PATCH_SIZE)
        click_idx = y * ARC_SIZE + x
        action = 7 + click_idx
        return min(action, self.n_actions - 1)

    def update_after_step(self, obs_next, action, reward):
        if self._random:
            return  # no STDP for control
        if obs_next is None:
            self._prev_patches = None
            return

        arr = np.asarray(obs_next, dtype=np.float32)
        patches = _extract_patches(arr)

        if patches is not None and self._prev_patches is not None:
            deltas = np.abs(patches - self._prev_patches)
        else:
            deltas = np.zeros(N_SUBUNITS, dtype=np.float32)

        # Action input: 1.0 for the clicked patch
        act_inputs = np.zeros(N_SUBUNITS, dtype=np.float32)
        if not self._masked:
            patch_clicked = _action_to_patch(action % self.n_actions)
            if patch_clicked >= 0:
                act_inputs[patch_clicked] = 1.0

        for i, sub in enumerate(self.subunits):
            sub.step(deltas[i], act_inputs[i])

        if patches is not None:
            self._prev_patches = patches.copy()
        self._step += 1

    def on_level_transition(self, new_level=0):
        pass  # STDP accumulates across levels

    def prepare_for_try2(self):
        for sub in self.subunits:
            sub.reset_state()
        self._try2 = True
        self._prev_patches = None
        self._step = 0
        self._visit_count[:] = 0

    def get_stats(self):
        w = np.array([s.w for s in self.subunits])
        spike_counts = np.array([s.spike_count for s in self.subunits])
        return {
            'w_norm':    float(np.linalg.norm(w)),
            'w_max':     float(w.max()),
            'w_mean':    float(w.mean()),
            'spike_total': int(spike_counts.sum()),
            'spike_counts': spike_counts.tolist(),
        }

    def compute_stage_metrics(self):
        vc = self._visit_count.astype(float)
        i3_cv = float(np.std(vc) / (np.mean(vc) + 1e-8)) if vc.sum() > 0 else 0.0
        w = np.array([s.w for s in self.subunits])
        return {
            'i3_cv': i3_cv,
            'r3_w_norm': float(np.linalg.norm(w)),
            'i5_max_level': 0,
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, seed, max_steps):
    obs = env.reset(seed=seed)
    steps = 0
    level = 0
    max_level = 0
    steps_to_first_progress = None
    t_start = time.time()
    fresh_episode = True

    while steps < max_steps:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition(new_level=0)
            level = 0
            fresh_episode = True
            continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr) % n_actions
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
            if steps_to_first_progress is None:
                steps_to_first_progress = steps
            level = cl
            max_level = max(max_level, cl)
            substrate.on_level_transition(new_level=cl)
        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition(new_level=0)
            level = 0
            fresh_episode = True
        else:
            obs = obs_next

    return steps_to_first_progress, round(time.time() - t_start, 2), max_level


# ---------------------------------------------------------------------------
# Draw runner
# ---------------------------------------------------------------------------

def run_draw(draw_idx, draw_seed, cond_name, max_steps, return_try1_spikes=False):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, cond_name, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    draw_results = []
    try2_progress = {}
    optimal_steps_d = {}
    mode = cond_name.lower()
    all_try1_spikes = []

    for game_idx, (game_name, label) in enumerate(zip(games, game_labels.values())):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = MAX_N_ACTIONS

        substrate = DendriticSubstrate(n_actions=n_actions, mode=mode)
        substrate._rng = np.random.RandomState(draw_seed * 100 + game_idx)

        p1, t1, ml1 = run_episode(env, substrate, n_actions, seed=0, max_steps=max_steps)
        try1_stats = substrate.get_stats()
        if return_try1_spikes:
            all_try1_spikes.append(np.array(try1_stats['spike_counts'], dtype=float))

        substrate.prepare_for_try2()
        np.random.seed(draw_seed * 1000 + 1)  # PRNG fix: independent try2 RNG
        p2, t2, ml2 = run_episode(env, substrate, n_actions, seed=TRY2_SEED, max_steps=max_steps)

        stage = substrate.compute_stage_metrics()
        stage['i5_max_level'] = ml2

        speedup = compute_progress_speedup(p1, p2)
        opt = get_optimal_steps(game_name, TRY2_SEED)
        eff_sq = 0.0
        if p2 is not None and opt is not None and opt > 0:
            eff_sq = round(min(1.0, opt / p2) ** 2, 6)

        try2_progress[label] = p2
        optimal_steps_d[label] = opt

        row = {
            'draw': draw_idx, 'label': label, 'game': game_name,
            'condition': cond_name,
            'p1': p1, 'p2': p2, 'speedup': speedup,
            'eff_sq': eff_sq, 'optimal_steps': opt,
            't1': t1, 't2': t2,
            'w_norm': try1_stats['w_norm'],
            'w_max':  try1_stats['w_max'],
            'spike_total': try1_stats['spike_total'],
            'i3_cv':  stage['i3_cv'],
            'r3_w_norm': stage['r3_w_norm'],
            'i5_max_level': stage['i5_max_level'],
        }
        masked_row = mask_result_row(row, game_labels)
        fn = os.path.join(draw_dir, label_filename(label, STEP))
        with open(fn, 'w') as f:
            f.write(json.dumps(masked_row) + '\n')
        draw_results.append(masked_row)

    rhae = compute_rhae_try2(try2_progress, optimal_steps_d)
    w_norm_mean = float(np.mean([r['w_norm'] for r in draw_results]))
    print(f"  [{cond_name}] Draw {draw_idx} RHAE={rhae:.6e}  w_norm={w_norm_mean:.4f}")

    if return_try1_spikes:
        return round(rhae, 7), w_norm_mean, draw_results, all_try1_spikes
    return round(rhae, 7), w_norm_mean, draw_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Tier 1: timing check ──────────────────────────────────────────────
    print("=== TIER 1: timing check ===")
    games_t1, _ = select_games(seed=DRAW_SEEDS[0])
    env_t1 = make_game(games_t1[0])
    try:
        na_t1 = int(env_t1.n_actions)
    except AttributeError:
        na_t1 = MAX_N_ACTIONS

    sub_t1 = DendriticSubstrate(n_actions=na_t1, mode='spike')
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)   # warmup
    t0 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t0

    ms_per_step = tier1_elapsed / TIER1_STEPS * 1000
    # Estimate: diag (3 draws × 3 games × 2 cond × 2 ep) + full (30 × 3 × 2 × 2) - reuse
    n_eps = N_DIAG_INIT * 3 * 2 * 2 + (N_DRAWS - N_DIAG_INIT) * 3 * 2 * 2 + N_DIAG_INIT * 3 * 2
    est_total_s = ms_per_step / 1000 * TRY1_STEPS * n_eps
    print(f"  {TIER1_STEPS} steps: {tier1_elapsed:.2f}s  ({ms_per_step:.2f}ms/step)")
    print(f"  Estimated total: {est_total_s:.0f}s ({est_total_s/60:.1f} min)")

    if est_total_s > MAX_SECONDS:
        max_steps = max(200, int(
            (MAX_SECONDS * 0.85) / (ms_per_step / 1000 * n_eps)
        ))
        print(f"  Budget exceeded — capping at {max_steps} steps")
    else:
        max_steps = TRY1_STEPS
        print(f"  Under budget — full {max_steps} steps")

    # ── Mandatory diagnostic: SPIKE vs SPIKE-MASKED ───────────────────────
    print(f"\n=== MANDATORY DIAGNOSTIC: {N_DIAG_INIT} draws SPIKE vs SPIKE-MASKED ===")
    diag_spike_rhae  = []
    diag_spike_spikes = []   # list of per-subunit spike_count arrays (3*3 games)
    diag_msk_spikes  = []

    t_diag = time.time()
    for di in range(N_DIAG_INIT):
        ds = DRAW_SEEDS[di]
        r_sp, _, _, spk_sp = run_draw(di, ds, 'SPIKE',        max_steps, return_try1_spikes=True)
        _,    _, _, spk_mk = run_draw(di, ds, 'SPIKE-MASKED', max_steps, return_try1_spikes=True)
        diag_spike_rhae.append(r_sp)
        diag_spike_spikes.extend(spk_sp)
        diag_msk_spikes.extend(spk_mk)
    print(f"  Diagnostic done in {time.time()-t_diag:.0f}s")

    # Compute per-game pattern_diff
    pattern_diffs = []
    for sp, mk in zip(diag_spike_spikes, diag_msk_spikes):
        baseline = mk.mean() + 1e-8
        diff = float(np.abs(sp - mk).mean() / baseline)
        pattern_diffs.append(diff)
    pattern_diff = float(np.mean(pattern_diffs)) if pattern_diffs else 0.0

    print(f"\n  Per-game pattern_diffs: {[round(d, 4) for d in pattern_diffs]}")
    print(f"  Mean pattern_diff (SPIKE vs MASKED): {pattern_diff:.4f}")

    if pattern_diff < ACTION_BLIND_THRESHOLD:
        print(f"\n*** DIAGNOSTIC FAIL: pattern_diff={pattern_diff:.4f} < {ACTION_BLIND_THRESHOLD} ***")
        print("Act_input has no effect on firing patterns — spatial subunits also action-blind.")
        summary = {
            'step': STEP, 'verdict': 'DIAGNOSTIC_FAIL',
            'pattern_diff': round(pattern_diff, 4),
            'per_game_diffs': [round(d, 4) for d in pattern_diffs],
            'note': 'Full experiment aborted — dendritic spatial subunits action-blind.',
        }
        with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        return

    print(f"\n  *** DIAGNOSTIC PASS: pattern_diff={pattern_diff:.4f} > {ACTION_BLIND_THRESHOLD} ***")
    print(f"  Act_input modulates firing patterns — spatial subunits ARE action-conditional.")
    print(f"  Proceeding to full {N_DRAWS}-draw experiment.")

    # ── Full experiment: SPIKE vs RANDOM, 30 draws ────────────────────────
    print(f"\n=== STEP {STEP}: SPIKE vs RANDOM, {N_DRAWS} draws, {max_steps} steps ===")
    print(f"Seeds: {DRAW_SEEDS[0]}-{DRAW_SEEDS[-1]}")

    rhae_by_cond = {c: [] for c in CONDITIONS}

    # Reuse SPIKE diagnostic draws; run RANDOM for first N_DIAG_INIT draws
    for di in range(N_DIAG_INIT):
        rhae_by_cond['SPIKE'].append(diag_spike_rhae[di])
        r_rd, _, _ = run_draw(di, DRAW_SEEDS[di], 'RANDOM', max_steps)
        rhae_by_cond['RANDOM'].append(r_rd)

    # Remaining draws
    for di in range(N_DIAG_INIT, N_DRAWS):
        r_sp, _, _ = run_draw(di, DRAW_SEEDS[di], 'SPIKE',  max_steps)
        r_rd, _, _ = run_draw(di, DRAW_SEEDS[di], 'RANDOM', max_steps)
        rhae_by_cond['SPIKE'].append(r_sp)
        rhae_by_cond['RANDOM'].append(r_rd)

    # Statistics
    spike_rhae = np.array(rhae_by_cond['SPIKE'])
    rand_rhae  = np.array(rhae_by_cond['RANDOM'])

    wins   = int(np.sum(spike_rhae > rand_rhae))
    losses = int(np.sum(spike_rhae < rand_rhae))
    ties   = int(np.sum(spike_rhae == rand_rhae))
    p_val  = binomial_p_one_sided(wins, wins + losses)

    spike_chain = float(np.mean(spike_rhae))
    rand_chain  = float(np.mean(rand_rhae))
    spike_nz    = int(np.sum(spike_rhae > 0))
    rand_nz     = int(np.sum(rand_rhae  > 0))

    if p_val <= 0.10 and spike_chain > rand_chain:
        verdict = 'SIGNAL'
    elif spike_chain > MLP_TP_BASELINE:
        verdict = 'ABOVE_BASELINE'
    elif spike_chain > rand_chain:
        verdict = 'SPIKE_BETTER_NOT_SIG'
    else:
        verdict = 'KILL'

    print(f"\n=== STEP {STEP} RESULTS ===")
    print(f"  SPIKE  chain_mean={spike_chain:.3e}  nz={spike_nz}/{N_DRAWS}")
    print(f"  RANDOM chain_mean={rand_chain:.3e}  nz={rand_nz}/{N_DRAWS}")
    print(f"  Paired: {wins}W-{losses}L-{ties}T  p={p_val:.4f}")
    print(f"  MLP_TP_BASELINE = {MLP_TP_BASELINE:.2e}")
    print(f"  pattern_diff (diagnostic) = {pattern_diff:.4f}")
    print(f"  Verdict: {verdict}")

    summary = {
        'step': STEP,
        'n_draws': N_DRAWS,
        'draw_seeds': DRAW_SEEDS,
        'diag_pattern_diff': round(pattern_diff, 4),
        'conditions': {
            'SPIKE':  {'chain_mean': round(spike_chain, 8), 'nz': spike_nz},
            'RANDOM': {'chain_mean': round(rand_chain,  8), 'nz': rand_nz},
        },
        'paired_wins': wins, 'paired_losses': losses, 'paired_ties': ties,
        'p_value': round(p_val, 6),
        'verdict': verdict,
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/summary.json")


if __name__ == '__main__':
    main()
