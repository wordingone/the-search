"""
Step 1390 — Slot-based object discovery.
Leo mail 3979, 2026-03-30. Gates confirmed before build.

Root cause from 1387-1389: pixel magnitude (total or localized) can't distinguish
animations from functional interactions — animations change globally; interactive
objects change LOCALLY when clicked.

New approach: measure TARGET slot delta vs OTHER slot delta per click.
- 8×8 patch grid on 64×64 canvas = 64 proto-object slots.
- Each slot: mean color (per patch).
- For click at (x,y): target_slot = (x//8) + (y//8)*8.
  effect_on_target[slot] += delta_at_target_slot
  effect_on_other[slot]  += mean_delta_at_non_target_slots
- Interactive slot: target_delta >> other_delta (ratio > RATIO_THRESH=3.0).
- Try2 SLOT:   score[action] = ratio[target_slot_of_action]. Softmax.
- Try2 RANDOM: uniform random.

Mandatory diagnostic after DIAG_STEPS=500 try1 steps on first 3 draws:
  If 0 responsive slots (ratio>3) on ALL 3 → DIAGNOSTIC_FAIL. Abort.

Conditions (50 draws, paired):
  SLOT:   slot-based try2.
  RANDOM: uniform random try2 (control).

Seeds: 14330-14379. TRY2_SEED=4.
"""

import os
import sys
import json
import time
import math
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

STEP        = 1390
N_DRAWS     = 50
DRAW_SEEDS  = [14330 + i for i in range(N_DRAWS)]
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

RESULTS_DIR     = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')
MLP_TP_BASELINE = 4.59e-5

N_SLOTS       = 64    # 8×8 patch grid on 64×64 canvas
SLOT_SIZE     = 8     # pixels per slot side
N_KEYBOARD    = 7     # action IDs 0..6 are keyboard/special
TEMPERATURE   = 3.0
RATIO_THRESH  = 3.0   # target_delta / other_delta threshold for interactive
EPS           = 1e-9

CONDITIONS    = ['SLOT', 'RANDOM']

DIAG_STEPS      = 500   # check diagnostic after this many try1 steps
N_DIAG_DRAWS    = 3     # first N draws used for mandatory diagnostic gate

MAX_N_ACTIONS = 4103
TIER1_STEPS   = 200


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
# Slot helpers
# ---------------------------------------------------------------------------

def _is_arc_obs(arr):
    return arr.ndim == 3 and arr.shape[-2] == 64 and arr.shape[-1] == 64


def _click_to_slot(action_id):
    """Map ARC click action → slot index (0..63). None for non-click."""
    if action_id < N_KEYBOARD:
        return None
    click_idx = action_id - N_KEYBOARD
    if click_idx >= 64 * 64:
        return None
    x = click_idx % 64
    y = click_idx // 64
    return (x // SLOT_SIZE) + (y // SLOT_SIZE) * 8


def _compute_slot_delta(obs_before, obs_after):
    """
    Compute per-slot mean absolute delta for ARC observations.
    obs shape: (C, 64, 64). Returns float64 array of shape (64,).

    Uses reshape trick: (C, 64, 64) → (C, 8, 8, 8, 8) where
    axes are (C, row_patch, row_pixel, col_patch, col_pixel).
    Mean over (C, row_pixel, col_pixel) → (row_patch, col_patch) = (8,8).
    Flattened index: row_patch * 8 + col_patch = (y//8)*8 + (x//8).
    Same ordering as _click_to_slot: (x//8) + (y//8)*8 ← same value.
    """
    C = obs_before.shape[0]
    b_r = obs_before.reshape(C, 8, SLOT_SIZE, 8, SLOT_SIZE)
    a_r = obs_after.reshape(C, 8, SLOT_SIZE, 8, SLOT_SIZE)
    delta = np.abs(a_r - b_r).mean(axis=(0, 2, 4))  # (8, 8)
    return delta.flatten().astype(np.float64)        # (64,)


def _build_slot_scores(n_actions, ratio):
    """
    Vectorized: score[action] = ratio[target_slot] for ARC click actions.
    Returns (n_actions,) float64.
    """
    scores = np.zeros(n_actions, dtype=np.float64)
    n_clicks = min(n_actions - N_KEYBOARD, 64 * 64)
    if n_clicks > 0:
        ci   = np.arange(n_clicks, dtype=np.int64)
        xs   = ci % 64
        ys   = ci // 64
        slots = (xs // SLOT_SIZE) + (ys // SLOT_SIZE) * 8
        scores[N_KEYBOARD:N_KEYBOARD + n_clicks] = ratio[slots]
    return scores


# ---------------------------------------------------------------------------
# Slot substrate
# ---------------------------------------------------------------------------

class SlotSubstrate:
    """
    Tabular slot-effect tracker. No neural network.

    Try1: argmin coverage (both conditions). Accumulates effect tables.
    Try2 SLOT:   softmax(ratio[target_slot] / T) per action.
    Try2 RANDOM: uniform random.
    """

    def __init__(self, n_actions, mode):
        self.n_actions = n_actions
        self.mode      = mode       # 'slot' or 'random'

        # Slot effect tables — carry over try1 → try2
        self._effect_on_target  = np.zeros(N_SLOTS, dtype=np.float64)
        self._effect_on_other   = np.zeros(N_SLOTS, dtype=np.float64)
        self._keyboard_effect   = np.zeros(max(N_KEYBOARD, n_actions), dtype=np.float64)

        # Episode state
        self._obs_prev       = None
        self._in_try2        = False
        self._step           = 0
        self._visit_count    = np.zeros(n_actions, dtype=np.int64)
        self._current_level  = 0
        self._try2_max_level = 0
        self._is_arc         = None

        # Diagnostics
        self._effect_events = 0

    def _detect_arc(self, obs_arr):
        if self._is_arc is None:
            self._is_arc = _is_arc_obs(np.asarray(obs_arr, dtype=np.float32))
        return self._is_arc

    def process(self, obs_arr):
        is_arc = self._detect_arc(obs_arr)

        if self._in_try2 and self.mode == 'slot':
            if is_arc:
                ratio  = (self._effect_on_target + EPS) / (self._effect_on_other + EPS)
                scores = _build_slot_scores(self.n_actions, ratio)
                # Keyboard: use keyboard_effect as fallback
                kb = min(N_KEYBOARD, self.n_actions, len(self._keyboard_effect))
                scores[:kb] = self._keyboard_effect[:kb]
            else:
                scores = self._keyboard_effect[:self.n_actions].copy()
            scores /= TEMPERATURE
            scores -= scores.max()
            probs   = np.exp(scores)
            probs  /= probs.sum()
            action  = int(np.random.choice(self.n_actions, p=probs))

        elif self._in_try2 and self.mode == 'random':
            action = int(np.random.randint(self.n_actions))

        else:
            # Try1: argmin coverage
            min_c      = self._visit_count.min()
            candidates = np.where(self._visit_count == min_c)[0]
            action     = int(np.random.choice(candidates))

        self._visit_count[action] += 1
        self._obs_prev = obs_arr
        self._step    += 1
        return action

    def update_after_step(self, obs_next, action, reward):
        if self._obs_prev is None or obs_next is None:
            return
        b = np.asarray(self._obs_prev, dtype=np.float32)
        a = np.asarray(obs_next,       dtype=np.float32)

        if not _is_arc_obs(b) or action < N_KEYBOARD:
            # Non-ARC or keyboard: accumulate global delta to keyboard table
            global_delta = float(np.mean(np.abs(a.flatten() - b.flatten())))
            if action < len(self._keyboard_effect):
                self._keyboard_effect[action] += global_delta
            return

        click_idx = action - N_KEYBOARD
        if click_idx >= 64 * 64:
            return

        x = click_idx % 64
        y = click_idx // 64
        target_slot = (x // SLOT_SIZE) + (y // SLOT_SIZE) * 8

        slot_delta = _compute_slot_delta(b, a)
        target_d   = float(slot_delta[target_slot])
        other_d    = float((slot_delta.sum() - target_d) / (N_SLOTS - 1))

        self._effect_on_target[target_slot] += target_d
        self._effect_on_other[target_slot]  += other_d

        if target_d > 0 or other_d > 0:
            self._effect_events += 1

    def on_level_transition(self, new_level=None):
        self._obs_prev = None
        if new_level is not None:
            self._current_level = new_level
            if self._in_try2:
                self._try2_max_level = max(self._try2_max_level, new_level)

    def prepare_for_try2(self):
        self._in_try2        = True
        self._step           = 0
        self._obs_prev       = None
        self._visit_count[:] = 0
        self._current_level  = 0
        # Effect tables carry over try1 → try2

    def compute_stage_metrics(self):
        ratio        = (self._effect_on_target + EPS) / (self._effect_on_other + EPS)
        n_responsive = int(np.sum(ratio > RATIO_THRESH))
        return {
            'i3_cv':              None,
            'i4_h_early':         None,
            'i4_h_late':          None,
            'i4_reduction':       None,
            'i1_within':          None,
            'i1_between':         None,
            'i1_pass':            None,
            'i5_max_level':       self._try2_max_level,
            'r3_weight_diff':     None,
            'n_responsive_slots': n_responsive,
            'target_effect_norm': round(float(np.linalg.norm(self._effect_on_target)), 4),
            'other_effect_norm':  round(float(np.linalg.norm(self._effect_on_other)),  4),
            'effect_events':      self._effect_events,
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, seed, max_steps):
    obs          = env.reset(seed=seed)
    steps        = 0
    level        = 0
    steps_to_first_progress = None
    t_start      = time.time()
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
            if steps_to_first_progress is None:
                steps_to_first_progress = steps
            level = cl
            substrate.on_level_transition(new_level=cl)
        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition(new_level=0)
            level = 0
            fresh_episode = True
        else:
            obs = obs_next

    return steps_to_first_progress, round(time.time() - t_start, 2)


# ---------------------------------------------------------------------------
# Binomial p-value (one-sided)
# ---------------------------------------------------------------------------

def binomial_p_one_sided(wins, n):
    if n == 0:
        return 1.0
    from math import comb
    p = sum(comb(n, k) * (0.5 ** n) for k in range(wins, n + 1))
    return round(p, 6)


# ---------------------------------------------------------------------------
# Draw runner
# ---------------------------------------------------------------------------

def run_draw(draw_idx, draw_seed, cond_name, max_steps, do_diag=False):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, cond_name, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    draw_results    = []
    try2_progress   = {}
    optimal_steps_d = {}
    mode = 'slot' if 'slot' in cond_name.lower() else 'random'

    diag_n_responsive_list = []

    for game_idx, (game_name, label) in enumerate(zip(games, game_labels.values())):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = MAX_N_ACTIONS

        substrate = SlotSubstrate(n_actions=n_actions, mode=mode)

        p1, t1 = run_episode(env, substrate, n_actions, seed=0,         max_steps=max_steps)

        # Diagnostic: read try1 effect tables BEFORE prepare_for_try2 clears state.
        # Ratio = target_delta / other_delta per slot. Count slots with ratio > RATIO_THRESH.
        if do_diag:
            ratio_try1 = (substrate._effect_on_target + EPS) / (substrate._effect_on_other + EPS)
            diag_n_responsive_list.append(int(np.sum(ratio_try1 > RATIO_THRESH)))

        substrate.prepare_for_try2()
        np.random.seed(draw_seed * 1000 + 1)   # PRNG fix
        p2, t2 = run_episode(env, substrate, n_actions, seed=TRY2_SEED, max_steps=max_steps)

        stage = substrate.compute_stage_metrics()

        speedup = compute_progress_speedup(p1, p2)
        opt     = get_optimal_steps(game_name, TRY2_SEED)
        eff_sq  = 0.0
        if p2 is not None and opt is not None and opt > 0:
            eff_sq = round(min(1.0, opt / p2) ** 2, 6)

        try2_progress[label]   = p2
        optimal_steps_d[label] = opt

        row = {
            'draw': draw_idx, 'label': label, 'game': game_name,
            'condition': cond_name,
            'p1': p1, 'p2': p2, 'speedup': speedup,
            'eff_sq': eff_sq, 'optimal_steps': opt,
            't1': t1, 't2': t2,
            'n_responsive_slots': stage['n_responsive_slots'],
            'target_effect_norm': stage['target_effect_norm'],
            'other_effect_norm':  stage['other_effect_norm'],
            'effect_events':      stage['effect_events'],
            'i5_max_level':       stage['i5_max_level'],
        }
        masked_row = mask_result_row(row, game_labels)

        fn = os.path.join(draw_dir, label_filename(label, STEP))
        with open(fn, 'w') as f:
            f.write(json.dumps(masked_row) + '\n')

        draw_results.append(masked_row)

    rhae    = compute_rhae_try2(try2_progress, optimal_steps_d)
    nz_resp = round(float(np.mean([r.get('n_responsive_slots', 0) for r in draw_results])), 1)

    print(f"  [{cond_name}] Draw {draw_idx} RHAE={rhae:.6e}  responsive_slots={nz_resp:.0f}")
    return round(rhae, 7), draw_results, diag_n_responsive_list


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

    sub_t1 = SlotSubstrate(n_actions=na_t1, mode='slot')
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)   # warmup
    t0 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t0

    ms_per_step  = tier1_elapsed / TIER1_STEPS * 1000
    n_eps_total  = N_DRAWS * 3 * 2 * len(CONDITIONS)
    est_total_s  = ms_per_step / 1000 * TRY1_STEPS * n_eps_total
    print(f"  {TIER1_STEPS} steps: {tier1_elapsed:.2f}s  ({ms_per_step:.2f}ms/step)")
    print(f"  Estimated total: {est_total_s:.0f}s ({est_total_s/60:.1f} min)")

    if est_total_s > MAX_SECONDS:
        max_steps = max(200, int(
            (MAX_SECONDS * 0.85) / (ms_per_step / 1000 * n_eps_total)
        ))
        print(f"  Budget exceeded — capping at {max_steps} steps")
    else:
        max_steps = TRY1_STEPS
        print(f"  Under budget — full {max_steps} steps")

    # ── Mandatory diagnostic: first N_DIAG_DRAWS draws (SLOT condition) ──
    print(f"\n=== MANDATORY DIAGNOSTIC: {N_DIAG_DRAWS} draws, {DIAG_STEPS} try1 steps ===")
    all_diag_responsive = []
    for di in range(N_DIAG_DRAWS):
        _, _, diag_list = run_draw(di, DRAW_SEEDS[di], 'SLOT', max_steps=DIAG_STEPS, do_diag=True)
        total_responsive = sum(diag_list) if diag_list else 0
        all_diag_responsive.append(total_responsive)
        print(f"  Diagnostic draw {di}: responsive_slots_sum={total_responsive}  (per-game: {diag_list})")

    if max(all_diag_responsive) == 0:
        print("\nDIAGNOSTIC_FAIL: 0 responsive slots (ratio>3) on all 3 diagnostic draws.")
        print("No interactive objects detected. Aborting — mechanism can't detect functional interactions.")
        summary = {
            'step': STEP, 'verdict': 'DIAGNOSTIC_FAIL',
            'diag_draws': all_diag_responsive,
        }
        fn = os.path.join(RESULTS_DIR, 'summary.json')
        with open(fn, 'w') as f:
            json.dump(summary, f, indent=2)
        return

    print(f"  Diagnostic PASS: max responsive slots = {max(all_diag_responsive)}")

    # ── Full experiment: SLOT vs RANDOM, 50 draws ─────────────────────────
    print(f"\n=== STEP {STEP}: SLOT vs RANDOM, {N_DRAWS} draws, {max_steps} steps ===")
    print(f"Seeds: {DRAW_SEEDS[0]}-{DRAW_SEEDS[-1]}")

    rhae_by_cond = {c: [] for c in CONDITIONS}

    for cond_name in CONDITIONS:
        print(f"\n--- Condition: {cond_name} ---")
        t_cond = time.time()
        for di, ds in enumerate(DRAW_SEEDS):
            rhae, _, _ = run_draw(di, ds, cond_name, max_steps)
            rhae_by_cond[cond_name].append(round(rhae, 7))
        print(f"  {cond_name} done in {time.time()-t_cond:.0f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        'step':            STEP,
        'n_draws':         N_DRAWS,
        'draw_seeds':      DRAW_SEEDS,
        'max_steps_used':  max_steps,
        'ms_per_step':     round(ms_per_step, 2),
        'mlp_tp_baseline': MLP_TP_BASELINE,
        'diag_draws':      all_diag_responsive,
        'conditions':      {},
    }

    print(f"\n=== RESULTS ===")
    for cond_name in CONDITIONS:
        rhae_list  = rhae_by_cond[cond_name]
        chain_mean = round(sum(rhae_list) / len(rhae_list), 7)
        nz         = sum(1 for r in rhae_list if r > 0)
        print(f"\n{cond_name}:  chain_mean={chain_mean:.6e}  nz={nz}/{N_DRAWS}")
        summary['conditions'][cond_name] = {
            'chain_mean': chain_mean,
            'nz':         nz,
            'rhae_list':  rhae_list,
        }

    # Paired sign test: SLOT vs RANDOM
    slot_rhae = rhae_by_cond['SLOT']
    rnd_rhae  = rhae_by_cond['RANDOM']
    wins   = sum(1 for a, b in zip(slot_rhae, rnd_rhae) if a > b)
    losses = sum(1 for a, b in zip(slot_rhae, rnd_rhae) if a < b)
    ties   = N_DRAWS - wins - losses
    p_val  = binomial_p_one_sided(wins, wins + losses)

    print(f"\nPaired sign test (SLOT vs RANDOM):")
    print(f"  Wins={wins}  Losses={losses}  Ties={ties}  p={p_val:.6f}")

    if wins > losses and p_val <= 0.10:
        verdict = 'SIGNAL'
    elif wins < losses:
        verdict = 'SLOT_WORSE'
    else:
        verdict = 'KILL'

    summary['paired_wins']   = wins
    summary['paired_losses'] = losses
    summary['paired_ties']   = ties
    summary['p_value']       = p_val
    summary['verdict']       = verdict

    fn = os.path.join(RESULTS_DIR, 'summary.json')
    with open(fn, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {fn}")

    print(f"\nVERDICT: {verdict}")
    for cond_name in CONDITIONS:
        c = summary['conditions'][cond_name]
        print(f"  {cond_name}  chain_mean: {c['chain_mean']:.6e}")


if __name__ == '__main__':
    main()
