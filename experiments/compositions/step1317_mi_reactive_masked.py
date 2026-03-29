"""
Step 1317 — Catalog item #17: MI-detected reactive (replicate Step 1161, ARC=0.200)
Leo mail 3725, 2026-03-29.

Replication of sub1161_defense_v67.py under current masked PRISM protocol.
Original Step 1161: ARC=0.200 — highest score in debate. ONE experiment, never replicated.

Architecture:
- Phase 1 (0-300 steps): cycle first min(n_actions, 20) actions (hold each 15 steps).
  Build per-action per-dim EMA stats of observation deltas.
- Step 300: compute MI_d = 0.5 * log(var_total[d] / mean_within_var[d]).
  Rank actions by MI-weighted expected effect. Select top-K.
- Phase 2 (300+): reactive cycling over top-K MI-ranked actions.
  Switch when dist-to-initial doesn't improve. Recompute MI every 200 steps.

Constitutional status: R2 borderline (MI is fixed formula, not learned). R3 weak (zero learned
params). This is a REPLICATION TEST — does the original signal hold under current protocol?
If yes: combine MI detection with multi-layer LPL for R2/R3 compliance in Step 1318.

Protocol: MBPP + 2 masked ARC, seed-free, 1 run per game.
  6 runs: 3 MI + 3 RAND (2K steps each). No T_chain (deferred to 1318 if replicated).

Kill criteria:
  MI RHAE ≤ RAND RHAE → catalog item killed
  MI action_KL < 0.01 → collapsed → KILL

Predictions (Leo):
  1. MI action_KL ≠ RAND (MI selects differently)
  2. MI RHAE > 0 on at least one game (original signal)
"""
import sys, os, time, json, logging

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
from substrates.step0674 import _enc_frame
from prism_masked import select_games, seal_mapping, label_filename, masked_game_list, masked_run_log

GAMES, GAME_LABELS = select_games(seed=1317)

MAX_STEPS      = 2_000
MAX_SECONDS    = 300

CONDITIONS = ['mi', 'rand']
LABELS     = {'mi': 'MI', 'rand': 'RAND'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1317')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'

SOLVER_PRESCRIPTIONS = {
    'ls20':  ('ls20_fullchain.json',  'all_actions'),
    'ft09':  ('ft09_fullchain.json',  'all_actions'),
    'vc33':  ('vc33_fullchain.json',  'all_actions_encoded'),
    'tr87':  ('tr87_fullchain.json',  'all_actions'),
    'sp80':  ('sp80_fullchain.json',  'all_actions'),
    'sb26':  ('sb26_fullchain.json',  'all_actions'),
    'tu93':  ('tu93_fullchain.json',  'all_actions'),
    'cn04':  ('cn04_fullchain.json',  'sequence'),
    'cd82':  ('cd82_fullchain.json',  'all_actions'),
    'lp85':  ('lp85_fullchain.json',  'all_actions'),
}

ACTION_OFFSET = {'ls20': -1, 'vc33': 7}

ENC_DIM        = 256
N_WARMUP_ACT   = 20      # explore first min(n_actions, 20) actions in phase 1
SUSTAIN_STEPS  = 15      # hold each action N steps during estimation
MI_WARMUP      = 300     # steps for MI estimation phase
MI_RECOMPUTE   = 200     # recompute MI every N steps in phase 2
MI_EPSILON     = 1e-8
MI_EMA         = 0.95
TOP_K          = 5

SEED_A = 0


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


def load_prescription(game_name):
    if game_name.lower() not in SOLVER_PRESCRIPTIONS:
        return None
    fname, field = SOLVER_PRESCRIPTIONS[game_name.lower()]
    path = os.path.join(PDIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(field)


def compute_solver_level_steps(game_name):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        return mbpp_game.compute_solver_steps(0)
    prescription = load_prescription(game_name)
    if prescription is None:
        return {}
    env = make_game(game_name)
    env.reset(seed=1)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103
    offset = ACTION_OFFSET.get(game_name.lower(), 0)
    level, level_first_step, step, fresh_episode = 0, {}, 0, True
    for action in prescription:
        action_int = (int(action) + offset) % n_actions
        obs_next, reward, done, info = env.step(action_int)
        step += 1
        if fresh_episode:
            fresh_episode = False
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level_first_step[cl] = step
            level = cl
        if done:
            env.reset(seed=1)
            fresh_episode = True
    return level_first_step


# ---------------------------------------------------------------------------
# Substrates
# ---------------------------------------------------------------------------

class MISubstrate:
    """MI-detected reactive. Fixed formula, zero learned params.

    Phase 1: cycle first min(n_actions, 20) actions, hold each 15 steps.
    Phase 2: reactive cycling over top-K MI-ranked actions.
    Deterministic — MI stats built from observations, no random init.
    """

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.step = 0
        self._n_warmup = min(n_actions, N_WARMUP_ACT)
        # MI statistics (per-action, per-dim EMA)
        self._mi_mu = np.zeros((self._n_warmup, ENC_DIM), np.float32)
        self._mi_var = np.full((self._n_warmup, ENC_DIM), 1e-4, np.float32)
        self._mi_var_total = np.zeros(ENC_DIM, np.float32)
        self._mi_count = np.zeros(self._n_warmup, np.float32)
        self._mi_values = np.zeros(ENC_DIM, np.float32)
        # Reactive state
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._prev_action_idx = 0
        self._best_actions = list(range(self._n_warmup))
        self._mi_computed = False
        self._current_action_pos = 0
        self._patience = 0
        # For I3 tracking
        self._action_counts = np.zeros(n_actions, np.float32)

    def _update_mi_stats(self, action_idx, delta):
        if action_idx >= self._n_warmup:
            return
        alpha = 1.0 - MI_EMA
        self._mi_count[action_idx] += 1
        self._mi_mu[action_idx] = MI_EMA * self._mi_mu[action_idx] + alpha * delta
        residual = delta - self._mi_mu[action_idx]
        self._mi_var[action_idx] = MI_EMA * self._mi_var[action_idx] + alpha * (residual ** 2)
        self._mi_var_total = MI_EMA * self._mi_var_total + alpha * (delta ** 2)

    def _compute_mi(self):
        active = self._mi_count > 5
        if active.sum() < 2:
            return
        mean_within_var = self._mi_var[active].mean(axis=0)
        ratio = self._mi_var_total / np.maximum(mean_within_var, MI_EPSILON)
        self._mi_values = np.maximum(0.5 * np.log(np.maximum(ratio, 1.0)), 0.0)

    def _rank_actions_by_mi(self):
        scores = []
        for a in range(self._n_warmup):
            score = float(np.sum(self._mi_values * np.abs(self._mi_mu[a])))
            scores.append((score, a))
        scores.sort(reverse=True)
        self._best_actions = [a for s, a in scores[:TOP_K] if s > 0.001]
        if not self._best_actions:
            self._best_actions = list(range(min(self._n_warmup, TOP_K)))
        self._mi_computed = True
        self._current_action_pos = 0

    def process(self, obs_raw):
        self.step += 1
        enc = _enc_frame(np.asarray(obs_raw, np.float32))

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._action_counts[0] += 1
            return 0

        delta = enc - self._prev_enc
        self._update_mi_stats(self._prev_action_idx, delta)
        dist = float(np.sum(np.abs(enc - self._enc_0)))

        # Phase 1: sustained-hold estimation
        if self.step <= MI_WARMUP:
            action_idx = ((self.step - 1) // SUSTAIN_STEPS) % self._n_warmup
            self._prev_dist = dist
            self._prev_enc = enc.copy()
            self._prev_action_idx = action_idx
            self._action_counts[action_idx % self.n_actions] += 1
            return action_idx

        # Compute/recompute MI
        if not self._mi_computed or (self.step - MI_WARMUP) % MI_RECOMPUTE == 0:
            self._compute_mi()
            self._rank_actions_by_mi()

        # Phase 2: reactive cycling
        if dist >= self._prev_dist:
            self._current_action_pos = (self._current_action_pos + 1) % len(self._best_actions)
            self._patience = 0
        else:
            self._patience += 1
            if self._patience > 10:
                self._patience = 0
                self._current_action_pos = (self._current_action_pos + 1) % len(self._best_actions)

        action = self._best_actions[self._current_action_pos]
        self._prev_dist = dist
        self._prev_enc = enc.copy()
        self._prev_action_idx = action
        self._action_counts[action % self.n_actions] += 1
        return action

    def update_after_step(self, obs_next, action, reward):
        pass

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._current_action_pos = 0
        self._patience = 0
        # Reset MI stats for fresh estimation on new level
        self._mi_mu[:] = 0
        self._mi_var[:] = 1e-4
        self._mi_count[:] = 0
        self._mi_var_total[:] = 0
        self._mi_values[:] = 0
        self._mi_computed = False

    def compute_weight_drift(self):
        return 0.0  # no learned weights


class RandSubstrate:
    """Pure random action selection baseline."""

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.step = 0
        self._action_counts = np.zeros(n_actions, np.float32)

    def process(self, obs_raw):
        self.step += 1
        action = int(np.random.choice(self.n_actions))
        self._action_counts[action] += 1
        return action

    def update_after_step(self, obs_next, action, reward):
        pass

    def on_level_transition(self):
        pass

    def compute_weight_drift(self):
        return 0.0


def make_substrate(condition, n_actions):
    if condition == 'mi':
        return MISubstrate(n_actions=n_actions)
    return RandSubstrate(n_actions=n_actions)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_arc_score(level_first_step, solver_level_steps):
    if not level_first_step or not solver_level_steps:
        return 0.0
    scores = []
    for lvl, s_step in solver_level_steps.items():
        a_step = level_first_step.get(lvl)
        if a_step is not None and s_step > 0:
            scores.append((s_step / a_step) ** 2)
    if not scores:
        return 0.0
    return round(float(np.mean(scores)), 6)


def compute_action_kl(action_log, n_actions):
    if len(action_log) < 400:
        return None
    early_c = np.zeros(n_actions, np.float32)
    late_c = np.zeros(n_actions, np.float32)
    for a in action_log[:200]:
        if 0 <= a < n_actions:
            early_c[a] += 1
    for a in action_log[-200:]:
        if 0 <= a < n_actions:
            late_c[a] += 1
    early_p = (early_c + 1e-8) / (early_c.sum() + 1e-8 * n_actions)
    late_p = (late_c + 1e-8) / (late_c.sum() + 1e-8 * n_actions)
    return round(float(np.sum(early_p * np.log(early_p / late_p + 1e-12))), 4)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, solver_level_steps, seed):
    obs = env.reset(seed=seed)
    action_log = []
    i3_counts_at_200 = None
    action_counts = np.zeros(n_actions, np.float32)

    steps = 0
    episode_step = 0
    level = 0
    max_level = 0
    level_first_step = {}
    t_start = time.time()
    fresh_episode = True

    while steps < MAX_STEPS:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            continue

        if steps == 200:
            i3_counts_at_200 = action_counts.copy()

        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr) % n_actions
        action_counts[action] += 1
        action_log.append(action)
        episode_step += 1

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
            if cl not in level_first_step:
                level_first_step[cl] = steps
            if cl > max_level:
                max_level = cl
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

    i3_cv = None
    if i3_counts_at_200 is not None:
        counts = i3_counts_at_200[:n_actions].astype(float)
        mean_c = counts.mean()
        if mean_c > 1e-8:
            i3_cv = round(float(counts.std() / mean_c), 4)

    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    action_kl = compute_action_kl(action_log, n_actions)
    wdrift = substrate.compute_weight_drift()

    return {
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(arc_score, 6),
        'I3_cv': i3_cv,
        'action_kl': action_kl,
        'wdrift': round(wdrift, 4),
    }


# ---------------------------------------------------------------------------
# Kill assessment
# ---------------------------------------------------------------------------

def kill_assessment(chain_summary):
    s_mi = chain_summary['mi']
    s_rand = chain_summary['rand']
    rhae_mi = s_mi.get('mean_RHAE', 0.0)
    rhae_rand = s_rand.get('mean_RHAE', 0.0)
    kl_mi = s_mi.get('mean_action_KL')

    lines = []
    kill = False

    # Primary: RHAE comparison
    if rhae_mi > rhae_rand:
        lines.append(
            f"  >>> SIGNAL: MI RHAE={rhae_mi:.2e} > RAND RHAE={rhae_rand:.2e} "
            f"→ catalog item signal replicated"
        )
        lines.append("  >>> PREDICTION 2 (MI RHAE > 0): CONFIRMED" if rhae_mi > 0 else
                     "  >>> PREDICTION 2 (MI RHAE > 0): WRONG (tied at 0)")
    else:
        lines.append(
            f"  >>> KILL: MI RHAE={rhae_mi:.2e} ≤ RAND RHAE={rhae_rand:.2e} "
            f"→ MI detection doesn't help task performance"
        )
        lines.append("  >>> PREDICTION 2 (MI RHAE > 0): WRONG")
        kill = True

    # action_KL (Leo prediction 1)
    kl_rand = s_rand.get('mean_action_KL', 0.0)
    if kl_mi is not None:
        kl_diff = abs(kl_mi - (kl_rand or 0.0))
        pred1 = "CONFIRMED" if kl_diff > 0.1 else "WRONG"
        lines.append(
            f"  >>> PREDICTION 1 (MI action_KL ≠ RAND): {pred1} "
            f"(MI={kl_mi:.4f}, RAND={kl_rand:.4f})"
        )
        if kl_mi < 0.01:
            lines.append(f"  >>> KILL: MI action_KL={kl_mi:.4f} < 0.01 → collapsed")
            kill = True

    if kill:
        lines.append("  >>> KILL TRIGGERED — catalog item #17 killed under current protocol")
    else:
        lines.append("  >>> NO KILL — signal replicated. → 1318: combine with multi-layer LPL")

    return lines, kill


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step 1317 — Catalog item #17: MI-detected reactive (replicate Step 1161)")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Conditions: {CONDITIONS}")
    print(f"MI: phase1 cycle first {N_WARMUP_ACT} actions (hold {SUSTAIN_STEPS} steps each)")
    print(f"    phase2 reactive cycling over top-{TOP_K} MI-ranked actions")
    print(f"RAND: pure random baseline")
    print(f"6 runs total (3 MI + 3 RAND), 2K steps each. No T_chain (deferred to 1318).")
    print()

    seal_mapping(RESULTS_DIR, GAMES, GAME_LABELS)

    print("Computing solver baselines...")
    solver_steps_cache = {}
    for game in GAMES:
        try:
            solver_steps_cache[game] = compute_solver_level_steps(game)
        except Exception:
            solver_steps_cache[game] = {}
    print()

    all_results = []

    for game in GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")
        solver_steps = solver_steps_cache[game]
        env = make_game(game)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = 4103

        t0 = time.time()
        for condition in CONDITIONS:
            substrate = make_substrate(condition, n_actions)
            result = run_episode(env, substrate, n_actions, solver_steps, seed=SEED_A)
            result['game'] = game
            result['condition'] = condition
            result['n_actions'] = n_actions
            all_results.append(result)

        elapsed = time.time() - t0
        print(masked_run_log(label, elapsed))

        game_results = [r for r in all_results if r['game'] == game]
        out_path = os.path.join(RESULTS_DIR, label_filename(label, 1317))
        with open(out_path, 'w') as f:
            for r in game_results:
                masked_r = {k: v for k, v in r.items() if k != 'game'}
                masked_r['label'] = label
                f.write(json.dumps(masked_r) + '\n')

    # Chain-level aggregates
    chain_data = {c: {'RHAE': [], 'action_KL': [], 'I3cv': []} for c in CONDITIONS}
    for r in all_results:
        cond = r['condition']
        chain_data[cond]['RHAE'].append(r.get('RHAE', 0.0) or 0.0)
        chain_data[cond]['action_KL'].append(r.get('action_kl', 0.0) or 0.0)
        chain_data[cond]['I3cv'].append(r.get('I3_cv', 0.0) or 0.0)

    chain_summary = {}
    for cond in CONDITIONS:
        d = chain_data[cond]
        chain_summary[cond] = {
            'mean_RHAE': round(float(np.mean(d['RHAE'])) if d['RHAE'] else 0.0, 8),
            'mean_action_KL': round(float(np.mean(d['action_KL'])) if d['action_KL'] else 0.0, 4),
            'mean_I3cv': round(float(np.mean(d['I3cv'])) if d['I3cv'] else 0.0, 4),
        }

    summary = {
        'chain_summary': chain_summary,
        'games_label_only': masked_game_list(GAME_LABELS),
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP 1317 — CHAIN-LEVEL SUMMARY\n")
    for cond in CONDITIONS:
        lbl = LABELS[cond]
        s = chain_summary[cond]
        print(f"  {lbl}:")
        print(f"    mean_RHAE={s['mean_RHAE']:.2e}  mean_action_KL={s['mean_action_KL']}  "
              f"mean_I3cv={s['mean_I3cv']}")
    print()
    print("KILL ASSESSMENT (chain aggregates only):")
    kill_lines, kill = kill_assessment(chain_summary)
    for line in kill_lines:
        print(line)
    print(sep)
    print(f"\nResults saved to {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()
