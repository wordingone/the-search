"""
Step 1318 — Catalog item #21: Eigenoptions on multi-layer LPL W3
Leo mail 3728, 2026-03-29.

Catalog item #21: eigenoptions / successor representation on W3.
Architecture: multi-layer LPL (same W1/W2 as 1313). One change — action selection:
  EIGEN: action = argmax(W3 @ h2)  — W3's learned structure read directly
  BASE (1313): softmax(W3 @ h2) with temperature

W3 updated by Hebbian: W3 += ETA * outer(action_onehot, h2)  (same as 1313)
W3 starts at zeros. argmax on all-zeros → action 0 initially, then W3 builds structure.

Constitutional: R2 PASS — action selection reads from W3, which is updated by same
prediction-error mechanism (LPL). W3 is both prediction memory and action selector.

Known risk: Hebbian W3 + argmax + zero init → action collapse (same as Step 1264).
Kill criterion I3cv > 20 designed to catch this.

Protocol: MBPP + 2 masked ARC, seed-free, 1 run per game.
  6 runs: 3 EIGEN + 3 RAND (2K steps each).

Kill criteria:
  EIGEN RHAE ≤ RAND RHAE → W3 structure doesn't help → KILL
  EIGEN action_KL < 0.01 → collapsed → KILL
  EIGEN I3cv > 20 → over-concentrated → KILL

Predictions (Leo):
  1. EIGEN action_KL ≠ RAND (W3 produces non-random selection)
  2. EIGEN I3cv > RAND I3cv (W3 concentrates on structured actions)
  3. RHAE: uncertain
"""
import sys, os, time, json, logging

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame
from prism_masked import select_games, seal_mapping, label_filename, masked_game_list, masked_run_log

GAMES, GAME_LABELS = select_games(seed=1318)

MAX_STEPS      = 2_000
MAX_SECONDS    = 300

CONDITIONS = ['eigen', 'rand']
LABELS     = {'eigen': 'EIGEN', 'rand': 'RAND'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1318')
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

ENC_DIM = 256
H1_DIM  = 128
H2_DIM  = 64
ETA     = 0.001
INF_LR  = 0.05
K_INF   = 5
W_CLIP  = 100.0

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

class EigenSubstrate:
    """Multi-layer LPL + argmax(W3@h2) action selection.

    W3 starts at zeros. Hebbian update: W3 += ETA * outer(action_onehot, h2).
    Action selection: argmax(W3 @ h2) — uses W3's learned h2→action mapping directly.
    W1/W2: same prediction-error Hebbian as 1313.
    """

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.W1 = np.eye(H1_DIM, ENC_DIM, dtype=np.float32) * 0.1
        self.W2 = np.eye(H2_DIM, H1_DIM, dtype=np.float32) * 0.1
        self.W3 = np.zeros((n_actions, H2_DIM), dtype=np.float32)
        self._W1_init = self.W1.copy()
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.step = 0
        self._action_counts = np.zeros(n_actions, np.float32)

    def _centered_encode(self, obs_raw):
        x = _enc_frame(np.asarray(obs_raw, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def _infer(self, enc):
        h1 = self.W1 @ enc
        h2 = self.W2 @ h1
        for _ in range(K_INF):
            enc_pred = self.W1.T @ h1
            e1 = enc - enc_pred
            h1_pred = self.W2.T @ h2
            e2 = h1 - h1_pred
            h1 = h1 + INF_LR * (self.W1 @ e1 - e2)
            h2 = h2 + INF_LR * (self.W2 @ e2)
        enc_pred = self.W1.T @ h1
        e1 = enc - enc_pred
        h1_pred = self.W2.T @ h2
        e2 = h1 - h1_pred
        return h1, h2, e1, e2

    def process(self, obs_raw):
        self.step += 1
        enc = self._centered_encode(obs_raw)
        h1, h2, e1, e2 = self._infer(enc)

        # Eigenoption selection: argmax of W3 @ h2
        scores = self.W3 @ h2
        action = int(np.argmax(scores))
        self._action_counts[action] += 1

        action_onehot = np.zeros(self.n_actions, np.float32)
        action_onehot[action] = 1.0

        # Hebbian updates: W1, W2 (prediction error), W3 (Hebbian)
        self.W1 += ETA * np.outer(h1, e1)
        self.W2 += ETA * np.outer(h2, e2)
        self.W3 += ETA * np.outer(action_onehot, h2)
        np.clip(self.W1, -W_CLIP, W_CLIP, out=self.W1)
        np.clip(self.W2, -W_CLIP, W_CLIP, out=self.W2)
        np.clip(self.W3, -W_CLIP, W_CLIP, out=self.W3)

        return action

    def update_after_step(self, obs_next, action, reward):
        pass

    def on_level_transition(self):
        pass

    def compute_weight_drift(self):
        return float(np.linalg.norm(self.W1 - self._W1_init))


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
    if condition == 'eigen':
        return EigenSubstrate(n_actions=n_actions)
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
    s_eigen = chain_summary['eigen']
    s_rand = chain_summary['rand']
    rhae_eigen = s_eigen.get('mean_RHAE', 0.0)
    rhae_rand = s_rand.get('mean_RHAE', 0.0)
    kl_eigen = s_eigen.get('mean_action_KL', 0.0)
    i3cv_eigen = s_eigen.get('mean_I3cv', 0.0)
    i3cv_rand = s_rand.get('mean_I3cv', 0.0)

    lines = []
    kill = False

    # Primary: RHAE comparison
    if rhae_eigen > rhae_rand:
        lines.append(
            f"  >>> SIGNAL: EIGEN RHAE={rhae_eigen:.2e} > RAND RHAE={rhae_rand:.2e} "
            f"→ W3 structure helps task performance"
        )
    else:
        lines.append(
            f"  >>> KILL: EIGEN RHAE={rhae_eigen:.2e} ≤ RAND RHAE={rhae_rand:.2e} "
            f"→ W3 argmax doesn't help"
        )
        kill = True

    # action_KL (prediction 1)
    kl_rand = s_rand.get('mean_action_KL', 0.0)
    kl_diff = abs(kl_eigen - kl_rand)
    pred1 = "CONFIRMED" if kl_diff > 0.1 else "WRONG"
    lines.append(
        f"  >>> PREDICTION 1 (EIGEN action_KL ≠ RAND): {pred1} "
        f"(EIGEN={kl_eigen:.4f}, RAND={kl_rand:.4f})"
    )
    if kl_eigen < 0.01:
        lines.append(f"  >>> KILL: EIGEN action_KL={kl_eigen:.4f} < 0.01 → collapsed")
        kill = True

    # I3cv (prediction 2)
    pred2 = "CONFIRMED" if i3cv_eigen > i3cv_rand else "WRONG"
    lines.append(
        f"  >>> PREDICTION 2 (EIGEN I3cv > RAND I3cv): {pred2} "
        f"(EIGEN={i3cv_eigen:.4f}, RAND={i3cv_rand:.4f})"
    )
    if i3cv_eigen > 20:
        lines.append(
            f"  >>> KILL: EIGEN I3cv={i3cv_eigen:.4f} > 20 → over-concentrated "
            f"(argmax collapse: W3 Hebbian + argmax locks onto one action)"
        )
        kill = True

    if kill:
        lines.append("  >>> KILL TRIGGERED — eigenoptions on Hebbian W3 killed")
    else:
        lines.append("  >>> NO KILL — W3 structure signal replicated")

    return lines, kill


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step 1318 — Catalog item #21: Eigenoptions on multi-layer LPL W3")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Conditions: {CONDITIONS}")
    print(f"EIGEN: argmax(W3@h2), W3 Hebbian, W1/W2 LPL prediction error (same as 1313)")
    print(f"RAND: pure random baseline")
    print(f"Note: argmax + Hebbian + zero W3 init → expected action collapse (I3cv kill)")
    print(f"6 runs total, 2K steps each.")
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
        out_path = os.path.join(RESULTS_DIR, label_filename(label, 1318))
        with open(out_path, 'w') as f:
            for r in game_results:
                masked_r = {k: v for k, v in r.items() if k != 'game'}
                masked_r['label'] = label
                f.write(json.dumps(masked_r) + '\n')

    # Chain-level aggregates
    chain_data = {c: {'RHAE': [], 'action_KL': [], 'I3cv': [], 'wdrift': []} for c in CONDITIONS}
    for r in all_results:
        cond = r['condition']
        chain_data[cond]['RHAE'].append(r.get('RHAE', 0.0) or 0.0)
        chain_data[cond]['action_KL'].append(r.get('action_kl', 0.0) or 0.0)
        chain_data[cond]['I3cv'].append(r.get('I3_cv', 0.0) or 0.0)
        chain_data[cond]['wdrift'].append(r.get('wdrift', 0.0) or 0.0)

    chain_summary = {}
    for cond in CONDITIONS:
        d = chain_data[cond]
        chain_summary[cond] = {
            'mean_RHAE': round(float(np.mean(d['RHAE'])) if d['RHAE'] else 0.0, 8),
            'mean_action_KL': round(float(np.mean(d['action_KL'])) if d['action_KL'] else 0.0, 4),
            'mean_I3cv': round(float(np.mean(d['I3cv'])) if d['I3cv'] else 0.0, 4),
            'mean_wdrift': round(float(np.mean(d['wdrift'])) if d['wdrift'] else 0.0, 4),
        }

    summary = {
        'chain_summary': chain_summary,
        'games_label_only': masked_game_list(GAME_LABELS),
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP 1318 — CHAIN-LEVEL SUMMARY\n")
    for cond in CONDITIONS:
        lbl = LABELS[cond]
        s = chain_summary[cond]
        print(f"  {lbl}:")
        print(f"    mean_RHAE={s['mean_RHAE']:.2e}  mean_action_KL={s['mean_action_KL']}  "
              f"mean_I3cv={s['mean_I3cv']}  mean_wdrift={s['mean_wdrift']}")
    print()
    print("KILL ASSESSMENT (chain aggregates only):")
    kill_lines, kill = kill_assessment(chain_summary)
    for line in kill_lines:
        print(line)
    print(sep)
    print(f"\nResults saved to {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()
