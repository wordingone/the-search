"""
Step 1322 — Multi-layer LPL K=50 + shared representation
Leo mail 3744, 2026-03-29.

## Question

LPL at K=5 compresses 7% (cr=0.93). CNN compresses 98% with gradient.
Whittington & Bogacz 2017: LPL approximates backprop as K→∞.
Does K=50 close the gap? Does better compression → better h1 structure → task progress?

## Architecture

K50: same multi-layer LPL as 1313 but K=50 inference iterations (was 5).
     Plus shared representation from 1321: action = softmax(h1[:n_actions]).
     No W3.

BASE: standard multi-layer LPL, K=5, W3 Hebbian (1313 style).

## Kill criteria

- K50 speedup ≤ BASE speedup AND K50 cr ≥ 0.93 → K=50 doesn't compress better → KILL
  (LPL is fundamentally weaker than gradient)
- K50 speedup ≤ BASE speedup AND K50 cr < 0.5 → better compression but no task progress → KILL
  (shared rep not enough; compression ≠ task progress)
- K50 speedup > BASE speedup → shared rep + better K works → NO KILL

## Predictions (Leo)
1. cr improves substantially with K=50 (better credit assignment → stronger compression)
2. If cr < 0.5: h1 more structured → possible L1 (shared rep has something to work with)
3. If cr still ~0.9: K=50 insufficient, LPL needs K>>50 → impractical direction
"""
import sys, os, time, json, logging
from collections import deque

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
from substrates.step0674 import _enc_frame
from prism_masked import (select_games, seal_mapping, label_filename,
                           masked_game_list, masked_run_log,
                           format_speedup, write_experiment_results)

GAMES, GAME_LABELS = select_games(seed=1322)

STEP = 1322
MAX_STEPS   = 2_000
MAX_SECONDS = 300

CONDITIONS = ['k50', 'base']
LABELS     = {'k50': 'K50', 'base': 'BASE'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1322')
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
K_INF_K50  = 50   # key change
K_INF_BASE = 5    # standard
W_CLIP  = 100.0
T_MIN   = 0.01

LOSS_CHECKPOINTS = [500, 2000]   # for compression ratio

SEED_A = 0
SEED_B = 1


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

class K50Substrate:
    """Multi-layer LPL K=50, shared representation (action from h1, no W3).

    K=50 gives 10× more inference per game step → closer to backprop approximation.
    Action selection reads h1 directly (shared representation, same as 1321).
    """

    K_INF = K_INF_K50

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.W1 = np.eye(H1_DIM, ENC_DIM, dtype=np.float32) * 0.1
        self.W2 = np.eye(H2_DIM, H1_DIM, dtype=np.float32) * 0.1
        self._W1_init = self.W1.copy()
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.step = 0
        self._action_counts = np.zeros(n_actions, np.float32)
        self._recent_pred_losses = deque(maxlen=50)
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def _centered_encode(self, obs_raw):
        x = _enc_frame(np.asarray(obs_raw, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def _infer(self, enc):
        h1 = self.W1 @ enc
        h2 = self.W2 @ h1
        for _ in range(self.K_INF):
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

        # Action from h1 (shared representation)
        if self.n_actions <= H1_DIM:
            logits = h1[:self.n_actions].astype(np.float32)
        else:
            indices = np.arange(self.n_actions) % H1_DIM
            logits = h1[indices].astype(np.float32)

        std_s = float(np.std(logits))
        T = max(1.0 / (std_s + 1e-8), T_MIN)
        l = logits * T
        l -= l.max()
        probs = np.exp(l)
        probs /= probs.sum()
        action = int(np.random.choice(self.n_actions, p=probs))
        self._action_counts[action] += 1

        self.W1 += ETA * np.outer(h1, e1)
        self.W2 += ETA * np.outer(h2, e2)
        np.clip(self.W1, -W_CLIP, W_CLIP, out=self.W1)
        np.clip(self.W2, -W_CLIP, W_CLIP, out=self.W2)

        enc_norm_sq = float(np.dot(enc, enc))
        pred_loss = float(np.dot(e1, e1)) / (enc_norm_sq + 1e-8)
        self._recent_pred_losses.append(pred_loss)
        for ck in self._pred_loss_at:
            if self._pred_loss_at[ck] is None and self.step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_pred_losses)), 6)

        return action

    def update_after_step(self, obs_next, action, reward):
        pass

    def on_level_transition(self):
        pass

    def compute_weight_drift(self):
        return float(np.linalg.norm(self.W1 - self._W1_init))

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(500)
        l_late = self._pred_loss_at.get(2000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)


class BaseSubstrate(K50Substrate):
    """Standard multi-layer LPL K=5, W3 Hebbian (1313 style)."""

    K_INF = K_INF_BASE

    def __init__(self, n_actions):
        super().__init__(n_actions)
        self.W3 = np.zeros((n_actions, H2_DIM), dtype=np.float32)

    def process(self, obs_raw):
        self.step += 1
        enc = self._centered_encode(obs_raw)
        h1, h2, e1, e2 = self._infer(enc)

        # Standard W3-based action
        action_scores = self.W3 @ h2
        std_s = float(np.std(action_scores))
        T = max(1.0 / (std_s + 1e-8), T_MIN)
        logits = action_scores * T
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(np.random.choice(self.n_actions, p=probs))
        self._action_counts[action] += 1

        action_onehot = np.zeros(self.n_actions, np.float32)
        action_onehot[action] = 1.0
        self.W1 += ETA * np.outer(h1, e1)
        self.W2 += ETA * np.outer(h2, e2)
        self.W3 += ETA * np.outer(action_onehot, h2)
        np.clip(self.W1, -W_CLIP, W_CLIP, out=self.W1)
        np.clip(self.W2, -W_CLIP, W_CLIP, out=self.W2)
        np.clip(self.W3, -W_CLIP, W_CLIP, out=self.W3)

        enc_norm_sq = float(np.dot(enc, enc))
        pred_loss = float(np.dot(e1, e1)) / (enc_norm_sq + 1e-8)
        self._recent_pred_losses.append(pred_loss)
        for ck in self._pred_loss_at:
            if self._pred_loss_at[ck] is None and self.step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_pred_losses)), 6)

        return action


def make_substrate(condition, n_actions):
    if condition == 'k50':
        return K50Substrate(n_actions=n_actions)
    return BaseSubstrate(n_actions=n_actions)


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

def run_episode(env, substrate, n_actions, solver_level_steps, seed, max_steps=MAX_STEPS):
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

    while steps < max_steps:
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

    return {
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(arc_score, 6),
        'I3_cv': i3_cv,
        'action_kl': action_kl,
        'wdrift': round(substrate.compute_weight_drift(), 4),
        'pred_loss_traj': substrate.get_pred_loss_trajectory(),
        'compression_ratio': substrate.get_compression_ratio(),
    }


# ---------------------------------------------------------------------------
# Game runner — try1 then try2 on same substrate
# ---------------------------------------------------------------------------

def run_game(condition, game_name, solver_level_steps):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    substrate = make_substrate(condition, n_actions)

    result_try1 = run_episode(env, substrate, n_actions, solver_level_steps, seed=SEED_A)
    result_try2 = run_episode(env, substrate, n_actions, solver_level_steps, seed=SEED_B)

    l1_try1 = result_try1['level_first_step'].get(1)
    l1_try2 = result_try2['level_first_step'].get(1)

    if l1_try1 is not None and l1_try2 is not None and l1_try2 > 0:
        speedup = round(l1_try1 / l1_try2, 4)
    elif l1_try1 is None and l1_try2 is not None:
        speedup = float('inf')
    elif l1_try1 is not None and l1_try2 is None:
        speedup = 0.0
    else:
        speedup = None

    return {
        'game': game_name,
        'condition': condition,
        'try1': result_try1,
        'try2': result_try2,
        'second_exposure_speedup': speedup,
        'compression_ratio': result_try1['compression_ratio'],
        'n_actions': n_actions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — Multi-layer LPL K=50 + shared representation")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Conditions: {CONDITIONS}")
    print(f"K50: K=50 inference iterations, shared rep (no W3, action from h1)")
    print(f"BASE: K=5 inference iterations, W3 Hebbian (1313 standard)")
    print(f"Primary metric: second_exposure_speedup. Diagnostic: compression_ratio.")
    print(f"6 runs (3 games × 2 conditions), try1+try2 per run (4K steps each).")
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
    speedup_by_game_cond = {}
    cr_by_game_cond = {}

    for game in GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")
        solver_steps = solver_steps_cache[game]

        t0 = time.time()
        for condition in CONDITIONS:
            result = run_game(condition, game, solver_steps)
            all_results.append(result)
            speedup_by_game_cond[(label, condition)] = result['second_exposure_speedup']
            cr_by_game_cond[(label, condition)] = result['compression_ratio']

        elapsed = time.time() - t0
        print(masked_run_log(label, elapsed))

        game_results = [r for r in all_results if r['game'] == game]
        out_path = os.path.join(RESULTS_DIR, label_filename(label, STEP))
        with open(out_path, 'w') as f:
            for r in game_results:
                masked_r = {k: v for k, v in r.items() if k != 'game'}
                masked_r['label'] = label
                f.write(json.dumps(masked_r, default=str) + '\n')

    # Chain speedup per condition
    speedup_by_condition = {}
    for cond in CONDITIONS:
        values = [v for (_, c), v in speedup_by_game_cond.items()
                  if c == cond and v is not None and v != float('inf') and v > 0]
        if values:
            speedup_by_condition[cond] = round(float(np.mean(values)), 4)
        else:
            any_inf = any(v == float('inf') for (_, c), v in speedup_by_game_cond.items()
                          if c == cond)
            speedup_by_condition[cond] = float('inf') if any_inf else None

    # Chain compression ratio per condition
    cr_by_condition = {}
    for cond in CONDITIONS:
        values = [v for (_, c), v in cr_by_game_cond.items()
                  if c == cond and v is not None]
        cr_by_condition[cond] = round(float(np.mean(values)), 4) if values else None

    write_experiment_results(RESULTS_DIR, STEP, speedup_by_condition, all_results, CONDITIONS)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP {STEP} — RESULT\n")
    for cond in CONDITIONS:
        lbl = LABELS[cond]
        sp = speedup_by_condition.get(cond)
        cr = cr_by_condition.get(cond)
        cr_str = f"{cr:.4f}" if cr is not None else "N/A"
        print(f"  {lbl}: second_exposure_speedup = {format_speedup(sp)}  |  cr = {cr_str}")
    print()

    print("  Per-game breakdown:")
    for label in sorted(set(l for l, _ in speedup_by_game_cond)):
        for cond in CONDITIONS:
            sp = speedup_by_game_cond.get((label, cond))
            cr = cr_by_game_cond.get((label, cond))
            cr_str = f"{cr:.4f}" if cr is not None else "N/A"
            print(f"    {label} / {LABELS[cond]}: speedup={format_speedup(sp)}  cr={cr_str}")
    print()

    sp_k50 = speedup_by_condition.get('k50')
    sp_base = speedup_by_condition.get('base')
    cr_k50 = cr_by_condition.get('k50')
    CR_REF = 0.93  # step 1310 reference

    print("KILL ASSESSMENT:")
    kill = False

    if sp_k50 is None or sp_k50 == 0.0:
        # No L1 or degraded — check compression for diagnostic
        if cr_k50 is not None and cr_k50 >= CR_REF:
            print(f"  >>> KILL: K50 speedup=N/A AND cr={cr_k50:.4f} ≥ {CR_REF} (no compression gain)")
            print(f"  >>> K=50 doesn't help. LPL is fundamentally weaker than gradient.")
            kill = True
        elif cr_k50 is not None and cr_k50 < 0.5:
            print(f"  >>> KILL: K50 speedup=N/A AND cr={cr_k50:.4f} < 0.5 (compresses but no task)")
            print(f"  >>> Better compression but shared rep insufficient for task progress.")
            kill = True
        elif cr_k50 is not None:
            print(f"  >>> KILL: K50 speedup=N/A, cr={cr_k50:.4f} (between 0.5-0.93, partial improvement)")
            kill = True
        else:
            print(f"  >>> KILL: K50 speedup=N/A, cr=N/A")
            kill = True
    elif sp_base is None:
        if sp_k50 == float('inf') or (sp_k50 is not None and sp_k50 > 0):
            print(f"  >>> SIGNAL: K50 speedup={format_speedup(sp_k50)}, BASE=N/A → K50 reaches L1, BASE can't")
            print(f"  >>> NO KILL — K=50 + shared rep improves reachability")
        else:
            print(f"  >>> KILL: K50 speedup=0.0 (degraded), BASE=N/A")
            kill = True
    elif sp_k50 > sp_base:
        print(f"  >>> SIGNAL: K50 speedup={format_speedup(sp_k50)} > BASE={format_speedup(sp_base)}")
        print(f"  >>> NO KILL — K=50 + shared rep improves second-exposure learning")
    else:
        print(f"  >>> KILL: K50 speedup={format_speedup(sp_k50)} ≤ BASE={format_speedup(sp_base)}")
        kill = True

    if cr_k50 is not None:
        if cr_k50 < CR_REF:
            print(f"  >>> PREDICTION 1 CONFIRMED: K50 cr={cr_k50:.4f} < {CR_REF} (better compression)")
        else:
            print(f"  >>> PREDICTION 1 WRONG: K50 cr={cr_k50:.4f} ≥ {CR_REF} (no compression gain from K=50)")

    if not kill:
        print("  >>> Diagnostics in diagnostics.json (not kill criteria)")
    print(sep)
    print(f"\nResults: {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()
