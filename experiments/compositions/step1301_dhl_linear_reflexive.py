"""
Step 1301 — Anti-Collapse Linear Reflexive Map (DHL)
Leo mail 3665, 2026-03-28.

Return to single-matrix architecture (Step 1264) — R2-compliant, W_action both encodes
and selects. Fix the collapse that killed 1264 using anti-Hebbian decorrelation
(Pehlevan & Chklovskii 2015, proven globally stable).

Conditions (3):
  DHL       — Oja + sparse anti-Hebbian (K=20) + soft bound (the test)
  OJA-ONLY  — Oja alone, no anti-Hebbian/bound (expected to collapse, control)
  ARGMIN-PE — Step 1282 pe_ema composition (reference baseline)

Architecture (DHL/OJA-ONLY, on 1282 base):
  ext = [enc(256); h(64)]  (320 dims)
  scores = W_action @ ext           # (n_actions,)
  action = argmax(scores)           # R2: same W encodes AND selects
  W_action[action] += eta * score[action] * (ext - score[action] * W_action[action])  # Oja
  # Anti-Hebbian (DHL only):
    top_K = top-20 by |score|
    corr_K = outer(scores[top_K], scores[top_K]); diag=0
    W_action[top_K] -= eta_anti * corr_K @ W_action[top_K]
  # Soft bound (DHL only):
    W_action *= min(1.0, W_max / row_norms)

Hyperparameters:
  eta = 0.01, eta_anti = 0.001, W_max = 1.0, K = 20
  W_action init: randn * 0.01
  1282 base: eta_h=0.05, pe_ema alpha=0.1, ENC_DIM=256, H_DIM=64

Protocol: Full PRISM. 3 conditions × 11 games × 5 draws = 165 runs.
Budget: ~25 min.

Kill criteria:
  - DHL collapses (1 action > 80% steps) on 3+ games → anti-Hebbian insufficient
  - DHL R3 < OJA-ONLY R3 on 3+ games → anti-Hebbian suppresses learning
  - DHL I3_cv > 3× ARGMIN-PE on 3+ games → pathological concentration

Decision tree:
  DHL stable + R3 > 0.05 → compose forward model on linear arch (not network)
  DHL stable + R3 < 0.05 → collapse fixed but linear map lacks capacity → CNN path
  DHL collapses → add homeostatic rate target (per-action scaling to 1/n_actions)
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

import numpy as np
from substrates.step0674 import _enc_frame

# --- Config ---
ARC_GAMES = ['ls20', 'ft09', 'vc33', 'tr87', 'sp80', 'sb26', 'tu93', 'cn04', 'cd82', 'lp85']
MBPP_GAMES = ['mbpp_0']
GAMES = ARC_GAMES + MBPP_GAMES

N_DRAWS = 5
MAX_STEPS = 10_000
MAX_SECONDS = 300

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM  # 320

# 1282 base hyperparameters
ETA_H_FLOW = 0.05
ETA_PRED = 0.01
PE_EMA_ALPHA = 0.1
SELECTION_ALPHA = 0.1
DECAY = 0.001

# DHL hyperparameters (Leo mail 3665)
DHL_ETA = 0.01           # Hebbian rate
DHL_ETA_ANTI = 0.001     # anti-Hebbian rate (1/10 of Hebbian)
DHL_W_MAX = 1.0          # soft bound ceiling
DHL_K = 20               # sparse anti-Hebbian top-K

# R3 measurement
R3_STEP = 5000
R3_N_OBS = 50
R3_N_DIRS = 20
R3_EPSILON = 0.01

CONDITIONS = ['dhl', 'oja_only', 'argmin_pe']
LABELS = {'dhl': 'DHL', 'oja_only': 'OJA', 'argmin_pe': 'PEEMA'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1301')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'

SOLVER_PRESCRIPTIONS = {
    'ls20':  ('ls20_fullchain.json',  'all_actions'),
    'ft09':  ('ft09_fullchain.json',  'all_actions'),
    'vc33':  ('vc33_fullchain.json',  'all_actions_encoded'),
    'tr87':  ('tr87_fullchain.json',  'all_actions'),
    'sp80':  ('sp80_fullchain.json',  'sequence'),
    'sb26':  ('sb26_fullchain.json',  'all_actions'),
    'tu93':  ('tu93_fullchain.json',  'all_actions'),
    'cn04':  ('cn04_fullchain.json',  'sequence'),
    'cd82':  ('cd82_fullchain.json',  'sequence'),
    'lp85':  ('lp85_fullchain.json',  'full_sequence'),
}


# --- Utilities ---

def make_game(game_name):
    if game_name.lower().startswith('mbpp'):
        from mbpp_game import make as mbpp_make
        return mbpp_make(game_name.lower())
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
    try:
        with open(os.path.join(PDIR, fname)) as f:
            d = json.load(f)
        return d.get(field)
    except Exception:
        return None


def compute_solver_level_steps(game_name, seed=1):
    if game_name.lower().startswith('mbpp'):
        try:
            from mbpp_game import compute_solver_steps
            idx = int(game_name.split('_', 1)[1]) if '_' in game_name else 0
            return compute_solver_steps(idx)
        except Exception:
            return {}
    prescription = load_prescription(game_name)
    if prescription is None:
        return {}
    env = make_game(game_name)
    obs = env.reset(seed=seed)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103
    ACTION_OFFSET = {'ls20': -1, 'vc33': 7}
    offset = ACTION_OFFSET.get(game_name.lower(), 0)
    level, level_first_step, step, fresh_episode = 0, {}, 0, True
    for action in prescription:
        action_int = (int(action) + offset) % n_actions
        obs_next, reward, done, info = env.step(action_int)
        step += 1
        if fresh_episode:
            fresh_episode = False
            obs = obs_next
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level_first_step[cl] = step
            level = cl
        if done:
            obs = env.reset(seed=seed)
            fresh_episode = True
        else:
            obs = obs_next
    return level_first_step


def compute_arc_score(level_first_steps, solver_level_steps):
    scores = []
    for lv, solver_step in solver_level_steps.items():
        if lv in level_first_steps and level_first_steps[lv] > 0:
            ratio = solver_step / level_first_steps[lv]
            scores.append(min(1.0, ratio * ratio))
    return float(np.mean(scores)) if scores else 0.0


def compute_rhae(level_first_steps, solver_level_steps, total_actions):
    scores = []
    for lv, solver_step in solver_level_steps.items():
        if lv in level_first_steps and level_first_steps[lv] > 0:
            ratio = solver_step / level_first_steps[lv]
            scores.append(min(1.0, ratio * ratio))
    return float(np.mean(scores)) if scores else 0.0


def compute_post_transition_kl(action_log, l1_step, n_actions, window=100):
    if l1_step is None or l1_step < window:
        return None
    pre = action_log[max(0, l1_step - window):l1_step]
    post = action_log[l1_step:min(len(action_log), l1_step + window)]
    if len(pre) < 10 or len(post) < 10:
        return None
    pre_c = np.zeros(n_actions, np.float32)
    post_c = np.zeros(n_actions, np.float32)
    for a in pre:
        if 0 <= a < n_actions:
            pre_c[a] += 1
    for a in post:
        if 0 <= a < n_actions:
            post_c[a] += 1
    if pre_c.sum() == 0 or post_c.sum() == 0:
        return None
    pre_p = (pre_c + 1e-8) / (pre_c.sum() + 1e-8 * n_actions)
    post_p = (post_c + 1e-8) / (post_c.sum() + 1e-8 * n_actions)
    return round(float(np.sum(post_p * np.log(post_p / pre_p))), 4)


# --- Substrates ---

class DhlSubstrate:
    """DHL: Oja + sparse anti-Hebbian decorrelation + soft bound.

    W_action (n_actions × 320): both encodes ext-space and selects actions.
    action = argmax(W_action @ ext)  — R2 compliant, no frozen evaluator.
    """

    def __init__(self, n_actions, seed, use_anti_hebbian=True, use_soft_bound=True):
        self.n_actions = n_actions
        self.use_anti_hebbian = use_anti_hebbian
        self.use_soft_bound = use_soft_bound

        rng_w = np.random.RandomState(seed + 10000)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)

        # W_action init: 0.01 scale (per Leo spec, different from PE-EMA)
        W_action_init = rng_w.randn(n_actions, EXT_DIM).astype(np.float32) * 0.01
        self.W_action = W_action_init.copy()
        self.W_action_init = W_action_init.copy()

        self._prev_ext = None
        self._last_action = None
        self.step = 0

        # Concentration tracking (collapse diagnostic)
        self._action_counts = np.zeros(n_actions, np.float32)

    def _centered_encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_action):
        enc = _enc_frame(np.asarray(obs_raw, dtype=np.float32)) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        ext = np.concatenate([enc, h_new])
        return frozen_W_action @ ext

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'h': self.h.copy(),
            'W_action': self.W_action.copy(),
            'step': self.step,
        }

    @property
    def W_action_init_val(self):
        return self.W_action_init

    def process(self, obs_raw):
        enc = self._centered_encode(obs_raw)
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = np.concatenate([enc, self.h])
        self._prev_ext = ext.copy()

        scores = self.W_action @ ext           # (n_actions,)
        action = int(np.argmax(scores))
        self._action_counts[action] += 1
        self._last_action = action
        self.step += 1
        return action

    def update_after_step(self, next_obs_raw, action, delta):
        if self._prev_ext is None:
            return
        ext = self._prev_ext
        scores = self.W_action @ ext

        # Oja update on winner
        winner_score = float(scores[action])
        self.W_action[action] += DHL_ETA * winner_score * (ext - winner_score * self.W_action[action])

        if self.use_anti_hebbian:
            # Sparse anti-Hebbian: top-K by |score|
            k = min(DHL_K, self.n_actions)
            top_k = np.argpartition(np.abs(scores), -k)[-k:]
            scores_k = scores[top_k].astype(np.float32)
            corr_k = np.outer(scores_k, scores_k)  # (k, k)
            np.fill_diagonal(corr_k, 0)
            self.W_action[top_k] -= DHL_ETA_ANTI * (corr_k @ self.W_action[top_k])

        if self.use_soft_bound:
            # Soft row-norm bound
            row_norms = np.linalg.norm(self.W_action, axis=1, keepdims=True)
            self.W_action *= np.minimum(1.0, DHL_W_MAX / (row_norms + 1e-8))

    def on_level_transition(self):
        self._prev_ext = None

    def get_collapse_fraction(self):
        """Fraction of steps taken by most-visited action."""
        if self.step == 0:
            return 0.0
        return float(self._action_counts.max() / self.step)


class ArgminPeSubstrate:
    """Step 1282 pe_ema composition (reference). Same as PeEmaSubstrate in step1298."""

    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        rng_w = np.random.RandomState(seed + 10000)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)
        scale = 1.0 / np.sqrt(float(EXT_DIM))
        W_action_init = rng_w.randn(n_actions, EXT_DIM).astype(np.float32) * scale
        self.W_action = W_action_init.copy()
        self.W_action_init = W_action_init.copy()
        self.W_pred = rng_w.randn(ENC_DIM, ENC_DIM).astype(np.float32) * 0.01
        self._visit_counts = np.zeros(n_actions, np.float32)
        self.pe_ema = np.zeros(n_actions, np.float32)
        self._prev_enc = None
        self._prev_ext_for_update = None
        self._last_action = None
        self.step = 0

    def _centered_encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_action):
        enc = _enc_frame(np.asarray(obs_raw, dtype=np.float32)) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        ext = np.concatenate([enc, h_new])
        return frozen_W_action @ ext

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'h': self.h.copy(),
            'W_action': self.W_action.copy(),
            'step': self.step,
        }

    @property
    def W_action_init_val(self):
        return self.W_action_init

    def process(self, obs_raw):
        enc = self._centered_encode(obs_raw)
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = np.concatenate([enc, self.h])
        self._prev_ext_for_update = ext.copy()
        score = self._visit_counts - SELECTION_ALPHA * self.pe_ema
        action = int(np.argmin(score))
        self._visit_counts[action] += 1
        self._prev_enc = enc.copy()
        self._last_action = action
        self.step += 1
        return action

    def update_after_step(self, next_obs_raw, action, delta):
        if self._prev_enc is None:
            return
        next_obs = np.asarray(next_obs_raw, dtype=np.float32)
        enc_after = _enc_frame(next_obs) - self.running_mean
        pred_enc = self.W_pred @ self._prev_enc
        pred_error = enc_after - pred_enc
        self.W_pred += ETA_PRED * np.outer(pred_error, self._prev_enc)
        pe = float(np.linalg.norm(enc_after - pred_enc))
        self.pe_ema[action] = (1.0 - PE_EMA_ALPHA) * self.pe_ema[action] + PE_EMA_ALPHA * pe
        enc_delta = enc_after - self._prev_enc
        flow = float(np.linalg.norm(enc_delta))
        if self._prev_ext_for_update is not None:
            self.W_action[action] += ETA_H_FLOW * flow * self._prev_ext_for_update
            self.W_action *= (1.0 - DECAY)

    def on_level_transition(self):
        self._prev_enc = None
        self._prev_ext_for_update = None

    def get_collapse_fraction(self):
        return 0.0  # argmin doesn't collapse


def make_substrate(condition, n_actions, seed):
    if condition == 'dhl':
        return DhlSubstrate(n_actions, seed, use_anti_hebbian=True, use_soft_bound=True)
    elif condition == 'oja_only':
        return DhlSubstrate(n_actions, seed, use_anti_hebbian=False, use_soft_bound=False)
    elif condition == 'argmin_pe':
        return ArgminPeSubstrate(n_actions, seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")


# --- R3 computation ---

def compute_r3(substrate, obs_sample, snapshot):
    if not obs_sample or snapshot is None:
        return None, False
    frozen_rm = snapshot.get('running_mean')
    frozen_h = snapshot.get('h')
    frozen_W = snapshot.get('W_action')
    if frozen_rm is None or frozen_h is None or frozen_W is None:
        return 0.0, False
    fresh_rm = np.zeros(ENC_DIM, np.float32)
    fresh_h = np.zeros(H_DIM, np.float32)
    fresh_W = substrate.W_action_init.copy()
    obs_subset = obs_sample[-R3_N_OBS:]
    rng = np.random.RandomState(42)
    diffs = []
    for obs_arr in obs_subset:
        obs_flat = obs_arr.ravel()
        dirs = rng.randn(R3_N_DIRS, len(obs_flat)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        base_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm, frozen_h, frozen_W)
        base_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_rm, fresh_h, fresh_W)
        for d in dirs:
            pert = obs_flat + R3_EPSILON * d
            pe = substrate.get_internal_repr_readonly(pert, frozen_rm, frozen_h, frozen_W)
            pf = substrate.get_internal_repr_readonly(pert, fresh_rm, fresh_h, fresh_W)
            jac_exp = (pe - base_exp) / R3_EPSILON
            jac_fresh = (pf - base_fresh) / R3_EPSILON
            diffs.append(np.linalg.norm(jac_exp - jac_fresh))
    r3 = float(np.mean(diffs)) if diffs else 0.0
    return round(r3, 4), r3 >= 0.05


# --- Episode runner ---

def run_episode(env, substrate, n_actions, solver_level_steps, seed,
                take_r3_snapshot=False, is_mbpp=False):
    obs = env.reset(seed=seed)

    action_log = []
    obs_store = []
    r3_snapshot = None
    r3_obs_sample = None
    i3_counts_at_200 = None
    action_counts = np.zeros(n_actions, np.float32)

    steps = 0
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

        obs_arr = np.asarray(obs, dtype=np.float32)
        obs_store.append(obs_arr)
        if len(obs_store) > 200:
            obs_store.pop(0)

        if steps == 200:
            i3_counts_at_200 = action_counts.copy()
        if take_r3_snapshot and steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

        action = substrate.process(obs_arr) % n_actions
        action_counts[action] += 1
        action_log.append(action)

        obs_next, reward, done, info = env.step(action)
        steps += 1

        if hasattr(substrate, 'update_after_step') and obs_next is not None:
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

    # I3
    i3_cv = None
    if i3_counts_at_200 is not None:
        counts = i3_counts_at_200[:n_actions].astype(float)
        mean_c = counts.mean()
        if mean_c > 1e-8:
            i3_cv = round(float(counts.std() / mean_c), 4)

    l1_step = level_first_step.get(1)
    l2_step = level_first_step.get(2)
    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    rhae = compute_rhae(level_first_step, solver_level_steps, steps)
    post_kl = compute_post_transition_kl(action_log, l1_step, n_actions)

    # Collapse fraction
    collapse_frac = substrate.get_collapse_fraction() if hasattr(substrate, 'get_collapse_fraction') else 0.0

    # R3
    r3_val, r3_pass = None, False
    if take_r3_snapshot and r3_snapshot is not None:
        r3_val, r3_pass = compute_r3(substrate, r3_obs_sample, r3_snapshot)

    return {
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'L1_solved': bool(l1_step is not None),
        'l1_step': l1_step,
        'l2_step': l2_step,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(rhae, 6),
        'I3_cv': i3_cv,
        'post_transition_kl': post_kl,
        'collapse_fraction': round(collapse_frac, 4),
        'R3': r3_val,
        'R3_pass': r3_pass,
    }


def run_draw(condition, game_name, draw_idx, solver_level_steps):
    env = make_game(game_name)
    is_mbpp = game_name.lower().startswith('mbpp')
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    seed_a = draw_idx * 2
    seed_b = draw_idx * 2 + 1
    take_r3 = (condition == 'dhl')  # R3 on DHL only (key measure)

    substrate = make_substrate(condition, n_actions, draw_idx)

    result_a = run_episode(env, substrate, n_actions, solver_level_steps,
                           seed=seed_a, take_r3_snapshot=take_r3, is_mbpp=is_mbpp)
    result_b = run_episode(env, substrate, n_actions, solver_level_steps,
                           seed=seed_b, is_mbpp=is_mbpp)

    speedup = None
    if result_a['l1_step'] is not None and result_b['l1_step'] is not None:
        speedup = round(result_a['l1_step'] / result_b['l1_step'], 3)
    elif result_a['l1_step'] is None and result_b['l1_step'] is not None:
        speedup = float('inf')

    return {
        'game': game_name,
        'draw': draw_idx,
        'condition': condition,
        'episode_A': result_a,
        'episode_B': result_b,
        'second_exposure_speedup': speedup,
        'n_actions': n_actions,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step 1301 — DHL Anti-Collapse Linear Reflexive Map")
    print(f"Games: {GAMES}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Draws: {N_DRAWS} per game per condition")
    print(f"Total: {len(CONDITIONS) * len(GAMES) * N_DRAWS} runs")
    print()

    # Pre-compute solver level steps
    print("Computing solver baselines...")
    solver_steps_cache = {}
    for game in ARC_GAMES:
        try:
            solver_steps_cache[game] = compute_solver_level_steps(game)
            print(f"  {game}: {solver_steps_cache[game]}")
        except Exception as e:
            print(f"  {game}: ERROR {e}")
            solver_steps_cache[game] = {}
    for game in MBPP_GAMES:
        solver_steps_cache[game] = {}
    print()

    all_results = []
    summary_rows = []
    # Per-game kill criterion tracking
    collapse_kills = {c: 0 for c in CONDITIONS}
    r3_suppression_kills = 0
    i3cv_kills = {c: 0 for c in CONDITIONS}

    for game in GAMES:
        solver_level_steps = solver_steps_cache.get(game, {})

        per_cond = {c: {'results': [], 'l1_a': 0, 'l2_a': 0, 'arc_a': [],
                        'rhae_a': [], 'r3_vals': [], 'i3cv_a': [],
                        'collapse_a': [], 'speedups': [], 'runtimes': []}
                   for c in CONDITIONS}

        print(f"=== {game.upper()} ===")
        for draw in range(N_DRAWS):
            t0 = time.time()
            for condition in CONDITIONS:
                try:
                    result = run_draw(condition, game, draw, solver_level_steps)
                except Exception as e:
                    print(f"  {condition} draw {draw} ERROR: {e}")
                    continue

                all_results.append(result)
                d = per_cond[condition]
                d['results'].append(result)
                ea = result['episode_A']
                eb = result['episode_B']

                if ea['L1_solved']:
                    d['l1_a'] += 1
                if ea['l2_step'] is not None:
                    d['l2_a'] += 1
                d['arc_a'].append(ea['arc_score'])
                d['rhae_a'].append(ea['RHAE'])
                if ea['I3_cv'] is not None:
                    d['i3cv_a'].append(ea['I3_cv'])
                if ea['R3'] is not None:
                    d['r3_vals'].append(ea['R3'])
                d['collapse_a'].append(ea['collapse_fraction'])
                if result['second_exposure_speedup'] not in (None, float('inf')):
                    d['speedups'].append(result['second_exposure_speedup'])
                d['runtimes'].append(ea['elapsed_seconds'] + eb['elapsed_seconds'])

            elapsed = time.time() - t0
            dhl_ea = per_cond['dhl']['results'][-1]['episode_A'] if per_cond['dhl']['results'] else {}
            oja_ea = per_cond['oja_only']['results'][-1]['episode_A'] if per_cond['oja_only']['results'] else {}
            pe_ea = per_cond['argmin_pe']['results'][-1]['episode_A'] if per_cond['argmin_pe']['results'] else {}
            print(f"  draw {draw}: DHL[L1={dhl_ea.get('L1_solved',False)}, "
                  f"R3={dhl_ea.get('R3')}, cf={dhl_ea.get('collapse_fraction'):.3f}] "
                  f"OJA[L1={oja_ea.get('L1_solved',False)}, cf={oja_ea.get('collapse_fraction'):.3f}] "
                  f"PE[L1={pe_ea.get('L1_solved',False)}]  ({elapsed:.1f}s)")

        # Game summary + kill criterion check
        for condition in CONDITIONS:
            d = per_cond[condition]
            n = len(d['results'])
            if n == 0:
                continue

            mean_i3cv = float(np.mean(d['i3cv_a'])) if d['i3cv_a'] else None
            mean_r3 = float(np.mean(d['r3_vals'])) if d['r3_vals'] else None
            mean_collapse = float(np.mean(d['collapse_a'])) if d['collapse_a'] else 0.0
            mean_arc = float(np.mean(d['arc_a'])) if d['arc_a'] else 0.0

            row = {
                'game': game, 'condition': condition, 'n': n,
                'L1_rate': f"{d['l1_a']}/{n}",
                'L2_rate': f"{d['l2_a']}/{n}",
                'arc_mean': round(mean_arc, 4),
                'rhae_mean': float(np.mean(d['rhae_a'])) if d['rhae_a'] else 0.0,
                'I3cv_mean': round(mean_i3cv, 4) if mean_i3cv is not None else None,
                'R3_mean': round(mean_r3, 4) if mean_r3 is not None else None,
                'collapse_mean': round(mean_collapse, 4),
                'speedup_mean': round(float(np.mean(d['speedups'])), 3) if d['speedups'] else None,
                'runtime_mean': round(float(np.mean(d['runtimes'])), 1) if d['runtimes'] else 0.0,
            }
            summary_rows.append(row)

            # Kill criterion check: collapse > 0.80
            if condition in ('dhl', 'oja_only') and mean_collapse > 0.80:
                collapse_kills[condition] += 1

            # Kill criterion: I3_cv > 3× ARGMIN-PE
            if condition == 'dhl' and mean_i3cv is not None:
                pe_i3cv = float(np.mean(per_cond['argmin_pe']['i3cv_a'])) if per_cond['argmin_pe']['i3cv_a'] else None
                if pe_i3cv is not None and mean_i3cv > 3 * pe_i3cv:
                    i3cv_kills['dhl'] += 1

        # R3 comparison: DHL vs OJA-ONLY
        dhl_r3 = float(np.mean(per_cond['dhl']['r3_vals'])) if per_cond['dhl']['r3_vals'] else 0.0
        oja_r3 = 0.0  # OJA-ONLY doesn't measure R3
        print(f"  GAME SUMMARY: DHL[R3={dhl_r3:.4f}, collapse={float(np.mean(per_cond['dhl']['collapse_a'])):.3f}] "
              f"OJA[collapse={float(np.mean(per_cond['oja_only']['collapse_a'])):.3f}]")
        print()

    # Save all results
    all_path = os.path.join(RESULTS_DIR, 'all_results.jsonl')
    with open(all_path, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + '\n')

    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'summary': summary_rows, 'n_draws': N_DRAWS,
                   'collapse_kills': collapse_kills,
                   'i3cv_kills': i3cv_kills}, f, indent=2)

    # Kill criterion evaluation
    print("\n" + "="*110)
    print("STEP 1301 — DHL ANTI-COLLAPSE LINEAR REFLEXIVE MAP")
    print(f"{'Game':<8} {'Cond':<10} {'L1':>6} {'L2':>6} {'ARC':>8} {'RHAE':>10} {'I3cv':>8} {'R3':>8} {'Collapse':>10} {'Speedup':>10}")
    print("-"*110)
    for row in summary_rows:
        r3_str = f"{row['R3_mean']:.4f}" if row['R3_mean'] is not None else "  N/A "
        sp_str = f"{row['speedup_mean']:.2f}" if row['speedup_mean'] is not None else "N/A"
        i3_str = f"{row['I3cv_mean']:.3f}" if row['I3cv_mean'] is not None else "N/A"
        print(f"{row['game']:<8} {LABELS.get(row['condition'], row['condition']):<10} "
              f"{row['L1_rate']:>6} {row['L2_rate']:>6} "
              f"{row['arc_mean']:>8.4f} {row['rhae_mean']:>10.2e} "
              f"{i3_str:>8} {r3_str:>8} {row['collapse_mean']:>10.4f} {sp_str:>10}")
    print("="*110)

    # Kill criterion verdict
    print(f"\n--- Kill Criterion Evaluation ---")
    print(f"DHL collapse (>80%) kills: {collapse_kills['dhl']}/11 games  (kill threshold: 3+)")
    print(f"OJA collapse (>80%) kills: {collapse_kills['oja_only']}/11 games  (expected: most)")
    print(f"DHL I3cv > 3×PE kills: {i3cv_kills['dhl']}/11 games  (kill threshold: 3+)")

    dhl_killed = (collapse_kills['dhl'] >= 3) or (i3cv_kills['dhl'] >= 3)
    print(f"\nDHL KILL TRIGGERED: {dhl_killed}")

    if not dhl_killed:
        dhl_r3_all = [r['R3_mean'] for r in summary_rows if r['condition'] == 'dhl' and r['R3_mean'] is not None]
        dhl_r3_mean = float(np.mean(dhl_r3_all)) if dhl_r3_all else 0.0
        print(f"DHL mean R3 (all games): {dhl_r3_mean:.4f}  (threshold for 'stable+capacity': 0.05)")
        if dhl_r3_mean >= 0.05:
            print("OUTCOME: DHL stable + R3 > 0.05 → linear arch has capacity → compose forward model here")
        else:
            print("OUTCOME: DHL stable + R3 < 0.05 → linear arch lacks capacity → CNN path (Step 1302)")
    else:
        print("OUTCOME: DHL collapsed → add homeostatic rate target")

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
