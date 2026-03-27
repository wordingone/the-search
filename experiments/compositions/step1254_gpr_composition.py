"""
Step 1254 — GPR Selection on 1251 Composition (Root Cause Test)

R3 hypothesis: the 1251 WiringA composition modifies its encoding via R3 (confirmed:
100/100), but argmin ignores h and alpha. Root cause test: replace argmin with
GPR salience-gated selection reading from attended_enc. If the R3-modified encoding
already carries action-relevant information, GPR + frozen random W_salience should
produce salience-action correlation rho > 0.3.

If rho is near zero: R3 modifies the encoding but the modifications are orthogonal
to game-relevant information. R3 = self-modification, not self-improvement.

NEW COMPONENT (C_GPR): GPR salience-gated selection.
  salience = W_salience @ attended_enc  [W_salience frozen random, NOT learned]
  if max(salience) < mean(salience) + std(salience): action = random
  else: action = argmax(salience)

Three-way comparison:
  Control C (argmin alone) — no R3 components
  WiringA/1251 (argmin + R3) — R3 exists, argmin ignores it [BASELINE FROM 1251]
  Step 1254 GPRWiringA — R3 exists, GPR reads from it [THIS EXPERIMENT]
"""
import sys, os, time, json, copy
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')

import numpy as np
from substrates.step0674 import _enc_frame

# ─── Config ───
GAMES = ['ls20', 'vc33', 'sp80']
N_DRAWS = 5
MAX_STEPS = 10_000
MAX_SECONDS = 300  # 5-min cap

# Shared hyperparameters (identical to 1251)
ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM   # 320
ETA_PRED = 0.01
ALPHA_EMA_RATE = 0.10
ALPHA_LO, ALPHA_HI = 0.10, 5.00
TRANSITION_SCALE = 0.30
SELF_OBS_SCALE = 0.05

# Instrumentation steps
I3_STEP = 200
I1_STEP = 1000
I4_STEPS = [100, 5000]
R3_STEP = 5000
R3_N_OBS = 100
R3_N_DIRS = 20
R3_EPSILON = 0.01
I4_WINDOW = 50

DIAG_DIR = os.path.join('B:/M/the-search/experiments', 'results', 'game_diagnostics')
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1254')

# ─── Hash ───
_HASH_H = np.random.RandomState(42).randn(12, ENC_DIM).astype(np.float32)

def hash_enc(x: np.ndarray) -> int:
    bits = (_HASH_H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits[:8], bitorder='big').tobytes().hex(), 16)


# ─── Game factory ───
def make_game(game_name: str):
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def get_n_actions(env, game_name: str) -> int:
    n = getattr(env, 'n_actions', None)
    if n is not None:
        return int(n)
    _table = {'ls20': 7, 'vc33': 4103, 'sp80': 4103, 'ft09': 4103}
    return _table.get(game_name.lower(), 4103)


def load_game_diag(game_name: str) -> dict:
    path = os.path.join(DIAG_DIR, f'{game_name.lower()}_diagnostic.json')
    with open(path) as f:
        return json.load(f)


def build_kb_profile(diag: dict) -> tuple:
    kb = diag.get('kb_responsiveness', {})
    delta = np.zeros(7, np.float32)
    responsive = np.zeros(7, bool)
    for i, key in enumerate([f'ACTION{j}' for j in range(1, 8)]):
        if key in kb:
            delta[i] = kb[key].get('delta_mean', 0.0)
            responsive[i] = kb[key].get('responsive', False)
    return delta, responsive


# ─── Utility ───
def action_entropy(action_seq: list, n_actions: int) -> float:
    if not action_seq:
        return 0.0
    counts = np.zeros(n_actions, np.float32)
    for a in action_seq:
        if 0 <= a < n_actions:
            counts[a] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def permutation_test_i1(within_dists, between_dists, n_perms=500, seed=0):
    rng = np.random.RandomState(seed)
    within = np.array(within_dists, dtype=np.float32)
    between = np.array(between_dists, dtype=np.float32)
    if len(within) == 0 or len(between) == 0:
        return 1.0
    observed = float(np.mean(between) - np.mean(within))
    all_dists = np.concatenate([within, between])
    n_w = len(within)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(len(all_dists))
        perm_w = all_dists[perm[:n_w]]
        perm_b = all_dists[perm[n_w:]]
        if float(np.mean(perm_b) - np.mean(perm_w)) >= observed:
            count += 1
    return count / n_perms


def spearman_rho(x, y):
    if len(x) < 2 or len(y) < 2:
        return None
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(np.float32)
    ry = np.argsort(np.argsort(y)).astype(np.float32)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = (np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom < 1e-8:
        return None
    return float(np.dot(rx, ry) / denom)


# ─────────────────────────────────────────────────────────────
# SUBSTRATES
# ─────────────────────────────────────────────────────────────

class ControlC:
    """C4 + C14 only. Argmin baseline. Identical to 1251."""

    def __init__(self, n_actions: int, seed: int):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.action_counts = np.zeros(n_actions, np.float32)
        self._last_enc = None
        self.step = 0

    def _centered_encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw: np.ndarray,
                                   frozen_running_mean: np.ndarray) -> np.ndarray:
        x = _enc_frame(np.asarray(obs_raw, dtype=np.float32))
        return x - frozen_running_mean

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, dtype=np.float32)
        enc = self._centered_encode(obs)
        self._last_enc = enc.copy()
        action = int(np.argmin(self.action_counts))
        self.action_counts[action] += 1
        self.step += 1
        return action

    def on_level_transition(self):
        pass

    def reset(self, seed: int):
        self.running_mean[:] = 0
        self.n_obs = 0
        self.action_counts[:] = 0
        self._last_enc = None
        self.step = 0
        self._rng = np.random.RandomState(seed)

    def get_state(self) -> dict:
        return {
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'action_counts': self.action_counts.copy(),
            'step': self.step,
        }

    def set_state(self, state: dict):
        self.running_mean = state['running_mean'].copy()
        self.n_obs = state['n_obs']
        self.action_counts = state['action_counts'].copy()
        self.step = state['step']


class GPRWiringA:
    """
    WiringA (all 7 components) with C14 (argmin) replaced by C_GPR.

    C_GPR: GPR salience-gated selection.
      salience = W_salience @ attended_enc  [W_salience: frozen random (n_actions, EXT_DIM)]
      if max(salience) < mean(salience) + std(salience): random ("I don't know")
      else: argmax(salience)  ("disinhibit winner")

    W_salience is FROZEN (never updated). This isolates whether the R3-modified
    encoding already carries useful structure, without learning a readout.

    Also tracks argmin counterfactual: what argmin WOULD have selected at each step
    (based on action_counts updated by GPR's actual selections).
    """

    def __init__(self, n_actions: int, seed: int):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rng_sub = np.random.RandomState(seed + 10000)
        rng_gpr = np.random.RandomState(seed + 20000)

        # C4
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0

        # C21/C26: Recurrent h — fixed random weights
        self.W_h = rng_sub.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rng_sub.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)

        # C25: Alpha attention
        self.alpha = np.ones(EXT_DIM, np.float32)
        self.pred_err_ema = np.zeros(EXT_DIM, np.float32)
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), np.float32)

        # C14: Global action counts — kept for argmin counterfactual only
        self.action_counts = np.zeros(n_actions, np.float32)

        # C1: Novelty tracking
        self.novelty_set = set()
        self.novelty_ema = 0.0

        # C15/C20: Transition detection
        self.transition_map = {}
        self.transition_inconsistency_ema = 0.0

        # C22: Self-observation (fixed random)
        self.W_self = rng_sub.randn(ENC_DIM, n_actions).astype(np.float32) * 0.01

        # C_GPR: Frozen random salience projection (n_actions × EXT_DIM)
        scale = 1.0 / np.sqrt(float(EXT_DIM))
        self.W_salience = rng_gpr.randn(n_actions, EXT_DIM).astype(np.float32) * scale

        # Per-action salience and state-change tracking (for salience-action correlation)
        self._sal_sum = np.zeros(n_actions, np.float64)
        self._sal_count = np.zeros(n_actions, np.int64)
        self._delta_sum = np.zeros(n_actions, np.float64)
        self._delta_count = np.zeros(n_actions, np.int64)

        # Argmin counterfactual tracking
        self._argmin_disagree = 0
        self._argmin_total = 0

        # History buffers
        self._prev_ext_enc = None
        self._prev_attended = None
        self._prev_action = None
        self._prev_obs_hash = None

        self.step = 0

    def _centered_encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def _update_alpha(self, ext_enc: np.ndarray):
        if self._prev_ext_enc is None or self._prev_action is None:
            return
        a_oh = np.zeros(self.n_actions, np.float32)
        a_oh[self._prev_action] = 1.0
        inp = np.concatenate([self._prev_ext_enc * self.alpha, a_oh])
        pred = self.W_pred @ inp
        error = (ext_enc * self.alpha) - pred
        err_norm = float(np.linalg.norm(error))
        if err_norm > 10.0:
            error = error * (10.0 / err_norm)
        if not np.any(np.isnan(error)):
            self.W_pred -= ETA_PRED * np.outer(error, inp)
            per_dim_err = np.abs(error)
            self.pred_err_ema = (1 - ALPHA_EMA_RATE) * self.pred_err_ema + ALPHA_EMA_RATE * per_dim_err
            raw_alpha = np.sqrt(self.pred_err_ema + 1e-8)
            mean_raw = raw_alpha.mean()
            if mean_raw > 1e-8 and not np.isnan(mean_raw):
                self.alpha = np.clip(raw_alpha / mean_raw, ALPHA_LO, ALPHA_HI)

    def _transition_signal(self, obs_hash: int, prev_hash) -> float:
        if prev_hash is None:
            return 0.0
        prev_nexts = self.transition_map.setdefault(prev_hash, set())
        was_inconsistent = len(prev_nexts) > 1
        prev_nexts.add(obs_hash)
        is_now_inconsistent = len(prev_nexts) > 1
        signal = 1.0 if (is_now_inconsistent and not was_inconsistent) else (
            0.5 if is_now_inconsistent else 0.0
        )
        self.transition_inconsistency_ema = (
            0.95 * self.transition_inconsistency_ema + 0.05 * float(is_now_inconsistent)
        )
        return signal

    def get_internal_repr_readonly(self, obs_raw: np.ndarray,
                                   frozen_running_mean: np.ndarray,
                                   frozen_h: np.ndarray,
                                   frozen_alpha: np.ndarray,
                                   frozen_action_counts: np.ndarray) -> np.ndarray:
        """R3 Jacobian read-only: same as WiringA (attended = alpha * [enc, h])."""
        obs = np.asarray(obs_raw, dtype=np.float32)
        x = _enc_frame(obs)
        enc_raw = x - frozen_running_mean
        self_obs = frozen_action_counts / (frozen_action_counts.sum() + 1e-8)
        enc = enc_raw + SELF_OBS_SCALE * (self.W_self @ self_obs)
        h_new = np.tanh(self.W_h @ frozen_h + self.W_x @ enc)
        ext_enc = np.concatenate([enc, h_new])
        attended = frozen_alpha * ext_enc
        return attended  # 320D

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, dtype=np.float32)

        # C4
        enc_raw = self._centered_encode(obs)

        # C22: Self-obs feedback
        self_obs = self.action_counts / (self.action_counts.sum() + 1e-8)
        enc = enc_raw + SELF_OBS_SCALE * (self.W_self @ self_obs)

        obs_hash = hash_enc(enc_raw)

        # C21/C26: Recurrent h
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        ext_enc = np.concatenate([enc, self.h])  # 320D

        # C25: Alpha attention
        self._update_alpha(ext_enc)
        attended = self.alpha * ext_enc  # 320D

        # C15/C20: Transition detection
        trans_signal = self._transition_signal(obs_hash, self._prev_obs_hash)
        attended_mod = attended * (1 + TRANSITION_SCALE * trans_signal)

        # C1: Novelty tracking
        novel = obs_hash not in self.novelty_set
        if novel:
            self.novelty_set.add(obs_hash)
            self.novelty_ema = 0.9 * self.novelty_ema + 0.1 * 1.0
        else:
            self.novelty_ema = 0.9 * self.novelty_ema + 0.1 * 0.0

        # C_GPR: Salience-gated selection (reads from attended_mod)
        salience = self.W_salience @ attended_mod  # (n_actions,)
        sal_max = float(np.max(salience))
        sal_mean = float(np.mean(salience))
        sal_std = float(np.std(salience))
        threshold = sal_mean + sal_std

        if sal_max < threshold:
            action = self._rng.randint(0, self.n_actions)
        else:
            action = int(np.argmax(salience))

        # Argmin counterfactual (before updating action_counts)
        argmin_action = int(np.argmin(self.action_counts))
        self._argmin_total += 1
        if action != argmin_action:
            self._argmin_disagree += 1

        # Update action counts with GPR-selected action
        self.action_counts[action] += 1

        # Track per-action salience (absolute value of the selected action's salience)
        self._sal_sum[action] += float(abs(salience[action]))
        self._sal_count[action] += 1

        # Store for R3 / I1
        self._prev_attended = attended_mod.copy()

        # Store for next step
        self._prev_ext_enc = ext_enc.copy()
        self._prev_obs_hash = obs_hash
        self._prev_action = action
        self.step += 1

        return action

    def record_state_change(self, action: int, delta: float):
        """Called by run_single after env.step with ||next_obs - obs||_2."""
        if 0 <= action < self.n_actions:
            self._delta_sum[action] += delta
            self._delta_count[action] += 1

    def on_level_transition(self):
        self._prev_ext_enc = None
        self._prev_attended = None
        self._prev_action = None
        self._prev_obs_hash = None

    def reset(self, seed: int):
        self.running_mean[:] = 0
        self.n_obs = 0
        self.h[:] = 0
        self.alpha[:] = 1
        self.pred_err_ema[:] = 0
        self.W_pred[:] = 0
        self.action_counts[:] = 0
        self.novelty_set.clear()
        self.novelty_ema = 0.0
        self.transition_map.clear()
        self.transition_inconsistency_ema = 0.0
        self._sal_sum[:] = 0
        self._sal_count[:] = 0
        self._delta_sum[:] = 0
        self._delta_count[:] = 0
        self._argmin_disagree = 0
        self._argmin_total = 0
        self._prev_ext_enc = None
        self._prev_attended = None
        self._prev_action = None
        self._prev_obs_hash = None
        self.step = 0
        self._rng = np.random.RandomState(seed)

    def get_state(self) -> dict:
        return {
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'h': self.h.copy(),
            'alpha': self.alpha.copy(),
            'pred_err_ema': self.pred_err_ema.copy(),
            'W_pred': self.W_pred.copy(),
            'action_counts': self.action_counts.copy(),
            'novelty_ema': self.novelty_ema,
            'transition_inconsistency_ema': self.transition_inconsistency_ema,
            'step': self.step,
        }

    def set_state(self, state: dict):
        self.running_mean = state['running_mean'].copy()
        self.n_obs = state['n_obs']
        self.h = state['h'].copy()
        self.alpha = state['alpha'].copy()
        self.pred_err_ema = state['pred_err_ema'].copy()
        self.W_pred = state['W_pred'].copy()
        self.action_counts = state['action_counts'].copy()
        self.novelty_ema = state['novelty_ema']
        self.transition_inconsistency_ema = state['transition_inconsistency_ema']
        self.step = state['step']


# ─────────────────────────────────────────────────────────────
# INSTRUMENTATION
# ─────────────────────────────────────────────────────────────

def compute_i3(action_counts_200: np.ndarray, kb_delta: np.ndarray) -> tuple:
    n = min(len(action_counts_200), 7)
    if n < 2:
        return None, None
    freq = action_counts_200[:n].astype(np.float32)
    ref = kb_delta[:n].astype(np.float32)
    if ref.max() - ref.min() < 1e-6:
        return None, None
    rho = spearman_rho(freq, ref)
    if rho is None:
        return None, None
    return rho, bool(rho > 0.5)


def compute_i1(repr_level_log: list) -> dict:
    if len(repr_level_log) < 4:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}
    rng = np.random.RandomState(1)
    n = len(repr_level_log)
    within_dists = []
    between_dists = []
    for _ in range(200):
        i, j = rng.choice(n, 2, replace=False)
        r1, l1 = repr_level_log[i]
        r2, l2 = repr_level_log[j]
        d = cosine_dist(r1, r2)
        if l1 == l2:
            within_dists.append(d)
        else:
            between_dists.append(d)
    if len(within_dists) < 2 or len(between_dists) < 2:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}
    p_val = permutation_test_i1(within_dists, between_dists, n_perms=500)
    within_mean = float(np.mean(within_dists))
    between_mean = float(np.mean(between_dists))
    i1_pass = bool(within_mean < between_mean and p_val < 0.05)
    return {
        'within': round(within_mean, 4),
        'between': round(between_mean, 4),
        'p_value': round(p_val, 4),
        'pass': i1_pass,
    }


def compute_i4(action_log: list, n_actions: int) -> dict:
    def entropy_at(step):
        if step > len(action_log):
            return None
        window = action_log[max(0, step - I4_WINDOW): step]
        if len(window) < 5:
            return None
        return action_entropy(window, n_actions)
    h100 = entropy_at(100)
    h5000 = entropy_at(5000)
    if h100 is None or h5000 is None or h100 < 1e-6:
        return {'entropy_100': h100, 'entropy_5000': h5000, 'reduction_pct': None, 'pass': False}
    reduction_pct = float((h100 - h5000) / h100 * 100)
    return {
        'entropy_100': round(h100, 4),
        'entropy_5000': round(h5000, 4),
        'reduction_pct': round(reduction_pct, 2),
        'pass': bool(reduction_pct > 10.0),
    }


def compute_r3_jacobian(substrate, obs_sample: list,
                        frozen_state: dict, n_actions: int,
                        rng_seed: int = 42) -> dict:
    """Sketched Jacobian. Same as 1251, updated for GPRWiringA."""
    rng = np.random.RandomState(rng_seed)
    if not obs_sample:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    fresh_running_mean = np.zeros(ENC_DIM, np.float32)
    frozen_rm = frozen_state['running_mean']

    _is_stateful = isinstance(substrate, GPRWiringA)
    if _is_stateful:
        frozen_h = frozen_state['h']
        frozen_alpha = frozen_state['alpha']
        frozen_ac = frozen_state['action_counts']
        fresh_h = np.zeros(H_DIM, np.float32)
        fresh_alpha = np.ones(EXT_DIM, np.float32)
        fresh_ac = np.zeros(n_actions, np.float32)
    else:
        frozen_h = fresh_h = frozen_alpha = fresh_alpha = None
        frozen_ac = fresh_ac = None

    diffs = []
    for obs_raw in obs_sample[:R3_N_OBS]:
        obs_flat = np.asarray(obs_raw, dtype=np.float32)

        if _is_stateful:
            baseline_exp = substrate.get_internal_repr_readonly(
                obs_flat, frozen_rm, frozen_h, frozen_alpha, frozen_ac)
        else:
            baseline_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm)

        J_exp_cols = []
        for _ in range(R3_N_DIRS):
            direction = rng.randn(*obs_flat.shape).astype(np.float32)
            direction /= (np.linalg.norm(direction) + 1e-8)
            perturbed = obs_flat + R3_EPSILON * direction
            if _is_stateful:
                pert = substrate.get_internal_repr_readonly(
                    perturbed, frozen_rm, frozen_h, frozen_alpha, frozen_ac)
            else:
                pert = substrate.get_internal_repr_readonly(perturbed, frozen_rm)
            J_exp_cols.append((pert - baseline_exp) / R3_EPSILON)
        J_exp = np.stack(J_exp_cols, axis=0)

        if _is_stateful:
            baseline_fresh = substrate.get_internal_repr_readonly(
                obs_flat, fresh_running_mean, fresh_h, fresh_alpha, fresh_ac)
        else:
            baseline_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_running_mean)

        J_fresh_cols = []
        for _ in range(R3_N_DIRS):
            direction = rng.randn(*obs_flat.shape).astype(np.float32)
            direction /= (np.linalg.norm(direction) + 1e-8)
            perturbed = obs_flat + R3_EPSILON * direction
            if _is_stateful:
                pert = substrate.get_internal_repr_readonly(
                    perturbed, fresh_running_mean, fresh_h, fresh_alpha, fresh_ac)
            else:
                pert = substrate.get_internal_repr_readonly(perturbed, fresh_running_mean)
            J_fresh_cols.append((pert - baseline_fresh) / R3_EPSILON)
        J_fresh = np.stack(J_fresh_cols, axis=0)

        diff = float(np.linalg.norm(J_exp - J_fresh))
        diffs.append(diff)

    if not diffs:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
    jacobian_diff = float(np.mean(diffs))
    return {
        'jacobian_diff': round(jacobian_diff, 6),
        'pass': bool(jacobian_diff > 0.05),
        'n_obs_used': len(diffs),
    }


def compute_salience_action_corr(substrate) -> dict:
    """
    Salience-action correlation (GPRWiringA only).
    Spearman rho between per-action mean salience and per-action mean state-change magnitude.
    PASS: rho > 0.3
    Only uses actions that were visited at least once.
    """
    if not isinstance(substrate, GPRWiringA):
        return {'rho': None, 'pass': False, 'n_actions_visited': 0}

    visited_sal = substrate._sal_count > 0
    visited_delta = substrate._delta_count > 0
    visited = visited_sal & visited_delta

    n_visited = int(visited.sum())
    if n_visited < 5:
        return {'rho': None, 'pass': False, 'n_actions_visited': n_visited}

    mean_sal = substrate._sal_sum[visited] / (substrate._sal_count[visited] + 1e-8)
    mean_delta = substrate._delta_sum[visited] / (substrate._delta_count[visited] + 1e-8)

    rho = spearman_rho(mean_sal, mean_delta)
    return {
        'rho': round(float(rho), 4) if rho is not None else None,
        'pass': bool(rho is not None and rho > 0.3),
        'n_actions_visited': n_visited,
    }


# ─────────────────────────────────────────────────────────────
# SINGLE RUN HARNESS
# ─────────────────────────────────────────────────────────────

def run_single(game_name: str, condition: str, draw: int, seed: int,
               n_actions: int, kb_delta: np.ndarray) -> dict:
    print(f"  {game_name.upper()} | {condition} | draw={draw} | seed={seed} ...", end='', flush=True)

    if condition == 'control_c':
        substrate = ControlC(n_actions=n_actions, seed=seed)
    elif condition == 'gpr_wiring_a':
        substrate = GPRWiringA(n_actions=n_actions, seed=seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    env = make_game(game_name)
    obs = env.reset(seed=seed)

    action_log = []
    repr_log = []
    obs_store = []
    level_actions = {}

    i3_action_counts = None
    r3_snapshot = None
    r3_obs_sample = None

    steps = 0
    level = 0
    level_start_step = 0
    t_start = time.time()
    fresh_episode = True
    l1_step = l2_step = None

    # For state-change tracking
    prev_obs_flat = None
    prev_action_for_delta = None

    while steps < MAX_STEPS:
        elapsed = time.time() - t_start
        if elapsed > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            prev_obs_flat = None
            prev_action_for_delta = None
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        obs_flat = obs_arr.ravel()

        obs_store.append(obs_arr)
        if len(obs_store) > 200:
            obs_store.pop(0)

        if steps <= I1_STEP and steps % 20 == 0:
            if hasattr(substrate, '_prev_attended') and substrate._prev_attended is not None:
                repr_log.append((substrate._prev_attended.copy(), level))
            elif hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
                repr_log.append((substrate._last_enc.copy(), level))

        if steps == I3_STEP:
            i3_action_counts = substrate.action_counts[:min(7, n_actions)].copy()

        if steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

        action = substrate.process(obs_arr) % n_actions
        action_log.append(action)

        obs, reward, done, info = env.step(action)
        steps += 1

        # State-change tracking: delta = ||next_obs_flat - obs_flat||_2
        if obs is not None and not fresh_episode:
            next_flat = np.asarray(obs, dtype=np.float32).ravel()
            delta = float(np.linalg.norm(next_flat - obs_flat))
            if isinstance(substrate, GPRWiringA):
                substrate.record_state_change(action, delta)

        if fresh_episode:
            fresh_episode = False
            prev_obs_flat = obs_flat.copy()
            prev_action_for_delta = action
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1_step is None:
                l1_step = steps
            if cl == 2 and l2_step is None:
                l2_step = steps
            level_actions[level + 1] = steps - level_start_step
            level = cl
            level_start_step = steps
            substrate.on_level_transition()

        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            level = 0
            level_start_step = steps

    elapsed = time.time() - t_start

    # ── Instrumentation ──
    if hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
        repr_log.append((substrate._last_enc.copy(), level))
    elif hasattr(substrate, '_prev_attended') and substrate._prev_attended is not None:
        repr_log.append((substrate._prev_attended.copy(), level))

    i3_rho, i3_pass = compute_i3(i3_action_counts, kb_delta) if i3_action_counts is not None else (None, None)
    i1_result = compute_i1(repr_log)
    i4_result = compute_i4(action_log, n_actions)

    i5_pass = None
    i5_level_actions = None
    if l1_step is not None:
        i5_level_actions = {int(k): int(v) for k, v in level_actions.items() if k >= 1}
        if 1 in i5_level_actions and 2 in i5_level_actions:
            i5_pass = bool(i5_level_actions[2] < i5_level_actions[1])

    r3_result = compute_r3_jacobian(substrate, r3_obs_sample, r3_snapshot, n_actions) \
        if (r3_snapshot is not None and r3_obs_sample) else \
        {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    sal_result = compute_salience_action_corr(substrate)

    # Argmin counterfactual
    argmin_disagree_rate = None
    if isinstance(substrate, GPRWiringA) and substrate._argmin_total > 0:
        argmin_disagree_rate = round(substrate._argmin_disagree / substrate._argmin_total, 4)

    result = {
        'game': game_name.lower(),
        'condition': condition,
        'draw': draw,
        'seed': seed,
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': level,
        'L1_solved': bool(l1_step is not None),
        'L2_solved': bool(l2_step is not None),
        'l1_step': l1_step,
        'l2_step': l2_step,

        'I3_spearman_rho': round(i3_rho, 4) if i3_rho is not None else None,
        'I3_pass': i3_pass,

        'I1_within_dist': i1_result['within'],
        'I1_between_dist': i1_result['between'],
        'I1_p_value': i1_result['p_value'],
        'I1_pass': i1_result['pass'],

        'I4_entropy_100': i4_result['entropy_100'],
        'I4_entropy_5000': i4_result['entropy_5000'],
        'I4_reduction_pct': i4_result['reduction_pct'],
        'I4_pass': i4_result['pass'],

        'I5_level_actions': i5_level_actions,
        'I5_pass': i5_pass,

        'R3_jacobian_diff': r3_result['jacobian_diff'],
        'R3_n_obs_used': r3_result['n_obs_used'],
        'R3_pass': r3_result['pass'],

        # New: salience-action correlation
        'SAL_rho': sal_result['rho'],
        'SAL_pass': sal_result['pass'],
        'SAL_n_actions_visited': sal_result['n_actions_visited'],

        # New: argmin counterfactual
        'argmin_disagree_rate': argmin_disagree_rate,
    }

    l1_str = f"L1@{l1_step}" if l1_step else "L1=None"
    i3_str = f"I3ρ={i3_rho:.2f}" if i3_rho is not None else "I3=null"
    sal_str = f"SAL={sal_result['rho']:.3f}" if sal_result['rho'] is not None else "SAL=null"
    r3_str = f"R3={r3_result['jacobian_diff']:.4f}" if r3_result['jacobian_diff'] else "R3=null"
    dis_str = f"dis={argmin_disagree_rate:.2f}" if argmin_disagree_rate is not None else ""
    print(f" {l1_str} | {i3_str} | I4={'P' if i4_result['pass'] else 'F'} | {r3_str} | {sal_str} | {dis_str} | {elapsed:.1f}s")

    return result


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    t_global = time.time()

    print("=" * 70)
    print("STEP 1254 — GPR SELECTION ON 1251 COMPOSITION (ROOT CAUSE TEST)")
    print("=" * 70)
    print(f"Games: {GAMES}")
    print(f"Conditions: control_c, gpr_wiring_a")
    print(f"Draws: {N_DRAWS} per condition per game = 30 runs")
    print(f"Budget: {MAX_STEPS} steps / {MAX_SECONDS}s per run")
    print()
    print("WiringA/1251 results used as third baseline (from results_1251/).")
    print()

    for game_name in GAMES:
        try:
            diag = load_game_diag(game_name)
            kb_delta, kb_responsive = build_kb_profile(diag)
            game_n_actions = diag.get('n_actions', 4103)
        except Exception as e:
            print(f"ERROR: Could not load diagnostics for {game_name}: {e}")
            continue

        print(f"\n{'─'*60}")
        print(f"GAME: {game_name.upper()} | n_actions={game_n_actions}")
        print(f"{'─'*60}")

        for condition in ['control_c', 'gpr_wiring_a']:
            print(f"\n  Condition: {condition}")
            for draw in range(1, N_DRAWS + 1):
                # Same seed scheme as 1251: control=draw*100, wiring=draw*100+1
                seed = draw * 100 + (0 if condition == 'control_c' else 1)
                result = run_single(
                    game_name=game_name,
                    condition=condition,
                    draw=draw,
                    seed=seed,
                    n_actions=game_n_actions,
                    kb_delta=kb_delta,
                )
                all_results.append(result)

                incr_path = os.path.join(RESULTS_DIR, f"{game_name}_{condition}_draw{draw}.json")
                with open(incr_path, 'w') as f:
                    json.dump(result, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"STEP 1254 COMPLETE — {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*70}\n")

    # ── Summary ──
    conditions_run = ['control_c', 'gpr_wiring_a']
    stages = ['I3_pass', 'I1_pass', 'I4_pass', 'R3_pass', 'SAL_pass']
    print(f"{'Stage':<12}", end='')
    for c in conditions_run:
        label = 'CtrlC' if c == 'control_c' else 'GPR'
        print(f"  {label:>8}", end='')
    print()
    print("-" * 40)

    for stage in stages:
        print(f"{stage:<12}", end='')
        for c in conditions_run:
            vals = [r[stage] for r in all_results if r['condition'] == c and r.get(stage) is not None]
            rate = sum(vals) / len(vals) if vals else None
            s = f"{rate:.2f}" if rate is not None else "  null"
            print(f"  {s:>8}", end='')
        print()

    print()
    # Salience-action correlation per game × condition
    print("Salience-action correlation (GPR only):")
    for game in GAMES:
        gpr_runs = [r for r in all_results if r['condition'] == 'gpr_wiring_a' and r['game'] == game]
        rhos = [r['SAL_rho'] for r in gpr_runs if r.get('SAL_rho') is not None]
        dis_rates = [r['argmin_disagree_rate'] for r in gpr_runs if r.get('argmin_disagree_rate') is not None]
        rho_str = f"rho={np.mean(rhos):.3f}±{np.std(rhos):.3f}" if rhos else "rho=null"
        dis_str = f"disagree={np.mean(dis_rates):.2f}" if dis_rates else ""
        print(f"  {game.upper()}: {rho_str}  {dis_str}")

    print()
    # R3 comparison: 1254 GPR vs 1251 WiringA
    print("R3 comparison (GPR 1254 vs WiringA 1251 baseline):")
    for game in GAMES:
        gpr_r3 = [r['R3_jacobian_diff'] for r in all_results
                  if r['condition'] == 'gpr_wiring_a' and r['game'] == game
                  and r['R3_jacobian_diff'] is not None]
        r3_str = f"R3={np.mean(gpr_r3):.4f}±{np.std(gpr_r3):.4f}" if gpr_r3 else "R3=null"
        print(f"  {game.upper()}: {r3_str}")

    # Save full results
    out_path = os.path.join(RESULTS_DIR, 'step1254_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'step': 1254,
            'conditions': conditions_run,
            'games': GAMES,
            'n_draws': N_DRAWS,
            'total_runs': len(all_results),
            'runs': all_results,
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
