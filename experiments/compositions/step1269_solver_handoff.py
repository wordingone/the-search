"""
Step 1269 — Solver handoff: substrate observes transport to L_max, then acts.

Direction (Leo, 2026-03-27): L_max unreachable without solver (Step 1268 confirmed: 0/30 draws).
Build solver transport. Measure whether observation during transport helps at L_max.

Phase 1 (Transport): Solver replays prescription through levels 0 to L_max-1.
Phase 2 (Substrate): Substrate acts at L_max. Stage metrics measured HERE ONLY.

Conditions:
  physarum_observe: substrate observes (obs, action, obs_next) during transport,
                    updates running_mean + h + W_action (Physarum flow), then acts at L_max
  control_c_cold:   substrate starts fresh (zero state) at L_max, argmin acts

Reference (not re-run): control_c_l1 = Step 1268 SP80 control_c (Lmax=0/5 confirmed).

Game: SP80 (6 levels, 107-action prescription, working solver, seed=1).
Transport: deterministic (seed=1 for all draws). Diversity from substrate random init.
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')

import numpy as np
from substrates.step0674 import _enc_frame

# ─── Config ───
GAME = 'sp80'
N_LEVELS = 6        # SP80 has 6 levels (0-indexed: 0-5)
TARGET_LEVEL = N_LEVELS - 1  # = 5, the last level
N_DRAWS = 5
MAX_STEPS = 10_000  # substrate phase only
MAX_SECONDS = 280   # per run safety cap

TRANSPORT_SEED = 1  # fixed for deterministic transport across all draws

# Shared hyperparameters
ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM   # 320

# Physarum hyperparameters
ETA_FLOW = 0.01
DECAY = 0.001

# Instrumentation
I3_STEP = 200
I1_STEP = 1000
R3_STEP = 5000
R3_N_OBS = 100
R3_N_DIRS = 20
R3_EPSILON = 0.01
I4_WINDOW = 50
ATTN_SNAPSHOT_STEPS = [1000, 5000]

DIAG_DIR = os.path.join('B:/M/the-search/experiments', 'results', 'game_diagnostics')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1269')

# Prescription map
GAME_PRESCRIPTIONS = {
    'sp80': ('sp80_fullchain.json', 'sequence', 6),
}


def load_prescription(game_name: str):
    fname, field, n_levels = GAME_PRESCRIPTIONS[game_name]
    path = os.path.join(PDIR, fname)
    with open(path) as f:
        d = json.load(f)
    return d.get(field, [])


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
    return {'ls20': 7, 'vc33': 4103, 'sp80': 4103, 'ft09': 4103}.get(game_name.lower(), 4103)


def load_game_diag(game_name: str) -> dict:
    with open(os.path.join(DIAG_DIR, f'{game_name.lower()}_diagnostic.json')) as f:
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
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
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
        if float(np.mean(all_dists[perm[n_w:]]) - np.mean(all_dists[perm[:n_w]])) >= observed:
            count += 1
    return count / n_perms


def spearman_rho(x, y):
    if len(x) < 2 or len(y) < 2:
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float32)
    ry = np.argsort(np.argsort(y)).astype(np.float32)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom < 1e-8:
        return None
    return float(np.dot(rx, ry) / denom)


# ─────────────────────────────────────────────────────────────
# SUBSTRATES
# ─────────────────────────────────────────────────────────────

class ControlC:
    """C4 + C14 only. Cold start at L_max (no observation during transport)."""

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

    def get_internal_repr_readonly(self, obs_raw, frozen_running_mean):
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

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'step': self.step,
        }


class PhysarumSubstrate:
    """
    Physarum flow dynamics + argmin (tube-weighted).
    Extended with observe_step() for passive learning during solver transport.
    """

    def __init__(self, n_actions: int, seed: int):
        self.n_actions = n_actions
        rng = np.random.RandomState(seed)
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

        self.action_counts = np.zeros(n_actions, np.float32)

        self._sal_sum = np.zeros(n_actions, np.float64)
        self._sal_steps = 0

        self._delta_sum = np.zeros(n_actions, np.float64)
        self._delta_count = np.zeros(n_actions, np.int64)

        self._prev_enc_flow = None
        self._prev_ext = None
        self._last_action = None
        self._attn_snapshots = {}
        self._prev_repr = None
        self.step = 0

    def _centered_encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_action) -> np.ndarray:
        obs = np.asarray(obs_raw, dtype=np.float32)
        enc = _enc_frame(obs) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        ext = np.concatenate([enc, h_new])
        return frozen_W_action @ ext

    def observe_step(self, obs_prev_raw, action: int, obs_next_raw):
        """
        Passive learning during solver transport.
        Updates running_mean, h, W_action from (obs_prev, action, obs_next).
        Does NOT select actions, does NOT update SAL/delta tracking.
        """
        obs_prev = np.asarray(obs_prev_raw, dtype=np.float32)
        obs_next = np.asarray(obs_next_raw, dtype=np.float32)

        # C4: Update running mean from obs_prev
        enc_prev = self._centered_encode(obs_prev)

        # C26: Update recurrent h
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc_prev)
        ext = np.concatenate([enc_prev, self.h])

        # Physarum flow: state change delta from obs_prev → obs_next
        enc_next = _enc_frame(obs_next) - self.running_mean  # read-only, no mean update
        delta = enc_next - enc_prev
        flow = float(np.linalg.norm(delta))

        if 0 <= action < self.n_actions:
            self.W_action[action] += ETA_FLOW * flow * ext
            self.W_action *= (1.0 - DECAY)

        # Store for potential continuation
        self._prev_enc_flow = enc_prev.copy()
        self._prev_ext = ext.copy()
        self._last_action = action

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, dtype=np.float32)

        enc = self._centered_encode(obs)

        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = np.concatenate([enc, self.h])

        activation = self.W_action @ ext

        tube_thickness = np.linalg.norm(self.W_action, axis=1)
        effective_count = self.action_counts / (1.0 + tube_thickness)
        action = int(np.argmin(effective_count))
        self.action_counts[action] += 1

        self._sal_sum += activation.astype(np.float64)
        self._sal_steps += 1

        if self.step in ATTN_SNAPSHOT_STEPS:
            self._attn_snapshots[self.step] = {
                'act_max': float(activation.max()),
                'act_mean': float(activation.mean()),
                'act_std': float(activation.std()),
            }

        self._prev_repr = activation.copy()
        self._prev_enc_flow = enc.copy()
        self._prev_ext = ext.copy()
        self._last_action = action
        self.step += 1

        return action

    def update_flow(self, next_obs_raw):
        if self._prev_enc_flow is None or self._last_action is None:
            return
        next_obs = np.asarray(next_obs_raw, dtype=np.float32)
        enc_after = _enc_frame(next_obs) - self.running_mean
        delta = enc_after - self._prev_enc_flow
        flow = float(np.linalg.norm(delta))
        self.W_action[self._last_action] += ETA_FLOW * flow * self._prev_ext
        self.W_action *= (1.0 - DECAY)

    def record_state_change(self, action: int, delta: float):
        if 0 <= action < self.n_actions:
            self._delta_sum[action] += delta
            self._delta_count[action] += 1

    def on_level_transition(self):
        self._prev_enc_flow = None
        self._prev_ext = None

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'h': self.h.copy(),
            'W_action': self.W_action.copy(),
            'step': self.step,
        }

    def set_state(self, state):
        self.running_mean = state['running_mean'].copy()
        self.n_obs = state['n_obs']
        self.h = state['h'].copy()
        self.W_action = state['W_action'].copy()
        self.step = state['step']


# ─────────────────────────────────────────────────────────────
# INSTRUMENTATION
# ─────────────────────────────────────────────────────────────

def compute_i3(action_counts_200, kb_delta):
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


def compute_i1(repr_level_log):
    if len(repr_level_log) < 4:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}
    rng = np.random.RandomState(1)
    n = len(repr_level_log)
    within_dists, between_dists = [], []
    for _ in range(200):
        i, j = rng.choice(n, 2, replace=False)
        r1, l1 = repr_level_log[i]
        r2, l2 = repr_level_log[j]
        d = cosine_dist(r1, r2)
        (within_dists if l1 == l2 else between_dists).append(d)
    if len(within_dists) < 2 or len(between_dists) < 2:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}
    p_val = permutation_test_i1(within_dists, between_dists)
    wm = float(np.mean(within_dists))
    bm = float(np.mean(between_dists))
    return {'within': round(wm, 4), 'between': round(bm, 4),
            'p_value': round(p_val, 4), 'pass': bool(wm < bm and p_val < 0.05)}


def compute_i4(action_log, n_actions):
    def entropy_at(step):
        if step > len(action_log):
            return None
        window = action_log[max(0, step - I4_WINDOW): step]
        return action_entropy(window, n_actions) if len(window) >= 5 else None
    h100 = entropy_at(100)
    h5000 = entropy_at(5000)
    if h100 is None or h5000 is None or h100 < 1e-6:
        return {'entropy_100': h100, 'entropy_5000': h5000, 'reduction_pct': None, 'pass': False}
    reduction_pct = float((h100 - h5000) / h100 * 100)
    return {'entropy_100': round(h100, 4), 'entropy_5000': round(h5000, 4),
            'reduction_pct': round(reduction_pct, 2), 'pass': bool(reduction_pct > 10.0)}


def compute_r3_jacobian(substrate, obs_sample, frozen_state, n_actions, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    if not obs_sample:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    _is_phy = isinstance(substrate, PhysarumSubstrate)

    if _is_phy:
        frozen_rm = frozen_state['running_mean']
        frozen_h = frozen_state['h']
        frozen_W_action = frozen_state['W_action']
        fresh_rm = np.zeros(ENC_DIM, np.float32)
        fresh_h = np.zeros(H_DIM, np.float32)
        fresh_W_action = substrate.W_action_init
    else:
        frozen_rm = frozen_state['running_mean']
        fresh_rm = np.zeros(ENC_DIM, np.float32)

    diffs = []
    for obs_raw in obs_sample[:R3_N_OBS]:
        obs_flat = np.asarray(obs_raw, dtype=np.float32)

        if _is_phy:
            baseline_exp = substrate.get_internal_repr_readonly(
                obs_flat, frozen_rm, frozen_h, frozen_W_action)
        else:
            baseline_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm)

        J_exp_cols = []
        for _ in range(R3_N_DIRS):
            d = rng.randn(*obs_flat.shape).astype(np.float32)
            d /= (np.linalg.norm(d) + 1e-8)
            perturbed = obs_flat + R3_EPSILON * d
            if _is_phy:
                pert = substrate.get_internal_repr_readonly(
                    perturbed, frozen_rm, frozen_h, frozen_W_action)
            else:
                pert = substrate.get_internal_repr_readonly(perturbed, frozen_rm)
            J_exp_cols.append((pert - baseline_exp) / R3_EPSILON)
        J_exp = np.stack(J_exp_cols, axis=0)

        if _is_phy:
            baseline_fresh = substrate.get_internal_repr_readonly(
                obs_flat, fresh_rm, fresh_h, fresh_W_action)
        else:
            baseline_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_rm)

        J_fresh_cols = []
        for _ in range(R3_N_DIRS):
            d = rng.randn(*obs_flat.shape).astype(np.float32)
            d /= (np.linalg.norm(d) + 1e-8)
            perturbed = obs_flat + R3_EPSILON * d
            if _is_phy:
                pert = substrate.get_internal_repr_readonly(
                    perturbed, fresh_rm, fresh_h, fresh_W_action)
            else:
                pert = substrate.get_internal_repr_readonly(perturbed, fresh_rm)
            J_fresh_cols.append((pert - baseline_fresh) / R3_EPSILON)
        J_fresh = np.stack(J_fresh_cols, axis=0)

        diffs.append(float(np.linalg.norm(J_exp - J_fresh)))

    if not diffs:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
    jd = float(np.mean(diffs))
    return {'jacobian_diff': round(jd, 6), 'pass': bool(jd > 0.05), 'n_obs_used': len(diffs)}


def compute_sal(substrate):
    if not isinstance(substrate, PhysarumSubstrate):
        return {'rho': None, 'pass': False, 'n_actions_visited': 0}
    if substrate._sal_steps == 0:
        return {'rho': None, 'pass': False, 'n_actions_visited': 0}

    mean_activation = substrate._sal_sum / substrate._sal_steps
    visited = substrate._delta_count > 0
    n_visited = int(visited.sum())
    if n_visited < 5:
        return {'rho': None, 'pass': False, 'n_actions_visited': n_visited}

    mean_delta = substrate._delta_sum[visited] / (substrate._delta_count[visited] + 1e-8)
    rho = spearman_rho(mean_activation[visited].tolist(), mean_delta.tolist())
    return {
        'rho': round(float(rho), 4) if rho is not None else None,
        'pass': bool(rho is not None and rho > 0.3),
        'n_actions_visited': n_visited,
    }


# ─────────────────────────────────────────────────────────────
# TRANSPORT PHASE
# ─────────────────────────────────────────────────────────────

def transport_to_lmax(env, prescription, n_actions, target_level, transport_seed,
                      substrate=None, substrate_observes=False):
    """
    Replay prescription until info['level'] first reaches target_level.
    Uses same level-tracking logic as validate_solvers.py (cl from info).
    Returns: (obs_at_target_level_entry, transport_steps, success)
    """
    obs = env.reset(seed=transport_seed)
    transport_steps = 0
    level = 0
    prev_level = 0
    fresh_episode = False  # validate_solvers does NOT skip first action

    for action in prescription:
        obs_prev = obs
        action_int = int(action) % n_actions
        obs_next, reward, done, info = env.step(action_int)
        transport_steps += 1

        if fresh_episode:
            # Skip first step result after reset (same as validate_solvers)
            fresh_episode = False
            obs = obs_next
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0

        # Substrate observes this transition (before level check)
        if substrate_observes and substrate is not None:
            substrate.observe_step(obs_prev, action_int, obs_next)

        if cl >= target_level:
            # Entered target level — hand off to substrate here
            return obs_next, transport_steps, True

        if cl > prev_level:
            prev_level = cl
            if cl > level:
                level = cl
            if substrate is not None and substrate_observes:
                substrate.on_level_transition()

        if done:
            obs = env.reset(seed=transport_seed)
            fresh_episode = True
        else:
            obs = obs_next

    # Prescription exhausted without reaching target level
    return obs, transport_steps, False


# ─────────────────────────────────────────────────────────────
# SINGLE RUN
# ─────────────────────────────────────────────────────────────

def run_single(game_name, condition, draw, substrate_seed, n_actions, kb_delta,
               prescription, n_levels):
    """
    Phase 1: Transport to L_max via prescription.
    Phase 2: Substrate acts at L_max. Stage metrics measured here only.
    """
    print(f"  {game_name.upper()} | {condition} | draw={draw} | seed={substrate_seed} ...",
          end='', flush=True)

    target_level = n_levels - 1

    # Build substrate
    if condition == 'physarum_observe':
        substrate = PhysarumSubstrate(n_actions=n_actions, seed=substrate_seed)
    elif condition == 'control_c_cold':
        substrate = ControlC(n_actions=n_actions, seed=substrate_seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    substrate_observes = (condition == 'physarum_observe')

    # ── Phase 1: Transport ──
    env = make_game(game_name)
    t_transport_start = time.time()

    starting_obs, transport_steps, transport_ok = transport_to_lmax(
        env, prescription, n_actions, target_level, TRANSPORT_SEED,
        substrate=substrate, substrate_observes=substrate_observes
    )
    transport_elapsed = time.time() - t_transport_start

    if not transport_ok:
        print(f" TRANSPORT_FAILED (prescription exhausted at level<{target_level})")
        return {
            'game': game_name, 'condition': condition, 'draw': draw, 'seed': substrate_seed,
            'transport_ok': False, 'transport_steps': transport_steps,
            'transport_elapsed': round(transport_elapsed, 2),
            'lmax_level': target_level, 'lmax_solved': False,
            'passed': False,
        }

    # ── Phase 2: Substrate acts at L_max ──
    obs = starting_obs
    action_log = []
    repr_log = []
    obs_store = []

    i3_action_counts = None
    r3_snapshot = None
    r3_obs_sample = None

    steps = 0
    phase2_level = target_level  # start AT target_level; solved = advancing past it (done=True)
    lmax_solved = False
    lmax_solved_step = None
    t_phase2 = time.time()
    fresh_episode = True

    while steps < MAX_STEPS:
        if time.time() - t_phase2 > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=TRANSPORT_SEED)
            if hasattr(substrate, 'on_level_transition'):
                substrate.on_level_transition()
            fresh_episode = True
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        obs_flat = obs_arr.ravel()

        obs_store.append(obs_arr)
        if len(obs_store) > 200:
            obs_store.pop(0)

        # I1 repr log
        if steps <= I1_STEP and steps % 20 == 0:
            if hasattr(substrate, '_prev_repr') and substrate._prev_repr is not None:
                repr_log.append((substrate._prev_repr.copy(), phase2_level))
            elif hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
                repr_log.append((substrate._last_enc.copy(), phase2_level))

        if steps == I3_STEP:
            i3_action_counts = substrate.action_counts[:min(7, n_actions)].copy()

        if steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

        action = substrate.process(obs_arr) % n_actions
        action_log.append(action)

        obs_next, reward, done, info = env.step(action)
        steps += 1

        if obs_next is not None and not fresh_episode:
            next_flat = np.asarray(obs_next, dtype=np.float32).ravel()
            delta = float(np.linalg.norm(next_flat - obs_flat))
            if isinstance(substrate, PhysarumSubstrate):
                substrate.update_flow(obs_next)
                substrate.record_state_change(action, delta)

        if fresh_episode:
            fresh_episode = False
            obs = obs_next
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > phase2_level:
            phase2_level = cl

        # lmax_solved = substrate caused done=True in L_max phase (completed the level)
        if done and not lmax_solved:
            lmax_solved = True
            lmax_solved_step = steps
            obs = env.reset(seed=TRANSPORT_SEED)
            if hasattr(substrate, 'on_level_transition'):
                substrate.on_level_transition()
            fresh_episode = True
        elif done:
            obs = env.reset(seed=TRANSPORT_SEED)
            if hasattr(substrate, 'on_level_transition'):
                substrate.on_level_transition()
            fresh_episode = True
        else:
            obs = obs_next

    phase2_elapsed = time.time() - t_phase2
    total_elapsed = transport_elapsed + phase2_elapsed

    # ── Stage metrics (L_max phase only) ──
    i3_rho, i3_pass = compute_i3(i3_action_counts, kb_delta) \
        if i3_action_counts is not None else (None, None)
    i1_result = compute_i1(repr_log)
    i4_result = compute_i4(action_log, n_actions)

    r3_result = compute_r3_jacobian(substrate, r3_obs_sample, r3_snapshot, n_actions) \
        if (r3_snapshot and r3_obs_sample) else \
        {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    sal_result = compute_sal(substrate)

    result = {
        'game': game_name.lower(),
        'condition': condition,
        'draw': draw,
        'seed': substrate_seed,
        'transport_ok': True,
        'transport_steps': transport_steps,
        'transport_elapsed': round(transport_elapsed, 2),
        'lmax_level': target_level,
        'lmax_solved': lmax_solved,
        'lmax_solved_step': lmax_solved_step,
        'steps_in_lmax': steps,
        'phase2_elapsed': round(phase2_elapsed, 2),
        'total_elapsed': round(total_elapsed, 2),

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
        'R3_jacobian_diff': r3_result['jacobian_diff'],
        'R3_n_obs_used': r3_result['n_obs_used'],
        'R3_pass': r3_result['pass'],
        'SAL_rho': sal_result['rho'],
        'SAL_pass': sal_result['pass'],
        'SAL_n_actions_visited': sal_result['n_actions_visited'],
    }

    transport_str = f"T={transport_steps}steps/{transport_elapsed:.1f}s"
    lmax_str = f"Lmax={'SOLVED@' + str(lmax_solved_step) if lmax_solved else 'FAIL'}"
    i3_str = f"I3ρ={i3_rho:.2f}" if i3_rho is not None else "I3=null"
    sal_str = f"SAL={sal_result['rho']:.3f}" if sal_result['rho'] is not None else "SAL=null"
    r3_str = f"R3={r3_result['jacobian_diff']:.4f}" if r3_result['jacobian_diff'] else "R3=null"
    i4_str = f"I4={i4_result['reduction_pct']:.1f}%" if i4_result['reduction_pct'] is not None else "I4=null"
    print(f" {transport_str} | {lmax_str} | {i3_str} | {r3_str} | {sal_str} | {i4_str} | {phase2_elapsed:.1f}s")

    return result


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1269 — SOLVER HANDOFF TO L_MAX (SP80)")
    print("=" * 70)
    print(f"Game: {GAME.upper()} ({N_LEVELS} levels, target L_max=L{TARGET_LEVEL})")
    print(f"Transport: deterministic prescription replay (seed={TRANSPORT_SEED})")
    print(f"Conditions: physarum_observe (warm observe), control_c_cold (fresh at L_max)")
    print(f"Draws: {N_DRAWS} per condition = {N_DRAWS * 2} new runs")
    print(f"Reference: control_c_l1 = Step 1268 SP80 (Lmax=0/5, confirmed)")
    print(f"Budget: {MAX_STEPS} substrate steps / {MAX_SECONDS}s per run")
    print()
    print("DESIGN TARGET: Does substrate observation during transport help at L_max?")
    print()

    # Load prescription
    prescription = load_prescription(GAME)
    print(f"Prescription loaded: {len(prescription)} actions for {GAME.upper()}")

    # Load game diagnostics
    diag = load_game_diag(GAME)
    kb_delta, _ = build_kb_profile(diag)
    game_n_actions = diag.get('n_actions', 4103)
    print(f"n_actions={game_n_actions}")
    print()

    all_results = []
    t_global = time.time()

    conditions = ['physarum_observe', 'control_c_cold']

    for condition in conditions:
        print(f"\n{'─'*60}")
        print(f"Condition: {condition}")
        print(f"{'─'*60}")
        for draw in range(1, N_DRAWS + 1):
            substrate_seed = draw * 100 + (1 if condition == 'physarum_observe' else 2)
            result = run_single(
                game_name=GAME,
                condition=condition,
                draw=draw,
                substrate_seed=substrate_seed,
                n_actions=game_n_actions,
                kb_delta=kb_delta,
                prescription=prescription,
                n_levels=N_LEVELS,
            )
            all_results.append(result)
            with open(os.path.join(RESULTS_DIR, f"{GAME}_{condition}_draw{draw}.json"), 'w') as f:
                json.dump(result, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"STEP 1269 COMPLETE — {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*70}\n")

    # ── Summary ──
    stages = ['I3_pass', 'I1_pass', 'I4_pass', 'R3_pass', 'SAL_pass']
    print(f"{'Stage':<12}", end='')
    for c in conditions:
        label = 'PHY_OBS' if c == 'physarum_observe' else 'CC_COLD'
        print(f"  {label:>9}", end='')
    print()
    print("-" * 40)
    for stage in stages:
        print(f"{stage:<12}", end='')
        for c in conditions:
            vals = [r[stage] for r in all_results if r['condition'] == c
                    and r.get(stage) is not None]
            rate = sum(vals) / len(vals) if vals else None
            print(f"  {f'{rate:.2f}' if rate is not None else '  null':>9}", end='')
        print()

    print()
    print("L_max solved (advanced past L5):")
    for c in conditions:
        runs = [r for r in all_results if r['condition'] == c]
        solved = [r for r in runs if r.get('lmax_solved')]
        steps = [r['lmax_solved_step'] for r in solved]
        label = 'PHY_OBS' if c == 'physarum_observe' else 'CC_COLD'
        print(f"  {label}: {len(solved)}/5  steps={steps}")
    print(f"  CC_L1 (Step 1268 ref): 0/5  (never reached L_max)")

    print()
    print("Transport stats (physarum_observe):")
    phy_runs = [r for r in all_results if r['condition'] == 'physarum_observe' and r.get('transport_ok')]
    if phy_runs:
        t_steps = [r['transport_steps'] for r in phy_runs]
        t_times = [r['transport_elapsed'] for r in phy_runs]
        print(f"  Transport steps: {t_steps[0]} (deterministic)")
        print(f"  Transport time: {np.mean(t_times):.1f}s±{np.std(t_times):.1f}s")

    print()
    print("SAL (L_max phase only):")
    for c in conditions:
        runs = [r for r in all_results if r['condition'] == c]
        rhos = [r['SAL_rho'] for r in runs if r.get('SAL_rho') is not None]
        label = 'PHY_OBS' if c == 'physarum_observe' else 'CC_COLD'
        if rhos:
            print(f"  {label}: rho={np.mean(rhos):.3f}±{np.std(rhos):.3f}")
        else:
            print(f"  {label}: null")

    print()
    print("R3 (L_max phase only):")
    for c in conditions:
        runs = [r for r in all_results if r['condition'] == c]
        r3s = [r['R3_jacobian_diff'] for r in runs if r.get('R3_jacobian_diff') is not None]
        label = 'PHY_OBS' if c == 'physarum_observe' else 'CC_COLD'
        if r3s:
            print(f"  {label}: R3={np.mean(r3s):.4f}±{np.std(r3s):.4f}")
        else:
            print(f"  {label}: null")

    print()
    print("I4 (L_max phase only):")
    for c in conditions:
        runs = [r for r in all_results if r['condition'] == c]
        reductions = [r['I4_reduction_pct'] for r in runs if r.get('I4_reduction_pct') is not None]
        label = 'PHY_OBS' if c == 'physarum_observe' else 'CC_COLD'
        if reductions:
            print(f"  {label}: {np.mean(reductions):.1f}%±{np.std(reductions):.1f}%")
        else:
            print(f"  {label}: null")

    out_path = os.path.join(RESULTS_DIR, 'step1269_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'step': 1269,
            'game': GAME,
            'n_levels': N_LEVELS,
            'target_level': TARGET_LEVEL,
            'transport_seed': TRANSPORT_SEED,
            'conditions': conditions,
            'n_draws': N_DRAWS,
            'total_runs': len(all_results),
            'runs': all_results,
        }, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == '__main__':
    main()
