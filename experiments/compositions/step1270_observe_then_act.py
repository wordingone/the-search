"""
Step 1270 — Observe solver demonstration, act on new instance.

Replaces step1269 L_max handoff approach.

Test: can the substrate extract the game model from watching a complete demonstration?

Phase 1 (Observe): Solver plays ALL levels on seed A. Substrate observes passively
                   (updates running_mean, h, W_action from every (obs,action,obs_next) triple).
Phase 2 (Act):     Substrate acts from L1 on seed B (different instance). No solver.

Conditions:
  observe_then_act: observe full solver demo on seed A, act on seed B
  cold_act:         no observation, act on seed B fresh (null baseline)
  solver_baseline:  solver plays seed B (target performance ceiling)

If observe > cold: substrate extracted something transferable from the demonstration.
If observe = cold: observation doesn't help.

Game: SP80 (107-action prescription, 6 levels, working solver).
Seed pairs: draw d → A = 2*d-1, B = 2*d (e.g., d=1: A=1, B=2)
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')

import numpy as np
from substrates.step0674 import _enc_frame

# ─── Config ───
GAME = 'sp80'
N_LEVELS = 6
N_DRAWS = 5
MAX_STEPS = 10_000
MAX_SECONDS = 280

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM

ETA_FLOW = 0.01
DECAY = 0.001

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
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1270')

GAME_PRESCRIPTIONS = {
    'sp80': ('sp80_fullchain.json', 'sequence', 6),
}


def load_prescription(game_name: str):
    fname, field, _ = GAME_PRESCRIPTIONS[game_name]
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


def load_game_diag(game_name: str) -> dict:
    with open(os.path.join(DIAG_DIR, f'{game_name.lower()}_diagnostic.json')) as f:
        return json.load(f)


def build_kb_profile(diag: dict) -> tuple:
    kb = diag.get('kb_responsiveness', {})
    delta = np.zeros(7, np.float32)
    for i, key in enumerate([f'ACTION{j}' for j in range(1, 8)]):
        if key in kb:
            delta[i] = kb[key].get('delta_mean', 0.0)
    return delta, None


def action_entropy(action_seq, n_actions):
    if not action_seq:
        return 0.0
    counts = np.zeros(n_actions, np.float32)
    for a in action_seq:
        if 0 <= a < n_actions:
            counts[a] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def permutation_test_i1(within_dists, between_dists, n_perms=500, seed=0):
    rng = np.random.RandomState(seed)
    within = np.array(within_dists, np.float32)
    between = np.array(between_dists, np.float32)
    if len(within) == 0 or len(between) == 0:
        return 1.0
    observed = float(np.mean(between) - np.mean(within))
    all_dists = np.concatenate([within, between])
    n_w = len(within)
    count = sum(
        1 for _ in range(n_perms)
        if float(np.mean(all_dists[rng.permutation(len(all_dists))[n_w:]]) -
                 np.mean(all_dists[rng.permutation(len(all_dists))[:n_w]])) >= observed
    )
    return count / n_perms


def spearman_rho(x, y):
    if len(x) < 2:
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float32)
    ry = np.argsort(np.argsort(y)).astype(np.float32)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom < 1e-8:
        return None
    return float(np.dot(rx, ry) / denom)


# ─── Substrates ───

class ControlC:
    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.action_counts = np.zeros(n_actions, np.float32)
        self._last_enc = None
        self.step = 0

    def _centered_encode(self, obs):
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_running_mean):
        return _enc_frame(np.asarray(obs_raw, np.float32)) - frozen_running_mean

    def process(self, obs_raw):
        enc = self._centered_encode(np.asarray(obs_raw, np.float32))
        self._last_enc = enc.copy()
        action = int(np.argmin(self.action_counts))
        self.action_counts[action] += 1
        self.step += 1
        return action

    def on_level_transition(self): pass

    def get_state(self):
        return {'running_mean': self.running_mean.copy(), 'n_obs': self.n_obs, 'step': self.step}


class PhysarumSubstrate:
    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        rng_w = np.random.RandomState(seed + 10000)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)
        scale = 1.0 / np.sqrt(float(EXT_DIM))
        W_init = rng_w.randn(n_actions, EXT_DIM).astype(np.float32) * scale
        self.W_action = W_init.copy()
        self.W_action_init = W_init.copy()
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

    def _centered_encode(self, obs):
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_action):
        obs = np.asarray(obs_raw, np.float32)
        enc = _enc_frame(obs) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        return frozen_W_action @ np.concatenate([enc, h_new])

    def observe_step(self, obs_prev_raw, action, obs_next_raw):
        """Passive learning: update running_mean, h, W_action. No action output."""
        obs_prev = np.asarray(obs_prev_raw, np.float32)
        obs_next = np.asarray(obs_next_raw, np.float32)
        enc_prev = self._centered_encode(obs_prev)
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc_prev)
        ext = np.concatenate([enc_prev, self.h])
        enc_next = _enc_frame(obs_next) - self.running_mean
        flow = float(np.linalg.norm(enc_next - enc_prev))
        if 0 <= action < self.n_actions:
            self.W_action[action] += ETA_FLOW * flow * ext
            self.W_action *= (1.0 - DECAY)
        self._prev_enc_flow = enc_prev.copy()
        self._prev_ext = ext.copy()
        self._last_action = action

    def process(self, obs_raw):
        obs = np.asarray(obs_raw, np.float32)
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
                'act_max': float(activation.max()), 'act_mean': float(activation.mean())}
        self._prev_repr = activation.copy()
        self._prev_enc_flow = enc.copy()
        self._prev_ext = ext.copy()
        self._last_action = action
        self.step += 1
        return action

    def update_flow(self, next_obs_raw):
        if self._prev_enc_flow is None or self._last_action is None:
            return
        enc_after = _enc_frame(np.asarray(next_obs_raw, np.float32)) - self.running_mean
        delta = enc_after - self._prev_enc_flow
        flow = float(np.linalg.norm(delta))
        self.W_action[self._last_action] += ETA_FLOW * flow * self._prev_ext
        self.W_action *= (1.0 - DECAY)

    def record_state_change(self, action, delta):
        if 0 <= action < self.n_actions:
            self._delta_sum[action] += delta
            self._delta_count[action] += 1

    def on_level_transition(self):
        self._prev_enc_flow = None
        self._prev_ext = None

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(), 'n_obs': self.n_obs,
            'h': self.h.copy(), 'W_action': self.W_action.copy(), 'step': self.step,
        }


# ─── Instrumentation ───

def compute_i3(action_counts_200, kb_delta):
    n = min(len(action_counts_200), 7)
    if n < 2:
        return None, None
    freq = action_counts_200[:n].astype(np.float32)
    ref = kb_delta[:n].astype(np.float32)
    if ref.max() - ref.min() < 1e-6:
        return None, None
    rho = spearman_rho(freq, ref)
    return (rho, bool(rho > 0.5)) if rho is not None else (None, None)


def compute_i1(repr_log):
    if len(repr_log) < 4:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}
    rng = np.random.RandomState(1)
    n = len(repr_log)
    within_dists, between_dists = [], []
    for _ in range(200):
        i, j = rng.choice(n, 2, replace=False)
        r1, l1 = repr_log[i]; r2, l2 = repr_log[j]
        d = cosine_dist(r1, r2)
        (within_dists if l1 == l2 else between_dists).append(d)
    if len(within_dists) < 2 or len(between_dists) < 2:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}
    p_val = permutation_test_i1(within_dists, between_dists)
    wm, bm = float(np.mean(within_dists)), float(np.mean(between_dists))
    return {'within': round(wm, 4), 'between': round(bm, 4),
            'p_value': round(p_val, 4), 'pass': bool(wm < bm and p_val < 0.05)}


def compute_i4(action_log, n_actions):
    def entropy_at(step):
        if step > len(action_log):
            return None
        window = action_log[max(0, step - I4_WINDOW): step]
        return action_entropy(window, n_actions) if len(window) >= 5 else None
    h100, h5000 = entropy_at(100), entropy_at(5000)
    if h100 is None or h5000 is None or h100 < 1e-6:
        return {'entropy_100': h100, 'entropy_5000': h5000, 'reduction_pct': None, 'pass': False}
    red = float((h100 - h5000) / h100 * 100)
    return {'entropy_100': round(h100, 4), 'entropy_5000': round(h5000, 4),
            'reduction_pct': round(red, 2), 'pass': bool(red > 10.0)}


def compute_r3_jacobian(substrate, obs_sample, frozen_state, n_actions, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    if not obs_sample:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
    _is_phy = isinstance(substrate, PhysarumSubstrate)
    if _is_phy:
        frozen_rm, frozen_h, frozen_W = (frozen_state['running_mean'],
                                          frozen_state['h'], frozen_state['W_action'])
        fresh_rm = np.zeros(ENC_DIM, np.float32)
        fresh_h = np.zeros(H_DIM, np.float32)
        fresh_W = substrate.W_action_init
    else:
        frozen_rm = frozen_state['running_mean']
        fresh_rm = np.zeros(ENC_DIM, np.float32)

    diffs = []
    for obs_raw in obs_sample[:R3_N_OBS]:
        obs_flat = np.asarray(obs_raw, np.float32)
        if _is_phy:
            be = substrate.get_internal_repr_readonly(obs_flat, frozen_rm, frozen_h, frozen_W)
            bf = substrate.get_internal_repr_readonly(obs_flat, fresh_rm, fresh_h, fresh_W)
        else:
            be = substrate.get_internal_repr_readonly(obs_flat, frozen_rm)
            bf = substrate.get_internal_repr_readonly(obs_flat, fresh_rm)
        J_e, J_f = [], []
        for J, base in [(J_e, be), (J_f, bf)]:
            for _ in range(R3_N_DIRS):
                d = rng.randn(*obs_flat.shape).astype(np.float32)
                d /= (np.linalg.norm(d) + 1e-8)
                perturbed = obs_flat + R3_EPSILON * d
                if _is_phy:
                    pert = substrate.get_internal_repr_readonly(
                        perturbed,
                        frozen_rm if J is J_e else fresh_rm,
                        frozen_h if J is J_e else fresh_h,
                        frozen_W if J is J_e else fresh_W)
                else:
                    pert = substrate.get_internal_repr_readonly(
                        perturbed, frozen_rm if J is J_e else fresh_rm)
                J.append((pert - base) / R3_EPSILON)
        diffs.append(float(np.linalg.norm(np.stack(J_e) - np.stack(J_f))))
    if not diffs:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
    jd = float(np.mean(diffs))
    return {'jacobian_diff': round(jd, 6), 'pass': bool(jd > 0.05), 'n_obs_used': len(diffs)}


def compute_sal(substrate):
    if not isinstance(substrate, PhysarumSubstrate):
        return {'rho': None, 'pass': False, 'n_actions_visited': 0}
    if substrate._sal_steps == 0:
        return {'rho': None, 'pass': False, 'n_actions_visited': 0}
    mean_act = substrate._sal_sum / substrate._sal_steps
    visited = substrate._delta_count > 0
    n_vis = int(visited.sum())
    if n_vis < 5:
        return {'rho': None, 'pass': False, 'n_actions_visited': n_vis}
    mean_delta = substrate._delta_sum[visited] / (substrate._delta_count[visited] + 1e-8)
    rho = spearman_rho(mean_act[visited].tolist(), mean_delta.tolist())
    return {'rho': round(float(rho), 4) if rho is not None else None,
            'pass': bool(rho is not None and rho > 0.3), 'n_actions_visited': n_vis}


# ─── Observe phase ───

def run_observe_phase(substrate, prescription, n_actions, seed_a):
    """Replay full prescription on seed_a. Substrate observes passively."""
    env = make_game(GAME)
    obs = env.reset(seed=seed_a)
    max_level_observed = 0
    level = 0
    n_observed = 0

    for action in prescription:
        obs_prev = obs
        action_int = int(action) % n_actions
        obs_next, reward, done, info = env.step(action_int)

        substrate.observe_step(obs_prev, action_int, obs_next)
        n_observed += 1

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > max_level_observed:
            max_level_observed = cl
            substrate.on_level_transition()

        if done:
            obs = env.reset(seed=seed_a)
        else:
            obs = obs_next

    return {'n_observed': n_observed, 'max_level_observed': max_level_observed}


def run_solver_baseline(prescription, n_actions, seed_b):
    """Solver plays seed_b. Returns max level reached."""
    env = make_game(GAME)
    obs = env.reset(seed=seed_b)
    max_level = 0
    level = 0
    level_transitions = {}
    fresh_episode = False

    for action in prescription:
        obs_prev = obs
        action_int = int(action) % n_actions
        obs_next, reward, done, info = env.step(action_int)

        if fresh_episode:
            fresh_episode = False
            obs = obs_next
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level = cl
            if cl > max_level:
                max_level = cl
                level_transitions[cl] = len(level_transitions) + 1

        if done:
            obs = env.reset(seed=seed_b)
            fresh_episode = True
        else:
            obs = obs_next

    return {'max_level': max_level, 'level_transitions': level_transitions}


# ─── Act phase (standard play from L1) ───

def run_act_phase(game_name, substrate, seed_b, n_actions, kb_delta):
    """Substrate acts from L1 on seed_b. Full stage instrumentation."""
    env = make_game(game_name)
    obs = env.reset(seed=seed_b)

    action_log = []
    repr_log = []
    obs_store = []
    level_actions = {}

    i3_action_counts = None
    r3_snapshot = None
    r3_obs_sample = None

    steps = 0
    level = 0
    l1_step = l2_step = None
    t_start = time.time()
    fresh_episode = True

    while steps < MAX_STEPS:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed_b)
            substrate.on_level_transition()
            fresh_episode = True
            continue

        obs_arr = np.asarray(obs, np.float32)
        obs_flat = obs_arr.ravel()

        obs_store.append(obs_arr)
        if len(obs_store) > 200:
            obs_store.pop(0)

        if steps <= I1_STEP and steps % 20 == 0:
            if hasattr(substrate, '_prev_repr') and substrate._prev_repr is not None:
                repr_log.append((substrate._prev_repr.copy(), level))
            elif hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
                repr_log.append((substrate._last_enc.copy(), level))

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
            next_flat = np.asarray(obs_next, np.float32).ravel()
            delta = float(np.linalg.norm(next_flat - obs_flat))
            if isinstance(substrate, PhysarumSubstrate):
                substrate.update_flow(obs_next)
                substrate.record_state_change(action, delta)

        if fresh_episode:
            fresh_episode = False
            obs = obs_next
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1_step is None:
                l1_step = steps
            if cl == 2 and l2_step is None:
                l2_step = steps
            level_actions[cl] = steps
            level = cl

        if done:
            obs = env.reset(seed=seed_b)
            substrate.on_level_transition()
            fresh_episode = True
            level = 0
        else:
            obs = obs_next

    elapsed = time.time() - t_start

    i3_rho, i3_pass = compute_i3(i3_action_counts, kb_delta) \
        if i3_action_counts is not None else (None, None)
    i1_result = compute_i1(repr_log)
    i4_result = compute_i4(action_log, n_actions)
    r3_result = compute_r3_jacobian(substrate, r3_obs_sample, r3_snapshot, n_actions) \
        if (r3_snapshot and r3_obs_sample) else \
        {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
    sal_result = compute_sal(substrate)

    return {
        'steps': steps, 'elapsed': round(elapsed, 2),
        'max_level': level, 'L1_solved': bool(l1_step is not None),
        'L2_solved': bool(l2_step is not None),
        'l1_step': l1_step, 'l2_step': l2_step,
        'level_actions': level_actions,
        'I3_spearman_rho': round(i3_rho, 4) if i3_rho is not None else None,
        'I3_pass': i3_pass,
        'I1_within_dist': i1_result['within'], 'I1_between_dist': i1_result['between'],
        'I1_p_value': i1_result['p_value'], 'I1_pass': i1_result['pass'],
        'I4_entropy_100': i4_result['entropy_100'], 'I4_entropy_5000': i4_result['entropy_5000'],
        'I4_reduction_pct': i4_result['reduction_pct'], 'I4_pass': i4_result['pass'],
        'R3_jacobian_diff': r3_result['jacobian_diff'], 'R3_pass': r3_result['pass'],
        'SAL_rho': sal_result['rho'], 'SAL_pass': sal_result['pass'],
        'SAL_n_actions_visited': sal_result['n_actions_visited'],
    }


# ─── Single run harness ───

def run_single(condition, draw, seed_a, seed_b, n_actions, kb_delta, prescription):
    print(f"  {condition} | draw={draw} | seedA={seed_a} seedB={seed_b} ...",
          end='', flush=True)
    t_start = time.time()

    if condition == 'observe_then_act':
        substrate = PhysarumSubstrate(n_actions=n_actions, seed=seed_b)
        obs_result = run_observe_phase(substrate, prescription, n_actions, seed_a)
        act_result = run_act_phase(GAME, substrate, seed_b, n_actions, kb_delta)

        result = {
            'condition': condition, 'draw': draw, 'seed_a': seed_a, 'seed_b': seed_b,
            'observe': obs_result,
            **{f'act_{k}': v for k, v in act_result.items()},
            'total_elapsed': round(time.time() - t_start, 2),
        }

    elif condition == 'cold_act':
        substrate = PhysarumSubstrate(n_actions=n_actions, seed=seed_b)
        act_result = run_act_phase(GAME, substrate, seed_b, n_actions, kb_delta)

        result = {
            'condition': condition, 'draw': draw, 'seed_a': None, 'seed_b': seed_b,
            'observe': None,
            **{f'act_{k}': v for k, v in act_result.items()},
            'total_elapsed': round(time.time() - t_start, 2),
        }

    elif condition == 'solver_baseline':
        sol_result = run_solver_baseline(prescription, n_actions, seed_b)
        result = {
            'condition': condition, 'draw': draw, 'seed_a': None, 'seed_b': seed_b,
            'observe': None,
            'act_max_level': sol_result['max_level'],
            'act_L1_solved': bool(sol_result['max_level'] >= 1),
            'act_L2_solved': bool(sol_result['max_level'] >= 2),
            'solver_level_transitions': sol_result['level_transitions'],
            'total_elapsed': round(time.time() - t_start, 2),
        }
    else:
        raise ValueError(f"Unknown condition: {condition}")

    max_level = result.get('act_max_level', 0)
    l1 = result.get('act_L1_solved', False)
    l2 = result.get('act_L2_solved', False)
    r3 = result.get('act_R3_jacobian_diff')
    sal = result.get('act_SAL_rho')
    elapsed = result['total_elapsed']
    print(f" L{max_level} L1={'Y' if l1 else 'N'} L2={'Y' if l2 else 'N'}"
          f" | R3={r3:.4f}" if r3 else f" L{max_level} L1={'Y' if l1 else 'N'} L2={'Y' if l2 else 'N'}"
          f" | {elapsed:.1f}s")

    return result


# ─── Main ───

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1270 — OBSERVE SOLVER DEMO, ACT ON NEW INSTANCE (SP80)")
    print("=" * 70)
    print(f"Substrate: Physarum+argmin (Step 1266 architecture)")
    print(f"Conditions: observe_then_act, cold_act, solver_baseline")
    print(f"Draws: {N_DRAWS} | Seed pairs: A=2d-1, B=2d")
    print(f"Question: does watching solver on seed A transfer to acting on seed B?")
    print()

    prescription = load_prescription(GAME)
    print(f"Prescription: {len(prescription)} actions for {GAME.upper()}")

    diag = load_game_diag(GAME)
    kb_delta, _ = build_kb_profile(diag)
    game_n_actions = diag.get('n_actions', 4103)
    print(f"n_actions={game_n_actions}")
    print()

    all_results = []
    t_global = time.time()

    conditions = ['observe_then_act', 'cold_act', 'solver_baseline']

    for condition in conditions:
        print(f"\n{'─'*60}")
        print(f"Condition: {condition}")
        print(f"{'─'*60}")
        for draw in range(1, N_DRAWS + 1):
            seed_a = 2 * draw - 1
            seed_b = 2 * draw
            result = run_single(
                condition=condition, draw=draw,
                seed_a=seed_a, seed_b=seed_b,
                n_actions=game_n_actions, kb_delta=kb_delta,
                prescription=prescription,
            )
            all_results.append(result)
            fname = f"{GAME}_{condition}_draw{draw}.json"
            with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
                json.dump(result, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"STEP 1270 COMPLETE — {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*70}\n")

    # ── Summary ──
    print(f"{'Condition':<22}  {'L1':<6}  {'L2':<6}  {'Lmax':<6}  R3        SAL")
    print("-" * 65)
    for cond in conditions:
        runs = [r for r in all_results if r['condition'] == cond]
        l1_rate = sum(1 for r in runs if r.get('act_L1_solved')) / len(runs) if runs else 0
        l2_rate = sum(1 for r in runs if r.get('act_L2_solved')) / len(runs) if runs else 0
        levels = [r.get('act_max_level', 0) for r in runs]
        r3s = [r['act_R3_jacobian_diff'] for r in runs
               if r.get('act_R3_jacobian_diff') is not None]
        sals = [r['act_SAL_rho'] for r in runs if r.get('act_SAL_rho') is not None]
        r3_str = f"{np.mean(r3s):.4f}" if r3s else "null"
        sal_str = f"{np.mean(sals):.3f}" if sals else "null"
        lmax_str = f"{np.mean(levels):.1f}" if levels else "null"
        print(f"  {cond:<20}  {l1_rate:.2f}   {l2_rate:.2f}   {lmax_str:<6}  {r3_str:<10}{sal_str}")

    print()
    print("Per-draw L1 solved:")
    for cond in ['observe_then_act', 'cold_act']:
        runs = [r for r in all_results if r['condition'] == cond]
        l1_draws = [r.get('act_L1_solved', False) for r in runs]
        print(f"  {cond}: {l1_draws} ({sum(l1_draws)}/5)")

    out_path = os.path.join(RESULTS_DIR, 'step1270_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'step': 1270, 'game': GAME, 'conditions': conditions,
            'n_draws': N_DRAWS, 'total_runs': len(all_results), 'runs': all_results,
        }, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == '__main__':
    main()
