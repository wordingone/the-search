"""
Step 1299 — Action-Aware Forward Model (C34, NEW).

Problem: 1297 showed W_drive learns (Oja fires) but action selection concentrates
on VISUALLY RESPONSIVE actions (large obs PE), not game-advancing ones. The obs PE
is action-BLIND — can't distinguish which actions matter.

Fix: W_forward[a] — per-action forward model. Predicts enc_{t+1} GIVEN action a.
  forward_pe[a] = ||enc_{t+1} - W_forward[a] @ enc_t||
For loop actions: forward_pe drops → plasticity gate CLOSES → substrate stops learning
For advancing actions: forward_pe stays high → plasticity gate OPEN → keeps learning

Architecture (on 1297 base: N=64, NEG-DIAG W_recur, W_drive init 0.1):
  NEW: W_forward dict (lazy init, cap 500; shared fallback for large action spaces)
  W_drive Oja, W_recur three-factor, W_readout — ALL gated by forward PE

Conditions (3):
  FORWARD-GATE: plasticity gated by normalized forward PE
  OBS-PE-GATE:  plasticity gated by normalized obs PE (action-blind, comparison)
  NO-GATE:      forward model exists and learns, but plasticity_gate=1.0 (ablation)

Protocol: Full PRISM (10 ARC + 1 MBPP). 10 draws × 2 episodes (A+B, same substrate).
Budget: 3 × 11 × 10 = 330 substrate instances, ~50 min.

New measurements:
  - Forward PE trajectory per action (frequency vs PE trajectory)
  - Phase boundary detection: forward PE at exact L1 step
  - Attractor reorganization: cluster activation distance before vs after L1
  - Second-exposure speedup (beat PE-EMA baseline 1.16x)
  - RHAE (beat PE-EMA baseline 8.86e-7)

Kill criteria:
  - Forward PE doesn't decrease for ANY action after 5K steps → model can't learn
  - Behavioral R3 same direction as 1297 (visually responsive concentration) → gating fails
  - L1 + speedup both ≤ PE-EMA across all games → forward model adds nothing

Component catalog: C34 (NEW). R2 compliant (W_forward gates plasticity only, not action selection).
Reference baseline: Step 1298 (20 draws). PE-EMA: {ft09:18, vc33:20, lp85:20}/20, RHAE=8.86e-7, speedup=1.16x.

Spec: Leo mail 3661, 2026-03-28.
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

N_DRAWS = 10
MAX_STEPS = 10_000
MAX_SECONDS = 300

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM  # 320

NETWORK_SIZE = 64
DT_SUB = 0.05
N_SUBSTEPS = 10
OJA_LR = 0.01
ETA_PRED = 0.01
FWD_LR = 0.01
W_RECUR_LR = 0.001
W_RO_LR = 0.001
PE_WINDOW = 500
NEG_DIAG_VAL = -0.1
FWD_CAP = 500  # max per-action forward models before shared fallback

SNAP_FREQ = 100
R3_STEP = 5000
R3_N_OBS = 50
R3_N_DIRS = 20
R3_EPSILON = 0.01

CONDITIONS = ['forward_gate', 'obs_pe_gate', 'no_gate']
LABELS = {'forward_gate': 'FWD', 'obs_pe_gate': 'OBS', 'no_gate': 'NOGATE'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1299')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'
DIAG_DIR = os.path.join('B:/M/the-search/experiments', 'results', 'game_diagnostics')

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


# --- Utilities (shared from 1297/1298) ---

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
        from mbpp_game import compute_solver_steps
        idx = int(game_name.split('_', 1)[1]) if '_' in game_name else 0
        return compute_solver_steps(idx)
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
            fresh_episode = False; obs = obs_next; continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level_first_step[cl] = step; level = cl
        if done:
            obs = env.reset(seed=seed); fresh_episode = True
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
    pre_counts = np.zeros(n_actions, np.float32)
    post_counts = np.zeros(n_actions, np.float32)
    for a in pre:
        if 0 <= a < n_actions:
            pre_counts[a] += 1
    for a in post:
        if 0 <= a < n_actions:
            post_counts[a] += 1
    pre_total = pre_counts.sum()
    post_total = post_counts.sum()
    if pre_total == 0 or post_total == 0:
        return None
    pre_p = (pre_counts + 1e-8) / (pre_total + 1e-8 * n_actions)
    post_p = (post_counts + 1e-8) / (post_total + 1e-8 * n_actions)
    return round(float(np.sum(post_p * np.log(post_p / pre_p))), 4)


def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# --- Substrate ---

class ForwardModelSubstrate:
    """1297 base + per-action forward model W_forward[a] gating all plasticity.

    W_forward[a]: ENC_DIM × ENC_DIM matrix. Predicts enc_{t+1} given action a.
    forward_pe = ||enc_{t+1} - W_forward[a] @ enc_t||
    plasticity_gate = min(forward_pe / forward_pe_median, 2.0)

    Lazy init: W_forward allocated on first visit (eye * 0.01).
    Cap: FWD_CAP (500) unique action models. Beyond cap: W_forward_shared.

    Modes:
      forward_gate: ALL plasticity gated by normalized forward PE
      obs_pe_gate:  ALL plasticity gated by normalized obs PE (action-blind comparison)
      no_gate:      forward model exists and learns, but gate=1.0 always
    """

    def __init__(self, n_actions, seed, mode='forward_gate'):
        self.n_actions = n_actions
        self.mode = mode
        rng_w = np.random.RandomState(seed + 10000)

        # Encoding pipeline
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0

        # Recurrent state
        self.W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)

        # Obs prediction model (for OBS-PE-GATE)
        self.W_pred = rng_w.randn(ENC_DIM, ENC_DIM).astype(np.float32) * 0.01

        # Competitive network (1297 base: W_drive init 0.1)
        self.W_drive = rng_w.randn(NETWORK_SIZE, EXT_DIM).astype(np.float32) * 0.1
        self.W_drive_init = self.W_drive.copy()
        self.W_inhibit = rng_w.uniform(0.5, 1.5, (NETWORK_SIZE, NETWORK_SIZE)).astype(np.float32)
        np.fill_diagonal(self.W_inhibit, 0.0)

        # W_readout: (n_actions × NETWORK_SIZE)
        self.W_readout = rng_w.randn(n_actions, NETWORK_SIZE).astype(np.float32) * 0.01
        self.W_readout_init = self.W_readout.copy()

        self.activation = np.zeros(NETWORK_SIZE, np.float32)
        self.prev_activation = np.zeros(NETWORK_SIZE, np.float32)

        # W_recur: NEG-DIAG base
        self.W_recur = rng_w.randn(NETWORK_SIZE, NETWORK_SIZE).astype(np.float32) * 0.01
        np.fill_diagonal(self.W_recur, NEG_DIAG_VAL)

        # NEW: Per-action forward model (C34)
        self.W_forward = {}                     # action -> ENC_DIM×ENC_DIM
        self.W_forward_shared = np.eye(ENC_DIM, dtype=np.float32) * 0.01  # fallback
        self._fwd_pe_window = []
        self._fwd_pe_median = 1.0
        # Per-action trajectory: action -> [(step, pe), ...]  (sampled, max 20 per action)
        self._action_fwd_pe = {}
        self._fwd_pe_at_l1 = None  # forward PE at the step L1 is first reached

        # Obs PE tracking (for OBS-PE-GATE)
        self._obs_pe_window = []
        self._obs_pe_median = 1.0

        # State tracking
        self._prev_enc = None
        self._last_action = None
        self.step = 0
        self.winner_seq = []
        self.activation_snaps = []
        self._snap_levels = []
        self._snap_steps = []
        self.oja_fire_count = 0
        self._current_level = 0

    def _centered_encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_drive):
        """R3: encode obs with experienced W_drive vs fresh W_drive."""
        enc = _enc_frame(np.asarray(obs_raw, dtype=np.float32)) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        ext = np.concatenate([enc, h_new])
        return frozen_W_drive @ ext

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'h': self.h.copy(),
            'W_drive': self.W_drive.copy(),
            'step': self.step,
        }

    def process(self, obs_raw, current_level=0):
        enc = self._centered_encode(obs_raw)
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = np.concatenate([enc, self.h])

        # Obs PE (action-blind, used for OBS-PE-GATE and comparison)
        obs_pe = 0.0
        if self._prev_enc is not None:
            pred = self.W_pred @ self._prev_enc
            obs_pe = float(np.linalg.norm(enc - pred))
            self.W_pred += ETA_PRED * np.outer(enc - pred, self._prev_enc)
            self._obs_pe_window.append(obs_pe)
            if len(self._obs_pe_window) > PE_WINDOW:
                self._obs_pe_window.pop(0)
            self._obs_pe_median = float(np.median(self._obs_pe_window)) if self._obs_pe_window else 1.0

        # Forward PE (action-aware, uses enc from PREVIOUS step)
        fwd_pe = 1.0
        if self._prev_enc is not None and self._last_action is not None:
            a = self._last_action
            enc_prev = self._prev_enc

            if a in self.W_forward:
                enc_pred_fwd = self.W_forward[a] @ enc_prev
            elif len(self.W_forward) < FWD_CAP:
                self.W_forward[a] = np.eye(ENC_DIM, dtype=np.float32) * 0.01
                enc_pred_fwd = self.W_forward[a] @ enc_prev
            else:
                enc_pred_fwd = self.W_forward_shared @ enc_prev

            fwd_pe = float(np.linalg.norm(enc - enc_pred_fwd))

            # Track forward PE trajectory per action (sampled: every 10 visits max)
            if a not in self._action_fwd_pe:
                self._action_fwd_pe[a] = []
            if len(self._action_fwd_pe[a]) < 20:
                self._action_fwd_pe[a].append((self.step, round(fwd_pe, 4)))

            # Update forward PE window + median
            self._fwd_pe_window.append(fwd_pe)
            if len(self._fwd_pe_window) > PE_WINDOW:
                self._fwd_pe_window.pop(0)
            self._fwd_pe_median = float(np.median(self._fwd_pe_window)) if self._fwd_pe_window else 1.0

            # Update forward model
            pred_err = enc - enc_pred_fwd
            if a in self.W_forward:
                self.W_forward[a] += FWD_LR * np.outer(pred_err, enc_prev)
            else:
                self.W_forward_shared += FWD_LR * np.outer(pred_err, enc_prev)

        # Phase boundary detection: record fwd_pe at L1 transition
        if current_level > self._current_level:
            if self._fwd_pe_at_l1 is None:
                self._fwd_pe_at_l1 = round(fwd_pe, 4)
            self._current_level = current_level

        # Compute plasticity gate
        if self.mode == 'forward_gate':
            gate = min(fwd_pe / self._fwd_pe_median, 2.0) if self._fwd_pe_median > 1e-8 else 1.0
        elif self.mode == 'obs_pe_gate':
            gate = min(obs_pe / self._obs_pe_median, 2.0) if self._obs_pe_median > 1e-8 else 1.0
        else:  # no_gate
            gate = 1.0

        # Network dynamics (same as 1297: NEG-DIAG W_recur)
        prev_act = self.prev_activation
        act = self.activation.copy()
        for _ in range(N_SUBSTEPS):
            drive = self.W_drive @ ext
            recurrent = self.W_recur @ prev_act
            inhibit = self.W_inhibit @ act
            d_act = -act + np.maximum(drive + recurrent - inhibit, 0.0)
            act = np.maximum(act + DT_SUB * d_act, 0.0)
        self.activation = act
        winner = int(np.argmax(self.activation))

        # Action selection from W_readout
        action_scores = self.W_readout @ self.activation
        action = int(np.argmax(action_scores))

        # Oja plasticity on W_drive (gated by win_act AND plasticity gate)
        win_act = float(self.activation[winner])
        if win_act > 1e-8:
            self.W_drive[winner] += OJA_LR * win_act * gate * (ext - self.W_drive[winner])
            self.oja_fire_count += 1

        # W_recur: three-factor (gated)
        if win_act > 1e-8:
            dW = W_RECUR_LR * gate * np.outer(self.activation, prev_act)
            self.W_recur += dW
            norms = np.linalg.norm(self.W_recur, axis=1, keepdims=True)
            self.W_recur /= np.maximum(norms, 1.0)
        np.fill_diagonal(self.W_recur, NEG_DIAG_VAL)

        # W_readout (gated)
        act_norm_val = float(np.linalg.norm(self.activation))
        if act_norm_val > 1e-8:
            act_norm_vec = self.activation / act_norm_val
            self.W_readout[action] += W_RO_LR * gate * act_norm_vec
            norms = np.linalg.norm(self.W_readout, axis=1, keepdims=True)
            self.W_readout /= np.maximum(norms, 1.0)

        # Tracking
        self.prev_activation = self.activation.copy()
        self.winner_seq.append(winner)
        if self.step % SNAP_FREQ == 0:
            self.activation_snaps.append(self.activation.copy())
            self._snap_levels.append(current_level)
            self._snap_steps.append(self.step)

        self._prev_enc = enc.copy()
        self._last_action = action
        self.step += 1
        return action

    def on_level_transition(self):
        pass

    def get_fwd_pe_trajectory(self):
        """Sample forward PE trajectory: top-5 most-visited actions, first vs last PE."""
        if not self._action_fwd_pe:
            return {}
        result = {}
        for a, traj in list(self._action_fwd_pe.items())[:20]:
            if traj:
                result[str(a)] = {
                    'n_entries': len(traj),
                    'first_step': traj[0][0], 'first_pe': traj[0][1],
                    'last_step': traj[-1][0], 'last_pe': traj[-1][1],
                    'decreasing': traj[-1][1] < traj[0][1] if len(traj) > 1 else None,
                }
        return result

    def fwd_pe_decreasing_any(self):
        """True if ANY action's forward PE decreased from first to last measurement."""
        for a, traj in self._action_fwd_pe.items():
            if len(traj) >= 5 and traj[-1][1] < traj[0][1] * 0.9:  # >10% decrease
                return True
        return False


# --- R3 computation ---

def compute_r3(substrate, obs_sample, snapshot):
    if not obs_sample or snapshot is None:
        return None, False
    frozen_rm = snapshot['running_mean']
    frozen_h = snapshot['h']
    frozen_W = snapshot['W_drive']
    fresh_W = substrate.W_drive_init.copy()
    fresh_rm = np.zeros(ENC_DIM, np.float32)
    fresh_h = np.zeros(H_DIM, np.float32)
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
            diffs.append(float(np.linalg.norm((pe - base_exp) - (pf - base_fresh))))
    if not diffs:
        return None, False
    mean_diff = float(np.mean(diffs))
    return round(mean_diff, 4), bool(mean_diff > 0.05)


# --- Attractor reorganization around L1 ---

def compute_attractor_reorganization(snaps, snap_levels, l1_level=1, window=20):
    """Cluster distance: mean cosine dist between activations 100 steps before vs after L1."""
    if l1_level not in snap_levels or len(snaps) < 4:
        return None
    l1_idx = [i for i, lv in enumerate(snap_levels) if lv >= l1_level]
    if not l1_idx:
        return None
    pivot = l1_idx[0]
    pre = snaps[max(0, pivot - window):pivot]
    post = snaps[pivot:min(len(snaps), pivot + window)]
    if len(pre) < 2 or len(post) < 2:
        return None
    pre_mean = np.mean(pre, axis=0)
    post_mean = np.mean(post, axis=0)
    return round(cosine_dist(pre_mean, post_mean), 4)


# --- Run episode ---

def run_episode(game_name, substrate, draw, seed, n_actions, solver_level_steps,
                take_r3_snapshot=False, is_mbpp=False):
    env = make_game(game_name)
    obs = env.reset(seed=seed)

    action_log = []
    obs_store = []
    r3_snapshot = None
    r3_obs_sample = None
    i3_counts_at_200 = None
    action_counts = np.zeros(n_actions, np.float32)

    mbpp_chars_100 = None
    mbpp_chars_1000 = None

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

        if is_mbpp:
            if steps == 100:
                mbpp_chars_100 = action_counts.copy()
            if steps == 1000:
                mbpp_chars_1000 = action_counts.copy()

        action = substrate.process(obs_arr, current_level=level) % n_actions
        action_counts[action] += 1
        action_log.append(action)

        obs_next, reward, done, info = env.step(action)
        steps += 1

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

    # I3 cv
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
    post_trans_kl = compute_post_transition_kl(action_log, l1_step, n_actions)

    # MBPP char stats
    def kl_uniform(counts_arr, n):
        total = counts_arr.sum()
        if total == 0:
            return None
        probs = counts_arr / total
        uniform = 1.0 / n
        return round(float(sum(p * np.log(p / uniform) for p in probs if p > 0)), 4)

    mbpp_kl_100 = kl_uniform(mbpp_chars_100, n_actions) if mbpp_chars_100 is not None else None
    mbpp_kl_1000 = kl_uniform(mbpp_chars_1000, n_actions) if mbpp_chars_1000 is not None else None

    # R3
    r3_val, r3_pass = None, False
    if take_r3_snapshot:
        r3_val, r3_pass = compute_r3(substrate, r3_obs_sample, r3_snapshot)

    # Attractor reorganization
    reorg = compute_attractor_reorganization(
        substrate.activation_snaps, substrate._snap_levels, l1_level=1)

    return {
        'game': game_name, 'condition': substrate.mode, 'draw': draw, 'seed': seed,
        'steps_taken': steps, 'runtime_seconds': round(elapsed, 2),
        'L1_solved': l1_step is not None,
        'L2_solved': l2_step is not None,
        'max_level': max_level,
        'l1_step': l1_step,
        'l2_step': l2_step,
        'arc_score': round(arc_score, 6),
        'RHAE': round(rhae, 8),
        'I3_cv': i3_cv,
        'post_transition_kl': post_trans_kl,
        'mbpp_kl_at_100': mbpp_kl_100,
        'mbpp_kl_at_1000': mbpp_kl_1000,
        'R3_jacobian_diff': r3_val,
        'R3_pass': r3_pass,
        'attractor_reorg_dist': reorg,
        'fwd_pe_at_l1': substrate._fwd_pe_at_l1,
        'fwd_pe_decreasing_any': substrate.fwd_pe_decreasing_any(),
        'fwd_pe_median_final': round(substrate._fwd_pe_median, 4),
        'oja_fire_count': substrate.oja_fire_count,
        'n_forward_models': len(substrate.W_forward),
        'r3_snapshot_taken': r3_snapshot is not None,
    }


# --- Run draw (2 episodes: A+B on same substrate) ---

def run_draw(game_name, condition, draw, seed_a, seed_b, n_actions, solver_level_steps):
    substrate = ForwardModelSubstrate(n_actions=n_actions, seed=draw * 7 + hash(condition) % 997, mode=condition)
    is_mbpp = game_name.lower().startswith('mbpp')

    t0 = time.time()
    print(f"  {game_name.upper()} | {LABELS[condition]} | draw={draw} | seedA={seed_a} seedB={seed_b} ...", end='', flush=True)

    ep_a = run_episode(game_name, substrate, draw, seed_a, n_actions, solver_level_steps,
                       take_r3_snapshot=True, is_mbpp=is_mbpp)
    ep_b = run_episode(game_name, substrate, draw, seed_b, n_actions, solver_level_steps,
                       take_r3_snapshot=False, is_mbpp=is_mbpp)

    # Second-exposure speedup
    if ep_a['l1_step'] is not None and ep_b['l1_step'] is not None:
        speedup = round(ep_a['l1_step'] / ep_b['l1_step'], 3) if ep_b['l1_step'] > 0 else None
    else:
        speedup = None

    l1_both = ep_a['L1_solved'] and ep_b['L1_solved']

    elapsed = time.time() - t0
    print(f" [A:Lmax={ep_a['max_level']} B:Lmax={ep_b['max_level']}] | "
          f"I3cv={ep_a['I3_cv'] or '?':.3f} | fwdPE_med={ep_a['fwd_pe_median_final']:.3f} | "
          f"fwdPE@L1={ep_a['fwd_pe_at_l1'] or '?'} | "
          f"speedup={'?' if speedup is None else f'{speedup:.2f}x'} | {elapsed:.1f}s")

    # Forward PE trajectory summary (first 5 actions seen)
    fwd_traj = substrate.get_fwd_pe_trajectory()
    decreasing_actions = sum(1 for v in fwd_traj.values() if v.get('decreasing'))

    return {
        # Episode A
        'A_steps': ep_a['steps_taken'], 'A_max_level': ep_a['max_level'],
        'A_L1_solved': ep_a['L1_solved'], 'A_l1_step': ep_a['l1_step'],
        'A_L2_solved': ep_a['L2_solved'],
        'A_arc_score': ep_a['arc_score'], 'A_RHAE': ep_a['RHAE'],
        'A_I3_cv': ep_a['I3_cv'], 'A_post_transition_kl': ep_a['post_transition_kl'],
        'A_mbpp_kl_100': ep_a['mbpp_kl_at_100'], 'A_mbpp_kl_1000': ep_a['mbpp_kl_at_1000'],
        # Episode B
        'B_steps': ep_b['steps_taken'], 'B_max_level': ep_b['max_level'],
        'B_L1_solved': ep_b['L1_solved'], 'B_l1_step': ep_b['l1_step'],
        'B_arc_score': ep_b['arc_score'], 'B_RHAE': ep_b['RHAE'],
        # Cross-episode
        'speedup_ratio': speedup, 'L1_both_episodes': l1_both,
        # R3 (episode A)
        'R3_jacobian_diff': ep_a['R3_jacobian_diff'], 'R3_pass': ep_a['R3_pass'],
        # Forward model diagnostics
        'fwd_pe_at_l1': ep_a['fwd_pe_at_l1'],
        'fwd_pe_median_final': ep_a['fwd_pe_median_final'],
        'fwd_pe_decreasing_any': ep_a['fwd_pe_decreasing_any'],
        'n_forward_models': ep_a['n_forward_models'],
        'n_fwd_pe_decreasing': decreasing_actions,
        'attractor_reorg_dist': ep_a['attractor_reorg_dist'],
        'oja_fire_count': ep_a['oja_fire_count'],
        # Compatibility
        'game': game_name, 'condition': condition, 'draw': draw,
        'seed_a': seed_a, 'seed_b': seed_b,
        'L1_solved': ep_a['L1_solved'],
        'L2_solved': ep_a['L2_solved'],
        'max_level': ep_a['max_level'],
        'arc_score': ep_a['arc_score'],
        'RHAE': ep_a['RHAE'],
        'I3_cv': ep_a['I3_cv'],
    }


# --- Main ---

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("STEP 1299 — Action-Aware Forward Model (C34)")
    print("FORWARD-GATE: plasticity gated by forward PE (action-aware)")
    print("OBS-PE-GATE:  plasticity gated by obs PE (action-blind, comparison)")
    print("NO-GATE:      forward model learns but gate=1.0 (ablation)")
    print(f"Protocol: {N_DRAWS} draws × 2 episodes (A+B, same substrate)")
    print("=" * 72)
    print(f"Games: {GAMES}")
    print(f"N_DRAWS={N_DRAWS} | MAX_STEPS={MAX_STEPS}/episode | MAX_SECONDS={MAX_SECONDS}/episode")
    print()

    print("Computing solver per-level step counts...")
    solver_steps = {}
    for g in GAMES:
        try:
            solver_steps[g] = compute_solver_level_steps(g, seed=1)
            print(f"  {g.upper()}: {solver_steps[g]}")
        except Exception as e:
            print(f"  {g.upper()}: ERROR -- {e}")
            solver_steps[g] = {}
    print()

    all_results = []
    t_global = time.time()

    for game_name in GAMES:
        is_mbpp = game_name.lower().startswith('mbpp')
        if is_mbpp:
            n_actions = 128
        else:
            try:
                with open(os.path.join(DIAG_DIR, f'{game_name.lower()}_diagnostic.json')) as f:
                    diag = json.load(f)
                n_actions = diag.get('n_actions', 4103)
            except Exception:
                n_actions = 4103

        slv = solver_steps.get(game_name, {})
        print(f"\n{'--'*36}")
        print(f"GAME: {game_name.upper()} | n_actions={n_actions} | draws={N_DRAWS}")
        print(f"{'--'*36}")

        for condition in CONDITIONS:
            print(f"\n  Condition: {condition} [{LABELS[condition]}]")
            for draw in range(1, N_DRAWS + 1):
                seed_a = draw * 100
                seed_b = draw * 100 + 50
                result = run_draw(
                    game_name=game_name, condition=condition, draw=draw,
                    seed_a=seed_a, seed_b=seed_b,
                    n_actions=n_actions, solver_level_steps=slv)
                all_results.append(result)
                fname = os.path.join(RESULTS_DIR,
                                     f"{game_name}_{condition}_draw{draw:02d}.json")
                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*72}")
    print(f"STEP 1299 COMPLETE -- {len(all_results)} draws in {total_elapsed:.1f}s")
    print(f"{'='*72}\n")

    # Summary table
    print(f"{'GAME':5} | {'COND':7} | {'L1_A':6} | {'L1_B':6} | {'SPEEDUP':8} | "
          f"{'I3cv':6} | {'RHAE':10} | {'R3':7} | {'fwdPE_dec':9} | {'@L1_PE':7}")
    print("-" * 90)
    for game in GAMES:
        for cond in CONDITIONS:
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            if not runs:
                continue
            n = len(runs)
            l1_a = sum(1 for r in runs if r.get('A_L1_solved'))
            l1_b = sum(1 for r in runs if r.get('B_L1_solved'))
            speedups = [r['speedup_ratio'] for r in runs if r.get('speedup_ratio') is not None]
            i3cv_vals = [r.get('I3_cv') or 0 for r in runs]
            rhae_vals = [r.get('A_RHAE') or 0 for r in runs]
            r3_vals = [r.get('R3_jacobian_diff') or 0 for r in runs]
            fwd_dec = sum(1 for r in runs if r.get('fwd_pe_decreasing_any'))
            l1_pes = [r['fwd_pe_at_l1'] for r in runs if r.get('fwd_pe_at_l1') is not None]
            mean_sp = f"{sum(speedups)/len(speedups):.2f}x" if speedups else "N/A"
            mean_r3 = f"{sum(r3_vals)/n:.4f}"
            mean_l1pe = f"{sum(l1_pes)/len(l1_pes):.3f}" if l1_pes else "N/A"
            print(f"{game:5} | {LABELS[cond]:7} | {l1_a:3}/{n} | {l1_b:3}/{n} | "
                  f"{mean_sp:8} | {sum(i3cv_vals)/n:6.3f} | "
                  f"{sum(rhae_vals)/n:.2e} | {mean_r3} | "
                  f"{fwd_dec:3}/{n}      | {mean_l1pe}")

    # Forward PE learning check
    print("\nFORWARD PE LEARNING (does any action show PE decrease after 5K steps?):")
    for cond in CONDITIONS:
        runs = [r for r in all_results if r['condition'] == cond]
        dec_count = sum(1 for r in runs if r.get('fwd_pe_decreasing_any'))
        print(f"  [{LABELS[cond]}] {dec_count}/{len(runs)} draws show forward PE decrease for ≥1 action")
        kill_check = dec_count == 0
        if kill_check:
            print(f"  *** KILL CRITERION: forward model can't learn ({LABELS[cond]}) ***")

    # Second-exposure speedup
    print("\nSecond-Exposure Speedup (BOTH A and B reached L1):")
    for cond in CONDITIONS:
        runs = [r for r in all_results if r['condition'] == cond]
        speedups = [r['speedup_ratio'] for r in runs if r.get('speedup_ratio') is not None]
        n_pairs = len(speedups)
        if speedups:
            mean_sp = sum(speedups) / n_pairs
            print(f"  [{LABELS[cond]}] n={n_pairs} | mean={mean_sp:.3f}x")
            if mean_sp > 1.16:
                print(f"  *** BEATS PE-EMA BASELINE (1.16x) ***")
        else:
            print(f"  [{LABELS[cond]}] 0 pairs with both A+B reaching L1")

    print("\nSTEP 1299 DONE")


if __name__ == '__main__':
    main()
