"""
Step 1380 — Action-gated SSM: multiplicative action gating forces action-conditional state.
Leo mail 3941, 2026-03-30.

1379 confirmed ACTION_BLIND: SSM ignores action token completely (ratio=1.00002).
Root cause: additive action in SSM input is dominated by obs→obs autocorrelation.
RTRL assigns negligible gradient to action weights → h encodes only observations.

Fix: multiplicative action gating.
  Standard: h_new = A * h_prev + B @ concat(obs, act_emb)     (additive, action-blind, 1379 arch)
  Gated:    h_inter = A * h_prev + B @ obs
            gate    = sigmoid(W_gate[:, prev_act] + b_gate)    (N_STATE,)
            h_new   = h_inter * gate

Gating is unavoidable: action scales EVERY state dimension. Wrong action → wrong gate →
wrong state → wrong prediction. RTRL MUST train W_gate to be action-conditional.

Conditions (paired, 30 draws):
  GATED:    Action-gated SSM. Frozen projection try2 (softmax(W_fixed @ h / T=3)).
  STANDARD: Standard SSM (1379 architecture). Frozen projection try2.
Both: COUNT try1, h reset between tries, weights carry, PRNG fix.

Action-blind diagnostic (10 draws, same seeds as first 10 GATED draws):
  GATED_MASKED: Same as GATED but gate input = zero vector (sigmoid(b_gate) only — no action).
  Kill criterion 1: GATED/GATED_MASKED pred_loss ratio < 1.05 → gating fails to force action conditioning.
  Kill criterion 2: ratio > 1.05 but GATED RHAE <= STANDARD RHAE → action-conditional h doesn't help.

Seeds: 13990-14019 (fresh).
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

STEP        = 1380
N_DRAWS     = 30
DRAW_SEEDS  = [13990 + i for i in range(N_DRAWS)]
N_DIAG      = 10                          # draws for GATED_MASKED action-blind diagnostic
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

RESULTS_DIR     = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')
MLP_TP_BASELINE = 4.59e-5

# SSM config
PROJ_DIM      = 64
ACT_EMBED_DIM = 16     # for STANDARD only
D             = 128
N_STATE       = 32
N_LAYERS      = 2
SSM_LR        = 1e-3
H_DIM         = N_LAYERS * N_STATE   # 64 — concat of all layer h for frozen projection

# Frozen projection config (same as step 1378)
TEMPERATURE   = 3.0

# Gating verdict threshold
GATE_THRESHOLD = 0.05   # |ratio - 1.0| > 5% → ACTION_CONDITIONAL (gating works)

CONDITIONS      = ['GATED', 'STANDARD']
DIAG_CONDITION  = 'GATED_MASKED'

MAX_N_ACTIONS      = 4103
I3_STEP            = 200
I4_EARLY_MAX       = 100
I4_LATE_MIN        = 1900
TIER1_STEPS        = 200


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
# Obs encoding
# ---------------------------------------------------------------------------

def _is_arc_obs(arr):
    return arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] == 64 and arr.shape[2] == 64


def _encode_obs(obs_arr):
    arr = np.asarray(obs_arr, dtype=np.float32)
    if _is_arc_obs(arr):
        frame = np.round(arr).astype(np.int32).squeeze(0)
        frame = np.clip(frame, 0, 15)
        one_hot = (frame[None, :, :] == np.arange(16, dtype=np.int32)[:, None, None]).astype(np.float32)
        return one_hot.flatten()
    return arr.flatten()


# ---------------------------------------------------------------------------
# Deterministic init helpers
# ---------------------------------------------------------------------------

def _det_init(m, n, scale=0.01):
    k = max(m, n)
    if k <= 256:
        W = det_weights(m, n)
        return (W * scale).astype(np.float32)
    else:
        i = np.arange(m, dtype=np.float64).reshape(-1, 1)
        j = np.arange(n, dtype=np.float64).reshape(1, -1)
        W = np.sin(i * 1.7 + j * 2.3 + 1.0)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        return (W / (norms + 1e-8) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def binomial_p_one_sided(wins, n):
    if n <= 0:
        return 1.0
    total = sum(math.comb(n, k) for k in range(wins, n + 1))
    return round(float(total) * (0.5 ** n), 6)


def entropy_from_actions(action_list, n_actions):
    if not action_list:
        return None
    counts = np.bincount(action_list, minlength=n_actions).astype(np.float32)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def visit_cv(counts):
    mean_c = float(counts.mean())
    if mean_c <= 0:
        return None
    return float(counts.std() / mean_c)


# ---------------------------------------------------------------------------
# Standard SSM Layer (1379 architecture — action in additive input)
# ---------------------------------------------------------------------------

class SSMLayer:
    def __init__(self, d, n_state, lr):
        self.d = d
        self.n = n_state
        self.lr = lr

        self.B       = _det_init(n_state, d, scale=0.1)
        self.C       = _det_init(d, n_state, scale=0.1)
        self.A_param = np.ones(n_state, dtype=np.float32) * 0.5
        self.W_delta = np.zeros((n_state, d), dtype=np.float32)
        self.b_delta = np.zeros(n_state, dtype=np.float32)

        self._init_B = self.B.copy()

        self.h = np.zeros(n_state, dtype=np.float32)
        self.S = np.zeros(n_state, dtype=np.float32)
        self._x = self._h_prev = self._A_diag = self._delta = self._delta_pre = None

    def forward(self, x):
        self._x = x
        self._h_prev = self.h.copy()
        delta_pre = np.clip(self.W_delta @ x + self.b_delta, -20.0, 20.0)
        delta = np.log1p(np.exp(delta_pre))
        self._delta_pre = delta_pre
        self._delta = delta
        A_diag = np.exp(-delta * self.A_param)
        self._A_diag = A_diag
        self.h = A_diag * self._h_prev + self.B @ x
        return self.C @ self.h

    def rtrl_update(self, e_y):
        if self._x is None:
            return np.zeros(self.d, dtype=np.float32)
        e_h = self.C.T @ e_y
        e_x = self.B.T @ e_h
        self.C -= self.lr * np.outer(e_y, self.h)
        self.B -= self.lr * np.outer(e_h, self._x)
        self.S = self._A_diag * self.S + (-self._delta * self._A_diag * self._h_prev)
        self.A_param -= self.lr * (e_h * self.S)
        self.A_param = np.clip(self.A_param, 0.01, 10.0)
        sigmoid_dp = 1.0 / (1.0 + np.exp(-self._delta_pre))
        dL_d_delta = e_h * (-self.A_param * self._A_diag * self._h_prev)
        dL_d_logit = dL_d_delta * sigmoid_dp
        self.W_delta -= self.lr * np.outer(dL_d_logit, self._x)
        self.b_delta -= self.lr * dL_d_logit
        return e_x

    def r3_weight_diff(self):
        return float(np.linalg.norm(self.B - self._init_B, 'fro'))

    def reset_state(self):
        self.h[:] = 0.0
        self.S[:] = 0.0
        self._x = self._h_prev = self._A_diag = self._delta = self._delta_pre = None


# ---------------------------------------------------------------------------
# Gated SSM Layer (NEW — multiplicative action gating)
# ---------------------------------------------------------------------------

class GatedSSMLayer:
    """
    h_inter = A_diag * h_prev + B @ obs_input
    gate    = sigmoid(W_gate[:, prev_act] + b_gate)   (N_STATE,)
    h_new   = h_inter * gate

    Action enters multiplicatively. If mask_gate=True, gate uses zero input → sigmoid(b_gate).
    RTRL: local gradient for W_gate/b_gate (immediate step credit).
          S trace for A_param (same as SSMLayer approximation, ignoring gate in recurrence).
    """

    def __init__(self, d, n_state, n_actions, lr):
        self.d = d
        self.n = n_state
        self.n_actions = n_actions
        self.lr = lr

        # Obs-only additive path (no action in B input)
        self.B       = _det_init(n_state, d, scale=0.1)
        self.C       = _det_init(d, n_state, scale=0.1)
        self.A_param = np.ones(n_state, dtype=np.float32) * 0.5
        self.W_delta = np.zeros((n_state, d), dtype=np.float32)
        self.b_delta = np.zeros(n_state, dtype=np.float32)

        # Action gate (init to 0 → gate ≈ 0.5)
        self.W_gate = np.zeros((n_state, n_actions), dtype=np.float32)
        self.b_gate = np.zeros(n_state, dtype=np.float32)

        self._init_B = self.B.copy()

        self.h = np.zeros(n_state, dtype=np.float32)
        self.S = np.zeros(n_state, dtype=np.float32)  # A RTRL trace

        self._x = self._h_prev = self._A_diag = self._delta = self._delta_pre = None
        self._gate = self._h_inter = None
        self._act = 0

    def forward(self, x, act, mask_gate=False):
        """
        x: obs input (D,)
        act: previous action index (int)
        mask_gate: if True, zero out action input to gate (action-blind test)
        """
        self._x = x
        self._h_prev = self.h.copy()
        self._act = act

        delta_pre = np.clip(self.W_delta @ x + self.b_delta, -20.0, 20.0)
        delta = np.log1p(np.exp(delta_pre))
        self._delta_pre = delta_pre
        self._delta = delta
        A_diag = np.exp(-delta * self.A_param)
        self._A_diag = A_diag

        h_inter = A_diag * self._h_prev + self.B @ x
        self._h_inter = h_inter

        if mask_gate:
            gate_logit = self.b_gate
        else:
            gate_logit = self.W_gate[:, act] + self.b_gate
        gate = 1.0 / (1.0 + np.exp(-np.clip(gate_logit, -20.0, 20.0)))
        self._gate = gate

        self.h = h_inter * gate
        return self.C @ self.h

    def rtrl_update(self, e_y):
        if self._x is None:
            return np.zeros(self.d, dtype=np.float32)

        e_h = self.C.T @ e_y

        # Update C
        self.C -= self.lr * np.outer(e_y, self.h)

        # Backprop through h_new = h_inter * gate
        e_h_inter = e_h * self._gate       # gradient to h_inter
        e_gate    = e_h * self._h_inter    # gradient to gate

        # Gradient to x (through B)
        e_x = self.B.T @ e_h_inter

        # Update B (additive path, obs only)
        self.B -= self.lr * np.outer(e_h_inter, self._x)

        # A_param RTRL (approximate: S trace for h_inter, ignores gate in recurrence)
        self.S = self._A_diag * self.S + (-self._delta * self._A_diag * self._h_prev)
        self.A_param -= self.lr * (e_h_inter * self.S)
        self.A_param = np.clip(self.A_param, 0.01, 10.0)

        # W_delta, b_delta
        sigmoid_dp  = 1.0 / (1.0 + np.exp(-self._delta_pre))
        dL_d_delta  = e_h_inter * (-self.A_param * self._A_diag * self._h_prev)
        dL_d_logit  = dL_d_delta * sigmoid_dp
        self.W_delta -= self.lr * np.outer(dL_d_logit, self._x)
        self.b_delta -= self.lr * dL_d_logit

        # W_gate, b_gate (local gradient — immediate step credit only)
        # dL/dgate = e_gate;  dgate/dz = gate*(1-gate)
        gate_grad = e_gate * self._gate * (1.0 - self._gate)
        self.W_gate[:, self._act] -= self.lr * gate_grad
        self.b_gate               -= self.lr * gate_grad

        return e_x

    def r3_weight_diff(self):
        return float(np.linalg.norm(self.B - self._init_B, 'fro'))

    def reset_state(self):
        self.h[:] = 0.0
        self.S[:] = 0.0
        self._x = self._h_prev = self._A_diag = self._delta = self._delta_pre = None
        self._gate = self._h_inter = None
        self._act = 0


# ---------------------------------------------------------------------------
# Unified SSM Substrate
# ---------------------------------------------------------------------------

class SSMSubstrate:
    """
    mode: 'gated'         — GatedSSMLayer, frozen projection try2.
          'gated_masked'  — GatedSSMLayer with mask_gate=True (action-blind test).
          'standard'      — SSMLayer (1379 FULL arch), frozen projection try2.

    W_fixed: (n_actions, H_DIM) frozen projection matrix (same per draw/game as step 1378).
    """

    def __init__(self, n_actions, mode='standard', W_fixed=None):
        self.n_actions = n_actions
        self.mode      = mode
        self._in_dim   = PROJ_DIM + ACT_EMBED_DIM if mode == 'standard' else PROJ_DIM
        self._W_fixed  = W_fixed  # (n_actions, H_DIM), set externally per draw/game

        self._obs_proj  = None
        self._W_in      = _det_init(D, self._in_dim, scale=0.1)
        self._b_in      = np.zeros(D, dtype=np.float32)

        if mode == 'standard':
            self._act_embed = _det_init(n_actions, ACT_EMBED_DIM, scale=0.1)
            self._layers    = [SSMLayer(D, N_STATE, SSM_LR) for _ in range(N_LAYERS)]
        else:
            self._act_embed = None
            self._layers    = [GatedSSMLayer(D, N_STATE, n_actions, SSM_LR) for _ in range(N_LAYERS)]

        self._W_pred = _det_init(PROJ_DIM, D, scale=0.1)
        self._b_pred = np.zeros(PROJ_DIM, dtype=np.float32)

        self._step         = 0
        self._prev_action  = 0
        self._prev_y       = None
        self._in_try2      = False
        self._visit_count  = np.zeros(n_actions, dtype=np.int32)

        self._try2_visit_at_i3   = None
        self._try2_actions_early = []
        self._try2_actions_late  = []
        self._try2_h_by_level    = {}
        self._current_level      = 0
        self._try2_max_level     = 0
        self._r3_diff            = None

        self._pred_losses_try1 = []
        self._pred_losses_try2 = []

    def _init_obs_proj(self, obs_flat):
        obs_dim = obs_flat.shape[0]
        i = np.arange(PROJ_DIM, dtype=np.float64).reshape(-1, 1)
        j = np.arange(obs_dim,  dtype=np.float64).reshape(1, -1)
        W = np.sin(i * 1.234 + j * 0.00731 + 0.5)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        self._obs_proj = (W / (norms + 1e-8)).astype(np.float32)

    def _ssm_forward(self, proj_obs):
        if self.mode == 'standard':
            act_emb = self._act_embed[self._prev_action]
            x_in = np.concatenate([proj_obs, act_emb])
            x = self._W_in @ x_in + self._b_in
            y = x
            for layer in self._layers:
                y = layer.forward(y)
        else:
            mask = (self.mode == 'gated_masked')
            x = self._W_in @ proj_obs + self._b_in
            y = x
            for layer in self._layers:
                y = layer.forward(y, self._prev_action, mask_gate=mask)
        return y

    def _rtrl_step(self, proj_obs_next, y):
        pred  = self._W_pred @ y + self._b_pred
        error = pred - proj_obs_next
        e     = self._W_pred.T @ error
        self._W_pred -= SSM_LR * np.outer(error, y)
        self._b_pred -= SSM_LR * error
        for layer in reversed(self._layers):
            e = layer.rtrl_update(e)
        return float(np.mean(error ** 2))

    def process(self, obs_arr):
        obs_flat = _encode_obs(obs_arr)
        if self._obs_proj is None:
            self._init_obs_proj(obs_flat)

        proj_obs = self._obs_proj @ obs_flat
        y = self._ssm_forward(proj_obs)
        self._prev_y = y

        if self._in_try2 and self._step % 200 == 0:
            h_concat = np.concatenate([l.h.copy() for l in self._layers])
            lvl = self._current_level
            if lvl not in self._try2_h_by_level:
                self._try2_h_by_level[lvl] = []
            if len(self._try2_h_by_level[lvl]) < 20:
                self._try2_h_by_level[lvl].append(h_concat)

        # Action selection: COUNT for try1, frozen projection for try2
        if self._in_try2:
            if self._W_fixed is not None:
                h_concat = np.concatenate([l.h for l in self._layers])
                logits = self._W_fixed @ h_concat / TEMPERATURE
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                action = int(np.random.choice(self.n_actions, p=probs))
            else:
                action = int(np.random.randint(self.n_actions))
        else:
            min_count  = self._visit_count.min()
            candidates = np.where(self._visit_count == min_count)[0]
            action     = int(np.random.choice(candidates))

        self._visit_count[action] += 1

        if self._in_try2 and self._step == I3_STEP:
            self._try2_visit_at_i3 = self._visit_count.copy()

        if self._in_try2:
            if self._step <= I4_EARLY_MAX:
                self._try2_actions_early.append(action)
            elif self._step >= I4_LATE_MIN:
                self._try2_actions_late.append(action)

        self._step += 1
        self._prev_action = action
        return action

    def update_after_step(self, obs_next, action, reward):
        if self._prev_y is None or self._obs_proj is None:
            return
        obs_next_flat = _encode_obs(np.asarray(obs_next, dtype=np.float32))
        proj_obs_next = self._obs_proj @ obs_next_flat
        loss = self._rtrl_step(proj_obs_next, self._prev_y)
        if self._in_try2:
            self._pred_losses_try2.append(loss)
        else:
            self._pred_losses_try1.append(loss)

    def on_level_transition(self, new_level=None):
        for layer in self._layers:
            layer.reset_state()
        self._prev_y = None
        if new_level is not None:
            self._current_level = new_level
            if self._in_try2:
                self._try2_max_level = max(self._try2_max_level, new_level)

    def prepare_for_try2(self):
        diffs = [l.r3_weight_diff() for l in self._layers]
        self._r3_diff = round(float(np.mean(diffs)), 6)
        for layer in self._layers:
            layer.reset_state()
        self._prev_y         = None
        self._step           = 0
        self._prev_action    = 0
        self._in_try2        = True
        self._visit_count[:] = 0
        self._current_level  = 0

    def compute_stage_metrics(self):
        i3_cv = None
        if self._try2_visit_at_i3 is not None:
            cv = visit_cv(self._try2_visit_at_i3[:self.n_actions])
            i3_cv = round(cv, 4) if cv is not None else None

        h_early = entropy_from_actions(self._try2_actions_early, self.n_actions)
        h_late  = entropy_from_actions(self._try2_actions_late,  self.n_actions)
        i4_reduction = None
        if h_early and h_late and h_early > 0:
            i4_reduction = round((h_early - h_late) / h_early, 4)

        i1_within = i1_between = i1_pass = None
        levels_ok = [lvl for lvl, vecs in self._try2_h_by_level.items() if len(vecs) >= 2]
        if len(levels_ok) >= 2:
            within_dists, between_dists = [], []
            for lvl in levels_ok:
                vecs = np.array(self._try2_h_by_level[lvl])
                for a in range(len(vecs)):
                    for b in range(a + 1, len(vecs)):
                        within_dists.append(float(np.linalg.norm(vecs[a] - vecs[b])))
            all_vecs = {lvl: np.array(self._try2_h_by_level[lvl]) for lvl in levels_ok}
            for ii in range(len(levels_ok)):
                for jj in range(ii + 1, len(levels_ok)):
                    for va in all_vecs[levels_ok[ii]]:
                        for vb in all_vecs[levels_ok[jj]]:
                            between_dists.append(float(np.linalg.norm(va - vb)))
            if within_dists and between_dists:
                i1_within  = round(float(np.mean(within_dists)), 4)
                i1_between = round(float(np.mean(between_dists)), 4)
                i1_pass    = bool(i1_within < i1_between)

        return {
            'i3_cv':          i3_cv,
            'i4_h_early':     round(h_early, 4) if h_early else None,
            'i4_h_late':      round(h_late,  4) if h_late  else None,
            'i4_reduction':   i4_reduction,
            'i1_within':      i1_within,
            'i1_between':     i1_between,
            'i1_pass':        i1_pass,
            'i5_max_level':   self._try2_max_level,
            'r3_weight_diff': self._r3_diff,
        }

    def get_stats(self):
        t1_mean = round(float(np.mean(self._pred_losses_try1)), 6) if self._pred_losses_try1 else None
        t2_mean = round(float(np.mean(self._pred_losses_try2)), 6) if self._pred_losses_try2 else None
        all_losses = self._pred_losses_try1 + self._pred_losses_try2
        overall    = round(float(np.mean(all_losses)), 6) if all_losses else None
        return {
            'pred_loss_try1':    t1_mean,
            'pred_loss_try2':    t2_mean,
            'pred_loss_overall': overall,
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, seed, max_steps):
    obs = env.reset(seed=seed)
    steps = 0
    level = 0
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
# Draw runner
# ---------------------------------------------------------------------------

def run_draw(draw_idx, draw_seed, cond_name, max_steps):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, cond_name, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    draw_results    = []
    try2_progress   = {}
    optimal_steps_d = {}
    mode = cond_name.lower().replace('-', '_')  # 'gated', 'standard', 'gated_masked'

    for game_idx, (game_name, label) in enumerate(zip(games, game_labels.values())):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = MAX_N_ACTIONS

        # Build frozen projection matrix: deterministic per (draw, game), same seed as step 1378
        w_rng  = np.random.RandomState(draw_seed * 100 + game_idx)
        W_fixed = w_rng.randn(n_actions, H_DIM).astype(np.float32)

        substrate = SSMSubstrate(n_actions=n_actions, mode=mode, W_fixed=W_fixed)

        p1, t1 = run_episode(env, substrate, n_actions, seed=0,         max_steps=max_steps)
        substrate.prepare_for_try2()
        np.random.seed(draw_seed * 1000 + 1)  # PRNG fix: same try2 RNG for both conditions
        p2, t2 = run_episode(env, substrate, n_actions, seed=TRY2_SEED, max_steps=max_steps)

        stats = substrate.get_stats()
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
            'pred_loss_try1':    stats['pred_loss_try1'],
            'pred_loss_try2':    stats['pred_loss_try2'],
            'pred_loss_overall': stats['pred_loss_overall'],
            'i3_cv':             stage['i3_cv'],
            'i4_reduction':      stage['i4_reduction'],
            'i1_pass':           stage['i1_pass'],
            'i5_max_level':      stage['i5_max_level'],
            'r3_weight_diff':    stage['r3_weight_diff'],
        }
        masked_row = mask_result_row(row, game_labels)

        fn = os.path.join(draw_dir, label_filename(label, STEP))
        with open(fn, 'w') as f:
            f.write(json.dumps(masked_row) + '\n')

        draw_results.append(masked_row)

    rhae = compute_rhae_try2(try2_progress, optimal_steps_d)
    pred_losses = [r['pred_loss_overall'] for r in draw_results if r.get('pred_loss_overall') is not None]
    draw_pred_loss = round(float(np.mean(pred_losses)), 6) if pred_losses else None

    print(f"  [{cond_name}] Draw {draw_idx} RHAE={rhae:.6e}  pred_loss={draw_pred_loss}")
    return round(rhae, 7), draw_pred_loss, draw_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Tier 1: timing check ──────────────────────────────────────────────
    print("=== TIER 1: timing check ===")
    games_t1, _ = select_games(seed=DRAW_SEEDS[0])
    env_t1 = make_game(games_t1[0])
    try:
        na_t1 = int(env_t1.n_actions)
    except AttributeError:
        na_t1 = MAX_N_ACTIONS

    w_rng_t1 = np.random.RandomState(DRAW_SEEDS[0] * 100)
    W_fixed_t1 = w_rng_t1.randn(na_t1, H_DIM).astype(np.float32)

    sub_t1 = SSMSubstrate(n_actions=na_t1, mode='gated', W_fixed=W_fixed_t1)
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)   # warmup
    t_tier1 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t_tier1

    ms_per_step = tier1_elapsed / TIER1_STEPS * 1000
    # draws × games × (try1+try2) × conditions + diag_draws × games × (try1+try2)
    n_eps_main = N_DRAWS * 3 * 2 * len(CONDITIONS)
    n_eps_diag = N_DIAG  * 3 * 2
    est_total_s = ms_per_step / 1000 * TRY1_STEPS * (n_eps_main + n_eps_diag)
    print(f"  {TIER1_STEPS} steps: {tier1_elapsed:.2f}s  ({ms_per_step:.2f}ms/step)")
    print(f"  Estimated total: {est_total_s:.0f}s ({est_total_s/60:.1f} min)")

    if est_total_s > MAX_SECONDS:
        max_steps = max(200, int(
            (MAX_SECONDS * 0.85) / (ms_per_step / 1000 * (n_eps_main + n_eps_diag))
        ))
        print(f"  Budget exceeded — capping at {max_steps} steps")
    else:
        max_steps = TRY1_STEPS
        print(f"  Under budget — full {max_steps} steps")

    # ── Main conditions (GATED, STANDARD) ────────────────────────────────
    print(f"\n=== STEP {STEP}: GATED vs STANDARD, {N_DRAWS} draws, {max_steps} steps ===")
    print(f"Seeds: {DRAW_SEEDS[0]}-{DRAW_SEEDS[-1]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rhae_by_cond      = {c: [] for c in CONDITIONS}
    pred_loss_by_cond = {c: [] for c in CONDITIONS}

    for cond_name in CONDITIONS:
        print(f"\n--- Condition: {cond_name} ---")
        t_cond = time.time()
        for di, ds in enumerate(DRAW_SEEDS):
            rhae, pred_loss, _ = run_draw(di, ds, cond_name, max_steps)
            rhae_by_cond[cond_name].append(round(rhae, 7))
            pred_loss_by_cond[cond_name].append(pred_loss)
        print(f"  Condition done in {time.time()-t_cond:.0f}s")

    # ── Action-blind diagnostic (GATED_MASKED, 10 draws) ─────────────────
    print(f"\n--- Diagnostic: GATED_MASKED ({N_DIAG} draws, same seeds as GATED draws 0-{N_DIAG-1}) ---")
    diag_rhae      = []
    diag_pred_loss = []
    t_diag = time.time()
    for di in range(N_DIAG):
        ds = DRAW_SEEDS[di]
        rhae, pred_loss, _ = run_draw(di, ds, DIAG_CONDITION, max_steps)
        diag_rhae.append(round(rhae, 7))
        diag_pred_loss.append(pred_loss)
    print(f"  Diagnostic done in {time.time()-t_diag:.0f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        'step':            STEP,
        'n_draws':         N_DRAWS,
        'n_diag':          N_DIAG,
        'draw_seeds':      DRAW_SEEDS,
        'max_steps_used':  max_steps,
        'ms_per_step':     round(ms_per_step, 2),
        'mlp_tp_baseline': MLP_TP_BASELINE,
        'conditions':      {},
    }

    print(f"\n=== RESULTS ===")
    for cond_name in CONDITIONS:
        rhae_list  = rhae_by_cond[cond_name]
        pred_list  = [p for p in pred_loss_by_cond[cond_name] if p is not None]
        chain_mean = round(sum(rhae_list) / len(rhae_list), 7)
        nz         = sum(1 for r in rhae_list if r > 0)
        pred_mean  = round(float(np.mean(pred_list)), 6) if pred_list else None
        verdict    = "SIGNAL" if chain_mean > MLP_TP_BASELINE else "KILL"
        print(f"  {cond_name}: chain_mean={chain_mean:.3e} ({nz}/{N_DRAWS} nz) [{verdict}]"
              f"  pred_loss={pred_mean}")
        summary['conditions'][cond_name] = {
            'chain_mean_rhae':    chain_mean,
            'nonzero_draws':      nz,
            'pred_loss_mean':     pred_mean,
            'rhae_per_draw':      rhae_list,      # internal — NOT mailed to Leo
            'pred_loss_per_draw': pred_loss_by_cond[cond_name],
        }

    # Diagnostic
    diag_chain   = round(sum(diag_rhae) / len(diag_rhae), 7)
    diag_pl_mean = round(float(np.mean([p for p in diag_pred_loss if p is not None])), 6) if any(p is not None for p in diag_pred_loss) else None
    summary['conditions'][DIAG_CONDITION] = {
        'chain_mean_rhae':    diag_chain,
        'nonzero_draws':      sum(1 for r in diag_rhae if r > 0),
        'pred_loss_mean':     diag_pl_mean,
        'rhae_per_draw':      diag_rhae,
        'pred_loss_per_draw': diag_pred_loss,
        'n_draws':            N_DIAG,
    }

    # ── Gating verdict ────────────────────────────────────────────────────
    # Compare GATED (first N_DIAG draws) vs GATED_MASKED to get pred_loss ratio
    gated_pred_diag = [p for p in pred_loss_by_cond['GATED'][:N_DIAG] if p is not None]
    gated_mean_diag = round(float(np.mean(gated_pred_diag)), 6) if gated_pred_diag else None

    gate_ratio = None
    gate_verdict = 'UNKNOWN'
    if gated_mean_diag is not None and diag_pl_mean is not None and diag_pl_mean > 0:
        gate_ratio = round(gated_mean_diag / diag_pl_mean, 6)
        pct_diff   = abs(gate_ratio - 1.0)
        if pct_diff < GATE_THRESHOLD:
            gate_verdict = 'GATE_FAILS'    # action not influencing prediction despite gating
        elif gate_ratio < 1.0 - GATE_THRESHOLD:
            gate_verdict = 'GATE_WORKS'    # GATED predicts better (lower loss) → action-conditional h
        else:
            gate_verdict = 'GATE_HURTS'    # masked predicts better (unexpected)

    print(f"\n=== GATING DIAGNOSTIC ===")
    print(f"  GATED pred_loss (draws 0-{N_DIAG-1}): {gated_mean_diag}")
    print(f"  GATED_MASKED pred_loss:                {diag_pl_mean}")
    print(f"  Ratio (GATED/MASKED):                  {gate_ratio}")
    if gate_verdict == 'GATE_WORKS':
        print(f"  *** GATE_WORKS: action-conditional prediction (ratio={gate_ratio:.4f} < {1.0-GATE_THRESHOLD:.2f}). h encodes actions. ***")
    elif gate_verdict == 'GATE_FAILS':
        print(f"  *** GATE_FAILS: gating doesn't force action conditioning (ratio={gate_ratio:.4f}, <{GATE_THRESHOLD*100:.0f}% diff). ***")
        print(f"  Kill criterion 1 triggered: gating architecture insufficient.")
    else:
        print(f"  {gate_verdict}: ratio={gate_ratio}")

    summary['gating_diagnostic'] = {
        'gated_pred_loss_diag':  gated_mean_diag,
        'masked_pred_loss_diag': diag_pl_mean,
        'ratio':                 gate_ratio,
        'verdict':               gate_verdict,
    }

    # ── RHAE paired comparison (GATED vs STANDARD) ───────────────────────
    gated_list    = rhae_by_cond['GATED']
    standard_list = rhae_by_cond['STANDARD']
    g_wins  = sum(1 for a, b in zip(gated_list, standard_list) if a > b)
    s_wins  = sum(1 for a, b in zip(gated_list, standard_list) if b > a)
    ties    = sum(1 for a, b in zip(gated_list, standard_list) if a == b)
    n_dec   = g_wins + s_wins
    p_val   = binomial_p_one_sided(g_wins, n_dec)

    gated_chain    = summary['conditions']['GATED']['chain_mean_rhae']
    standard_chain = summary['conditions']['STANDARD']['chain_mean_rhae']
    if gated_chain > MLP_TP_BASELINE and gated_chain > standard_chain:
        rhae_verdict = 'GATED_SIGNAL'
    elif standard_chain > MLP_TP_BASELINE and gated_chain <= standard_chain:
        rhae_verdict = 'STANDARD_DOMINATES'
    else:
        rhae_verdict = 'BOTH_KILL'

    print(f"\n=== RHAE PAIRED (GATED vs STANDARD) ===")
    print(f"  GATED wins {g_wins}/{N_DRAWS}, STANDARD wins {s_wins}/{N_DRAWS}, ties {ties}/{N_DRAWS}  p={p_val:.4f}")
    print(f"  RHAE verdict: {rhae_verdict}")

    summary['rhae_paired'] = {
        'gated_wins': g_wins, 'standard_wins': s_wins, 'ties': ties,
        'p_value': p_val,
        'rhae_verdict': rhae_verdict,
    }

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {RESULTS_DIR}/summary.json")
    return rhae_by_cond


if __name__ == '__main__':
    main()
