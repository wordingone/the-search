"""
Step 1381 — Inverse dynamics: change the objective, not the architecture.
Leo mail 3943, 2026-03-30.

1379: SSM is action-blind — obs prediction doesn't need actions (ratio=1.00002).
1380: Multiplicative gating degenerates — architecture can't force what objective ignores (ratio=1.000119).
Root cause: prediction loss gradient is action-indifferent. Any architecture trained only by obs
prediction cannot become action-conditional.

Fix: add inverse dynamics head. TWO prediction heads, joint RTRL:
  Forward head (existing): predict proj_obs_{t+1} from y_t. Loss = MSE.
  Inverse head (NEW):      predict action_t from (y_t, proj_obs_{t+1}). Loss = CrossEntropy.

Joint loss: L = L_forward + L_inverse.

Inverse head FORCES action info into h: cannot predict which action was taken without encoding
action-state relationships. RTRL optimizes both: obs-conditional h AND action-conditional h.

Conditions (paired, 30 draws):
  INV: Forward + inverse heads. Frozen projection try2 (W_fixed @ h_concat / T=3).
  FWD: Forward head only (= 1380 STANDARD architecture). Frozen projection try2. Control.
Both: COUNT try1, PRNG fix, h reset, weights carry. Same W_fixed per (draw, game).

Action-blind diagnostic (10 draws):
  INV_MASKED: Same as INV but action token zeroed in SSM input. Inverse head still active.
  Tests: does masking SSM input token change pred_loss? If ratio > 1.05 with masked input →
  h is action-conditional through GRADIENT (not input token).

Kill criteria:
  1. INV action-blind ratio (INV/INV_MASKED) < 1.05 → inverse head didn't force action info.
  2. ratio > 1.05 but INV RHAE ≤ FWD RHAE → action features exist but are backward-looking.

Seeds: 14020-14049 (fresh).
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

STEP        = 1381
N_DRAWS     = 30
DRAW_SEEDS  = [14020 + i for i in range(N_DRAWS)]
N_DIAG      = 10
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

RESULTS_DIR     = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')
MLP_TP_BASELINE = 4.59e-5

# SSM config (same as 1378-1380)
PROJ_DIM      = 64
ACT_EMBED_DIM = 16
D             = 128
N_STATE       = 32
N_LAYERS      = 2
SSM_LR        = 1e-3
H_DIM         = N_LAYERS * N_STATE   # 64 — concat of all layer h for frozen projection

# Frozen projection (same as 1378/1380)
TEMPERATURE   = 3.0

# Verdict threshold
ACTION_BLIND_THRESHOLD = 0.05   # |ratio - 1.0| > 5% → action-conditional

CONDITIONS     = ['INV', 'FWD']
DIAG_CONDITION = 'INV_MASKED'

MAX_N_ACTIONS = 4103
I3_STEP       = 200
I4_EARLY_MAX  = 100
I4_LATE_MIN   = 1900
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
# SSM Layer (same as 1379-1380)
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
# SSM Substrate
# ---------------------------------------------------------------------------

class SSMSubstrate:
    """
    mode: 'inv'        — forward + inverse heads. Frozen projection try2.
          'fwd'        — forward head only (control). Frozen projection try2.
          'inv_masked' — same as 'inv' but action token zeroed in SSM input.

    W_fixed: (n_actions, H_DIM) frozen projection matrix.
    """

    def __init__(self, n_actions, mode='fwd', W_fixed=None):
        self.n_actions = n_actions
        self.mode      = mode
        self._W_fixed  = W_fixed
        self._in_dim   = PROJ_DIM + ACT_EMBED_DIM

        self._obs_proj  = None
        self._act_embed = _det_init(n_actions, ACT_EMBED_DIM, scale=0.1)
        self._zero_act  = np.zeros(ACT_EMBED_DIM, dtype=np.float32)
        self._W_in      = _det_init(D, self._in_dim, scale=0.1)
        self._b_in      = np.zeros(D, dtype=np.float32)
        self._layers    = [SSMLayer(D, N_STATE, SSM_LR) for _ in range(N_LAYERS)]

        # Forward prediction head
        self._W_pred = _det_init(PROJ_DIM, D, scale=0.1)
        self._b_pred = np.zeros(PROJ_DIM, dtype=np.float32)

        # Inverse dynamics head: predict action from (y_t, proj_obs_{t+1})
        # Input dim = D + PROJ_DIM = 192
        self._W_inv  = _det_init(n_actions, D + PROJ_DIM, scale=0.1)

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

        self._pred_losses   = []
        self._inv_correct   = 0     # inverse head: correctly predicted actions
        self._inv_total     = 0     # inverse head: total predictions

    def _init_obs_proj(self, obs_flat):
        obs_dim = obs_flat.shape[0]
        i = np.arange(PROJ_DIM, dtype=np.float64).reshape(-1, 1)
        j = np.arange(obs_dim,  dtype=np.float64).reshape(1, -1)
        W = np.sin(i * 1.234 + j * 0.00731 + 0.5)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        self._obs_proj = (W / (norms + 1e-8)).astype(np.float32)

    def _ssm_forward(self, proj_obs):
        # Mask action token in SSM input for inv_masked
        if self.mode == 'inv_masked':
            act_emb = self._zero_act
        else:
            act_emb = self._act_embed[self._prev_action]
        x_in = np.concatenate([proj_obs, act_emb])
        x = self._W_in @ x_in + self._b_in
        y = x
        for layer in self._layers:
            y = layer.forward(y)
        return y

    def _rtrl_step(self, proj_obs_next, y, action):
        # Forward head
        pred_obs  = self._W_pred @ y + self._b_pred
        error_obs = pred_obs - proj_obs_next
        e_y       = self._W_pred.T @ error_obs
        self._W_pred -= SSM_LR * np.outer(error_obs, y)
        self._b_pred -= SSM_LR * error_obs
        fwd_loss  = float(np.mean(error_obs ** 2))

        # Inverse head (only for 'inv' and 'inv_masked' modes)
        if self.mode in ('inv', 'inv_masked'):
            z         = np.concatenate([y, proj_obs_next])   # (D + PROJ_DIM,)
            logits    = self._W_inv @ z
            logits   -= logits.max()
            softmax_p = np.exp(logits); softmax_p /= softmax_p.sum()
            e_logits  = softmax_p.copy(); e_logits[action] -= 1.0   # (softmax - one_hot)
            e_z       = self._W_inv.T @ e_logits                     # (D + PROJ_DIM,)
            e_y_inv   = e_z[:D]                                       # gradient to y
            self._W_inv -= SSM_LR * np.outer(e_logits, z)
            e_y = e_y + e_y_inv   # combine forward + inverse gradients

            # Track accuracy
            self._inv_total += 1
            if int(np.argmax(logits)) == action:
                self._inv_correct += 1

        # RTRL through SSM layers
        for layer in reversed(self._layers):
            e_y = layer.rtrl_update(e_y)

        return fwd_loss

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

        # Action selection: COUNT try1, frozen projection try2
        if self._in_try2:
            if self._W_fixed is not None:
                h_concat = np.concatenate([l.h for l in self._layers])
                logits = self._W_fixed @ h_concat / TEMPERATURE
                logits -= logits.max()
                probs = np.exp(logits); probs /= probs.sum()
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
        loss = self._rtrl_step(proj_obs_next, self._prev_y, action)
        self._pred_losses.append(loss)

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
        pred_mean = round(float(np.mean(self._pred_losses)), 6) if self._pred_losses else None
        inv_acc   = None
        if self._inv_total > 0:
            inv_acc = round(self._inv_correct / self._inv_total, 4)
        return {
            'pred_loss_mean': pred_mean,
            'inv_accuracy':   inv_acc,
            'inv_total':      self._inv_total,
            'inv_random_baseline': round(1.0 / self.n_actions, 4) if self.n_actions > 0 else None,
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
    mode = cond_name.lower()  # 'inv', 'fwd', 'inv_masked'

    for game_idx, (game_name, label) in enumerate(zip(games, game_labels.values())):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = MAX_N_ACTIONS

        # Frozen projection: deterministic per (draw, game), same seed as 1378/1380
        w_rng   = np.random.RandomState(draw_seed * 100 + game_idx)
        W_fixed = w_rng.randn(n_actions, H_DIM).astype(np.float32)

        substrate = SSMSubstrate(n_actions=n_actions, mode=mode, W_fixed=W_fixed)

        p1, t1 = run_episode(env, substrate, n_actions, seed=0,         max_steps=max_steps)
        substrate.prepare_for_try2()
        np.random.seed(draw_seed * 1000 + 1)  # PRNG fix
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
            'pred_loss_mean':     stats['pred_loss_mean'],
            'inv_accuracy':       stats['inv_accuracy'],
            'inv_random_baseline': stats['inv_random_baseline'],
            'i3_cv':              stage['i3_cv'],
            'i4_h_early':         stage['i4_h_early'],
            'i4_reduction':       stage['i4_reduction'],
            'i1_pass':            stage['i1_pass'],
            'i5_max_level':       stage['i5_max_level'],
            'r3_weight_diff':     stage['r3_weight_diff'],
        }
        masked_row = mask_result_row(row, game_labels)

        fn = os.path.join(draw_dir, label_filename(label, STEP))
        with open(fn, 'w') as f:
            f.write(json.dumps(masked_row) + '\n')

        draw_results.append(masked_row)

    rhae = compute_rhae_try2(try2_progress, optimal_steps_d)
    pred_losses = [r['pred_loss_mean'] for r in draw_results if r.get('pred_loss_mean') is not None]
    draw_pred   = round(float(np.mean(pred_losses)), 6) if pred_losses else None
    inv_accs    = [r['inv_accuracy'] for r in draw_results if r.get('inv_accuracy') is not None]
    draw_inv_acc = round(float(np.mean(inv_accs)), 4) if inv_accs else None

    print(f"  [{cond_name}] Draw {draw_idx} RHAE={rhae:.6e}  pred_loss={draw_pred}  inv_acc={draw_inv_acc}")
    return round(rhae, 7), draw_pred, draw_inv_acc, draw_results


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

    w_rng_t1  = np.random.RandomState(DRAW_SEEDS[0] * 100)
    W_fixed_t1 = w_rng_t1.randn(na_t1, H_DIM).astype(np.float32)

    sub_t1 = SSMSubstrate(n_actions=na_t1, mode='inv', W_fixed=W_fixed_t1)
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)   # warmup
    t_tier1 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t_tier1

    ms_per_step = tier1_elapsed / TIER1_STEPS * 1000
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

    # ── Main conditions (INV, FWD) ────────────────────────────────────────
    print(f"\n=== STEP {STEP}: INV vs FWD, {N_DRAWS} draws, {max_steps} steps ===")
    print(f"Seeds: {DRAW_SEEDS[0]}-{DRAW_SEEDS[-1]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rhae_by_cond     = {c: [] for c in CONDITIONS}
    pred_by_cond     = {c: [] for c in CONDITIONS}
    inv_acc_by_cond  = {c: [] for c in CONDITIONS}

    for cond_name in CONDITIONS:
        print(f"\n--- Condition: {cond_name} ---")
        t_cond = time.time()
        for di, ds in enumerate(DRAW_SEEDS):
            rhae, pred, inv_acc, _ = run_draw(di, ds, cond_name, max_steps)
            rhae_by_cond[cond_name].append(round(rhae, 7))
            pred_by_cond[cond_name].append(pred)
            inv_acc_by_cond[cond_name].append(inv_acc)
        print(f"  Condition done in {time.time()-t_cond:.0f}s")

    # ── Action-blind diagnostic (INV_MASKED, 10 draws) ────────────────────
    print(f"\n--- Diagnostic: INV_MASKED ({N_DIAG} draws, seeds {DRAW_SEEDS[0]}-{DRAW_SEEDS[N_DIAG-1]}) ---")
    diag_rhae = []; diag_pred = []; diag_inv_acc = []
    t_diag = time.time()
    for di in range(N_DIAG):
        ds = DRAW_SEEDS[di]
        rhae, pred, inv_acc, _ = run_draw(di, ds, DIAG_CONDITION, max_steps)
        diag_rhae.append(round(rhae, 7)); diag_pred.append(pred); diag_inv_acc.append(inv_acc)
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
        pred_list  = [p for p in pred_by_cond[cond_name] if p is not None]
        acc_list   = [a for a in inv_acc_by_cond[cond_name] if a is not None]
        chain_mean = round(sum(rhae_list) / len(rhae_list), 7)
        nz         = sum(1 for r in rhae_list if r > 0)
        pred_mean  = round(float(np.mean(pred_list)), 6) if pred_list else None
        acc_mean   = round(float(np.mean(acc_list)), 4)  if acc_list  else None
        verdict    = "SIGNAL" if chain_mean > MLP_TP_BASELINE else "KILL"
        print(f"  {cond_name}: chain_mean={chain_mean:.3e} ({nz}/{N_DRAWS} nz) [{verdict}]"
              f"  pred={pred_mean}  inv_acc={acc_mean}")
        summary['conditions'][cond_name] = {
            'chain_mean_rhae':    chain_mean,
            'nonzero_draws':      nz,
            'pred_loss_mean':     pred_mean,
            'inv_acc_mean':       acc_mean,
            'rhae_per_draw':      rhae_list,      # internal — NOT mailed to Leo
            'pred_per_draw':      pred_by_cond[cond_name],
        }

    # Diagnostic entry
    diag_chain  = round(sum(diag_rhae) / len(diag_rhae), 7) if diag_rhae else 0.0
    diag_p_mean = round(float(np.mean([p for p in diag_pred if p is not None])), 6) if any(p is not None for p in diag_pred) else None
    diag_a_mean = round(float(np.mean([a for a in diag_inv_acc if a is not None])), 4) if any(a is not None for a in diag_inv_acc) else None
    summary['conditions'][DIAG_CONDITION] = {
        'chain_mean_rhae': diag_chain,
        'nonzero_draws':   sum(1 for r in diag_rhae if r > 0),
        'pred_loss_mean':  diag_p_mean,
        'inv_acc_mean':    diag_a_mean,
        'rhae_per_draw':   diag_rhae,
        'pred_per_draw':   diag_pred,
        'n_draws':         N_DIAG,
    }

    # ── Action-blind verdict ──────────────────────────────────────────────
    # Compare INV (draws 0-9) vs INV_MASKED pred_loss
    inv_pred_diag = [p for p in pred_by_cond['INV'][:N_DIAG] if p is not None]
    inv_mean_diag = round(float(np.mean(inv_pred_diag)), 6) if inv_pred_diag else None

    ratio = verdict_ab = None
    if inv_mean_diag is not None and diag_p_mean is not None and diag_p_mean > 0:
        ratio = round(inv_mean_diag / diag_p_mean, 6)
        pct   = abs(ratio - 1.0)
        if pct < ACTION_BLIND_THRESHOLD:
            verdict_ab = 'ACTION_BLIND'
        elif ratio < 1.0 - ACTION_BLIND_THRESHOLD:
            verdict_ab = 'ACTION_CONDITIONAL'
        else:
            verdict_ab = 'UNEXPECTED'

    print(f"\n=== ACTION-BLIND DIAGNOSTIC ===")
    print(f"  INV pred_loss (draws 0-{N_DIAG-1}): {inv_mean_diag}")
    print(f"  INV_MASKED pred_loss:                {diag_p_mean}")
    print(f"  Ratio (INV/MASKED):                  {ratio}")
    if verdict_ab == 'ACTION_CONDITIONAL':
        print(f"  *** ACTION_CONDITIONAL: inverse head forces action info (ratio={ratio:.4f} < {1-ACTION_BLIND_THRESHOLD:.2f}). ***")
        print(f"  h is action-conditional. Kill criterion 1 NOT triggered.")
    elif verdict_ab == 'ACTION_BLIND':
        print(f"  *** ACTION_BLIND: inverse head failed (ratio={ratio:.4f}). Kill criterion 1 triggered. ***")
    else:
        print(f"  {verdict_ab}: ratio={ratio}")

    summary['action_blind_diagnostic'] = {
        'inv_pred_loss_diag':    inv_mean_diag,
        'masked_pred_loss_diag': diag_p_mean,
        'ratio':                 ratio,
        'verdict':               verdict_ab,
    }

    # ── RHAE paired (INV vs FWD) ──────────────────────────────────────────
    inv_list = rhae_by_cond['INV']
    fwd_list = rhae_by_cond['FWD']
    i_wins = sum(1 for a, b in zip(inv_list, fwd_list) if a > b)
    f_wins = sum(1 for a, b in zip(inv_list, fwd_list) if b > a)
    ties   = sum(1 for a, b in zip(inv_list, fwd_list) if a == b)
    n_dec  = i_wins + f_wins
    p_val  = binomial_p_one_sided(i_wins, n_dec)

    inv_chain = summary['conditions']['INV']['chain_mean_rhae']
    fwd_chain = summary['conditions']['FWD']['chain_mean_rhae']
    if inv_chain > MLP_TP_BASELINE and inv_chain > fwd_chain:
        rhae_verdict = 'INV_SIGNAL'
    elif fwd_chain > MLP_TP_BASELINE and inv_chain <= fwd_chain:
        rhae_verdict = 'FWD_DOMINATES'
    else:
        rhae_verdict = 'BOTH_KILL'

    print(f"\n=== RHAE PAIRED (INV vs FWD) ===")
    print(f"  INV wins {i_wins}/{N_DRAWS}, FWD wins {f_wins}/{N_DRAWS}, ties {ties}/{N_DRAWS}  p={p_val:.4f}")
    print(f"  RHAE verdict: {rhae_verdict}")

    # Inverse head accuracy summary
    inv_acc_overall = summary['conditions']['INV'].get('inv_acc_mean')
    print(f"\n=== INVERSE HEAD ACCURACY ===")
    print(f"  INV inv_acc = {inv_acc_overall}")
    # FWD has no inverse head, skip

    summary['rhae_paired'] = {
        'inv_wins': i_wins, 'fwd_wins': f_wins, 'ties': ties,
        'p_value': p_val, 'rhae_verdict': rhae_verdict,
    }

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {RESULTS_DIR}/summary.json")
    return rhae_by_cond


if __name__ == '__main__':
    main()
