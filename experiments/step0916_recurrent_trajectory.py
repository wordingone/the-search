"""
step0916_recurrent_trajectory.py -- Recurrent hidden state as trajectory working memory.

R3 hypothesis: A fixed-random recurrent encoder h_t = tanh(W_h @ h_{t-1} + W_x @ enc_t)
provides implicit working memory for sequential ordering. After clicking tile A then B,
h encodes trajectory context — ext_enc = [enc, h] looks different from clicking B then A.
Alpha concentrates on h dimensions tracking sequential progress. 800b tracks change in
extended encoding, providing richer change signal for navigation.

Architecture: echo-state / reservoir over trajectories.
- W_h (64×64), W_x (64×256): FIXED random. Only W_pred trains.
- ext_enc = concat([enc, h]) = 320D
- Alpha (320D): clamped 0.1-5.0 from prediction error on ext_enc
- W_pred (320 × 320+n_actions): forward model on ext_enc
- delta_per_action: EMA of ||alpha * (ext_enc_t - ext_enc_{t-1})|| = 800b on extended space

Graph ban check: h = f(h_{t-1}, enc_t) — accumulated from TRAJECTORY sequence, not
per-(state,action) table. No per-state data. ALLOWED.

Run: FT09 (68 actions) + LS20 (4 actions). 25K, 10 seeds, cold start, substrate_seed=seed.
Kill criterion: if h fails to differentiate sequences by 5K steps, killed.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM   # 320
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(delta, temp, rng):
    x = delta / temp
    x = x - np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class RecurrentTrajectory916:
    """Recurrent trajectory encoder + clamped alpha + 800b change-tracking."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)  # fixed substrate seed

        # Fixed random recurrent weights (never trained)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Trainable forward model on extended encoding
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), dtype=np.float32)

        # Alpha on extended encoding
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)

        # 800b change-tracking
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)

        # Recurrent state
        self.h = np.zeros(H_DIM, dtype=np.float32)

        # Running mean for centering
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        # History buffers
        self._pred_errors = deque(maxlen=200)
        self._prev_ext = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean

        # Update recurrent state
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)

        # Extended encoding
        return np.concatenate([enc, self.h]).astype(np.float32)

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY:
            return
        mean_errors = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(mean_errors)) or np.any(np.isinf(mean_errors)):
            return
        raw_alpha = np.sqrt(np.clip(mean_errors, 0, 1e6) + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8 or np.isnan(mean_raw):
            return
        self.alpha = raw_alpha / mean_raw
        self.alpha = np.clip(self.alpha, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        ext_enc = self._encode(obs)

        if self._prev_ext is not None and self._prev_action is not None:
            # Forward model update on ext_enc
            inp = np.concatenate([self._prev_ext * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W_pred @ inp
            error = (ext_enc * self.alpha) - pred

            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            # 800b change-tracking on extended encoding
            weighted_delta = (ext_enc - self._prev_ext) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1 - ALPHA_EMA) * self.delta_per_action[a] + ALPHA_EMA * change

        # Action selection
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_action(self.delta_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_ext = ext_enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_action = None
        # Do NOT reset h — trajectory context persists (test key design choice)

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def h_variance(self):
        """Average variance in h across episode — should be >0 if h differentiates."""
        return float(np.var(self.h))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    concs = []
    for seed in seeds:
        sub = RecurrentTrajectory916(n_actions=n_actions, seed=seed)
        env = make_game(game_name)
        obs = env.reset(seed=seed * 1000)
        step = 0; completions = 0; current_level = 0
        while step < n_steps:
            if obs is None:
                obs = env.reset(seed=seed * 1000)
                sub.on_level_transition()
                continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
            obs, _, done, info = env.step(action)
            step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > current_level:
                completions += (cl - current_level)
                current_level = cl
                sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed * 1000)
                current_level = 0
                sub.on_level_transition()
        results.append(completions)
        concs.append(sub.alpha_conc())
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}  h_var={sub.h_variance():.4f}")
    return results, concs


print("=" * 70)
print("STEP 916 — RECURRENT TRAJECTORY ENCODING")
print("=" * 70)
print("R3: fixed-random recurrent h_t = tanh(W_h@h + W_x@enc). ext_enc=320D.")
print("Alpha + 800b on extended encoding. FT09 + LS20, 25K, 10 seeds cold.")
t0 = time.time()

# FT09 (68 actions)
print("\n--- FT09 (68 actions, 25K, 10 seeds) ---")
ft09_results, ft09_concs = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS)
ft09_mean = np.mean(ft09_results)
ft09_std = np.std(ft09_results)
ft09_zeros = sum(1 for x in ft09_results if x == 0)
print(f"  FT09: L1={ft09_mean:.1f}/seed  std={ft09_std:.1f}  zero={ft09_zeros}/10  alpha_conc={np.mean(ft09_concs):.2f}")
print(f"  {ft09_results}")

# LS20 (4 actions)
print("\n--- LS20 (4 actions, 25K, 10 seeds) ---")
ls20_results, ls20_concs = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS)
ls20_mean = np.mean(ls20_results)
ls20_std = np.std(ls20_results)
ls20_zeros = sum(1 for x in ls20_results if x == 0)
print(f"  LS20: L1={ls20_mean:.1f}/seed  std={ls20_std:.1f}  zero={ls20_zeros}/10  alpha_conc={np.mean(ls20_concs):.2f}")
print(f"  {ls20_results}")

print(f"\n{'='*70}")
print(f"STEP 916 RESULTS:")
print(f"  FT09: L1={ft09_mean:.1f}/seed  zero={ft09_zeros}/10  (baseline: 0.0/10)")
print(f"  LS20: L1={ls20_mean:.1f}/seed  zero={ls20_zeros}/10  (895h cold: 268.0/seed 0/10)")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 916 DONE")
