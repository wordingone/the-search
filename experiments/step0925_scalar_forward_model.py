"""
step0925_scalar_forward_model.py -- Scalar forward model on recurrent state h.

R3 hypothesis: State-conditioned change prediction via tiny scalar model (132 params).
h (64D recurrent state) encodes trajectory position. W_scalar predicts change MAGNITUDE
for each action from current h. Converges in ~1K steps (vs 256D W which never converges).
Enables state-conditioned action selection: "from THIS trajectory state, which action
produces the most change?"

Architecture:
- 916 recurrent h (echo-state, fixed W_h/W_x): trajectory state
- W_scalar (h_dim + n_actions,): scalar forward model on h. TINY — 132D for FT09.
- Training: delta rule on observed ||enc_t - enc_{t-1}|| (scalar)
- Action: softmax on 0.5*W_scalar_score + 0.5*delta_per_action
- 800b delta_per_action: global fallback (exploration history)

Graph ban check: W_scalar = continuous fn of h. h = recurrent (per-trajectory).
No discrete per-state counting. ALLOWED.

Run: FT09 (68 actions) + VC33 (68 actions) + LS20 (4 actions). 25K, 10 seeds cold.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
ETA_W_PRED = 0.01    # full forward model eta
ETA_SCALAR = 0.01    # scalar model eta
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
SCALAR_WEIGHT = 0.5  # blend: 0.5*scalar + 0.5*delta_per_action
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_action(scores, temp, rng):
    x = np.array(scores) / temp; x -= np.max(x); e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(scores), p=probs))


class ScalarForwardModel925:
    """916 recurrent h + scalar W_scalar for state-conditioned change prediction."""

    def __init__(self, n_actions, seed):
        self._n = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed recurrent weights (916 architecture)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Full forward model (for alpha signal generation only)
        EXT = ENC_DIM + H_DIM
        self.W_pred = np.zeros((EXT, EXT + n_actions), dtype=np.float32)
        self.alpha = np.ones(EXT, dtype=np.float32)

        # TINY scalar model: h + action → scalar change magnitude
        self.W_scalar = np.zeros(H_DIM + n_actions, dtype=np.float32)

        # 800b delta
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)

        # Hidden state
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)

        self._prev_enc = None
        self._prev_ext = None
        self._prev_h = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1-a)*self._running_mean + a*enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        ext = np.concatenate([enc, self.h]).astype(np.float32)
        return enc, ext

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY: return
        me = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(me)) or np.any(np.isinf(me)): return
        ra = np.sqrt(np.clip(me, 0, 1e6) + 1e-8); mr = np.mean(ra)
        if mr < 1e-8 or np.isnan(mr): return
        self.alpha = np.clip(ra / mr, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        enc, ext = self._encode(obs)

        if self._prev_enc is not None and self._prev_action is not None:
            # Alpha signal generation (same as 916)
            inp = np.concatenate([self._prev_ext * self.alpha,
                                   one_hot(self._prev_action, self._n)])
            pred = self.W_pred @ inp
            error = (ext * self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0: error *= 10.0/en
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W_PRED * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            # 800b change-tracking
            weighted_delta = (ext - self._prev_ext) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1-ALPHA_EMA)*self.delta_per_action[a] + ALPHA_EMA*change

            # Scalar model update: predict change from PREVIOUS h + action
            actual_change = float(np.linalg.norm(enc - self._prev_enc))  # raw obs change
            if self._prev_h is not None:
                scalar_inp = np.concatenate([self._prev_h, one_hot(self._prev_action, self._n)])
                pred_change = float(self.W_scalar @ scalar_inp)
                scalar_error = actual_change - pred_change
                self.W_scalar += ETA_SCALAR * scalar_error * scalar_inp

        # Action selection: scalar model + 800b delta
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n))
        else:
            h_now = self.h.copy()
            scalar_scores = np.array([
                float(self.W_scalar @ np.concatenate([h_now, one_hot(a, self._n)]))
                for a in range(self._n)
            ], dtype=np.float32)
            combined = SCALAR_WEIGHT * scalar_scores + (1 - SCALAR_WEIGHT) * self.delta_per_action
            action = softmax_action(combined, SOFTMAX_TEMP, self._rng)

        self._prev_enc = enc.copy()
        self._prev_ext = ext.copy()
        self._prev_h = self.h.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_ext = None
        self._prev_h = None; self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def scalar_top_actions(self, k=7):
        h_now = self.h.copy()
        scores = np.array([
            float(self.W_scalar @ np.concatenate([h_now, one_hot(a, self._n)]))
            for a in range(self._n)
        ])
        return list(np.argsort(scores)[-k:])


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps, baseline_actions=None):
    results = []
    for seed in seeds:
        sub = ScalarForwardModel925(n_actions=n_actions, seed=seed)
        env = make_game(game_name)
        obs = env.reset(seed=seed * 1000)
        step = 0; completions = 0; current_level = 0
        while step < n_steps:
            if obs is None:
                obs = env.reset(seed=seed * 1000); sub.on_level_transition(); continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
            obs, _, done, info = env.step(action)
            step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > current_level:
                completions += (cl - current_level); current_level = cl
                sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed * 1000); current_level = 0
                sub.on_level_transition()
        results.append(completions)
        top = sub.scalar_top_actions(7) if baseline_actions else []
        overlap = [a for a in top if a in (baseline_actions or [])]
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}"
              + (f"  scalar_top7={top}  baseline_overlap={overlap}" if baseline_actions else ""))
    return results


FT09_BASELINE = [17, 19, 15, 21, 65, 26]
VC33_BASELINE = [6, 13, 31, 59, 92, 24, 82]

print("=" * 70)
print("STEP 925 — SCALAR FORWARD MODEL ON RECURRENT STATE h")
print("=" * 70)
print("W_scalar (132D): h + action → scalar change. State-conditioned selection.")
t0 = time.time()

print("\n--- FT09 (68 actions, 25K, 10 seeds) ---")
ft09 = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS, FT09_BASELINE)
ft09_mean = np.mean(ft09); ft09_z = sum(1 for x in ft09 if x == 0)
print(f"  FT09: L1={ft09_mean:.1f}/seed  std={np.std(ft09):.1f}  zero={ft09_z}/10")
print(f"  {ft09}")

print("\n--- VC33 (68 actions, 25K, 10 seeds) ---")
vc33 = run_game("VC33", 68, TEST_SEEDS, TEST_STEPS, VC33_BASELINE)
vc33_mean = np.mean(vc33); vc33_z = sum(1 for x in vc33 if x == 0)
print(f"  VC33: L1={vc33_mean:.1f}/seed  std={np.std(vc33):.1f}  zero={vc33_z}/10")
print(f"  {vc33}")

print("\n--- LS20 (4 actions, 25K, 10 seeds) ---")
ls20 = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS)
ls20_mean = np.mean(ls20); ls20_z = sum(1 for x in ls20 if x == 0)
print(f"  LS20: L1={ls20_mean:.1f}/seed  std={np.std(ls20):.1f}  zero={ls20_z}/10")
print(f"  {ls20}")

print(f"\n{'='*70}")
print(f"STEP 925 RESULTS (scalar forward model on h):")
print(f"  FT09: L1={ft09_mean:.1f}/seed  zero={ft09_z}/10  (baseline: 0/10)")
print(f"  VC33: L1={vc33_mean:.1f}/seed  zero={vc33_z}/10  (baseline: 0/10)")
print(f"  LS20: L1={ls20_mean:.1f}/seed  zero={ls20_z}/10  (916 best: 290.7)")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 925 DONE")
