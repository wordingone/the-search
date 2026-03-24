"""Step 951: Mean-subtracted Hebbian (signal quality, not timing).

R3 hypothesis: Mean-subtracted prediction error provides differential action
reinforcement from step 1. Actions producing above-average novelty get reinforced,
below-average get SUPPRESSED (signed updates). Creates differentiation that raw
delta cannot — analogous to advantage estimation in RL.

Changes from 948 (h_dim=64, fixed W_h):
  - delta_ema tracks running mean of delta (EMA decay=0.99)
  - W_a update uses delta_centered = delta - delta_ema (signed)
  - W_pred uses raw error (unchanged)
  - W_h fixed random (same as 948, not trained)

Kill: LS20 ≤1/10 seeds nonzero (no improvement over 948).
Success: 3+ seeds L1>0 (mean subtraction breaks symmetry).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
ETA_PRED = 0.01
ETA_ACTION = 0.001
EPSILON = 0.20
DELTA_EMA_DECAY = 0.99
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class HebbianMeanSubtracted951:
    """Hebbian RNN with mean-subtracted delta for signed W_a updates."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed random recurrent weights (same as 948)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        self.W_pred = np.zeros((ENC_DIM, H_DIM), dtype=np.float32)
        self.W_a = np.zeros((n_actions, H_DIM), dtype=np.float32)

        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._delta_ema = 0.0   # running mean of delta
        self._prev_h = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = sigmoid(self.W_h @ self.h + self.W_x @ enc)
        return enc

    def process(self, obs):
        enc = self._encode(obs)

        if self._prev_h is not None and self._prev_action is not None:
            pred = self.W_pred @ self._prev_h
            error = enc - pred
            delta = float(np.linalg.norm(error))

            if delta > 10.0:
                error = error * (10.0 / delta)
                delta = 10.0

            if not np.any(np.isnan(error)):
                # Update delta EMA (running mean of prediction error magnitude)
                self._delta_ema = DELTA_EMA_DECAY * self._delta_ema + (1 - DELTA_EMA_DECAY) * delta

                # W_pred: raw error (all prediction errors are informative)
                self.W_pred += ETA_PRED * np.outer(error, self._prev_h)

                # W_a: mean-subtracted (signed) — above-average novelty reinforced,
                # below-average suppressed
                delta_centered = delta - self._delta_ema
                self.W_a[self._prev_action] += ETA_ACTION * delta_centered * self.h

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            scores = self.W_a @ self.h
            action = int(np.argmax(scores))

        self._prev_h = self.h.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_h = None
        self._prev_action = None

    def w_a_norm(self):
        return float(np.linalg.norm(self.W_a))

    def delta_ema(self):
        return self._delta_ema


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    w_norms = []
    delta_emas = []
    for seed in seeds:
        sub = HebbianMeanSubtracted951(n_actions=n_actions, seed=seed)
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
        w_norms.append(sub.w_a_norm())
        delta_emas.append(sub.delta_ema())
        print(f"    seed={seed}: L1={completions:4d}  W_a_norm={sub.w_a_norm():.3f}  delta_ema={sub.delta_ema():.3f}")
    return results, w_norms, delta_emas


if __name__ == "__main__":
    import os, time

    print("=" * 70)
    print("STEP 951 — HEBBIAN RNN MEAN-SUBTRACTED DELTA")
    print("=" * 70)
    t0 = time.time()

    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), 'unknown')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), 'unknown')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")
    print(f"H_DIM={H_DIM}  ETA_PRED={ETA_PRED}  ETA_ACTION={ETA_ACTION}  EPSILON={EPSILON}  DELTA_EMA_DECAY={DELTA_EMA_DECAY}")
    print()

    print("--- LS20 (4 actions, 10K steps) ---")
    ls20_r, ls20_norms, ls20_demas = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS)
    ls20_mean = np.mean(ls20_r)
    ls20_nonzero = sum(1 for x in ls20_r if x > 0)
    print(f"  LS20: L1={ls20_mean:.1f}/seed  nonzero={ls20_nonzero}/10  W_a_norm={np.mean(ls20_norms):.3f}  delta_ema={np.mean(ls20_demas):.3f}")
    print(f"  {ls20_r}")

    print()
    print("--- FT09 (68 actions, 10K steps) ---")
    ft09_r, ft09_norms, ft09_demas = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS)
    ft09_mean = np.mean(ft09_r)
    ft09_nonzero = sum(1 for x in ft09_r if x > 0)
    print(f"  FT09: L1={ft09_mean:.1f}/seed  nonzero={ft09_nonzero}/10  W_a_norm={np.mean(ft09_norms):.3f}  delta_ema={np.mean(ft09_demas):.3f}")
    print(f"  {ft09_r}")

    print()
    print("=" * 70)
    print("STEP 951 RESULTS (vs 948: LS20 nonzero=1/10, mean=9.6):")

    if ls20_nonzero >= 3:
        verdict = f"SUCCESS — {ls20_nonzero}/10 seeds, mean subtraction breaks symmetry"
    elif ls20_nonzero <= 1:
        verdict = f"KILL — {ls20_nonzero}/10 seeds, no improvement over 948"
    else:
        verdict = f"MARGINAL — {ls20_nonzero}/10 seeds (need 3+ for success)"

    print(f"  LS20: L1={ls20_mean:.1f}/seed  {ls20_r}")
    print(f"  FT09: L1={ft09_mean:.1f}/seed  {ft09_r}")
    print(f"  VERDICT: {verdict}")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 951 DONE")
