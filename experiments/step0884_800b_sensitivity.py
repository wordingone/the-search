"""
step0884_800b_sensitivity.py -- Sensitivity analysis on step800b hyperparameters.

R3 hypothesis: does 800b's cold navigation performance depend critically on
specific hyperparameter choices, or is it robust across a range?

Sweeps:
1. EMA alpha: [0.05, 0.10, 0.20, 0.50] (current: 0.10)
2. Epsilon (random fraction): [0.10, 0.20, 0.30, 0.50] (current: 0.20)
3. Init delta: [0.01, 0.10, 1.0, 10.0] (current: 1.0)

Metric: L1 completions (cold, 3 seeds). 10K steps (fast sweep).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

TEST_SEEDS = [6, 7, 8]  # 3 seeds for speed
TEST_STEPS = 10_000
N_ACTIONS = 4


class Configurable800b(BaseSubstrate):
    """Configurable version of step800b."""

    def __init__(self, n_actions=4, seed=0, alpha=0.10, epsilon=0.20, init_delta=1.0):
        self._n_actions = n_actions
        self._alpha = alpha
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        self.delta_per_action = np.full(n_actions, init_delta, dtype=np.float32)
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None

    def process(self, observation):
        enc = _enc_frame(np.asarray(observation, dtype=np.float32))
        self._last_enc = enc
        if self._prev_enc is not None and self._prev_action is not None:
            change = float(np.sum((enc - self._prev_enc) ** 2))
            a = self._prev_action
            self.delta_per_action[a] = (1 - self._alpha) * self.delta_per_action[a] + self._alpha * change
        if self._rng.random() < self._epsilon:
            action = self._rng.randint(0, self._n_actions)
        else:
            action = int(np.argmax(self.delta_per_action))
        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def get_state(self):
        return {"delta": self.delta_per_action.copy()}

    def set_state(self, s):
        self.delta_per_action = s["delta"].copy()

    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_cold(alpha, epsilon, init_delta, seeds, n_steps):
    totals = []
    for ts in seeds:
        sub = Configurable800b(n_actions=N_ACTIONS, seed=0,
                               alpha=alpha, epsilon=epsilon, init_delta=init_delta)
        sub.reset(0)
        env = make_game(); obs = env.reset(seed=ts * 1000)
        completions = 0; current_level = 0; step = 0
        while step < n_steps:
            if obs is None:
                obs = env.reset(seed=ts * 1000); current_level = 0
                sub.on_level_transition(); continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
            obs, _, done, info = env.step(action); step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > current_level:
                completions += (cl - current_level); current_level = cl
                sub.on_level_transition()
            if done:
                obs = env.reset(seed=ts * 1000); current_level = 0
                sub.on_level_transition()
        totals.append(completions)
    return float(np.mean(totals))


print("=" * 70)
print("STEP 884 — 800b SENSITIVITY ANALYSIS")
print("=" * 70)
print(f"Metric: mean L1/seed over {TEST_STEPS} steps (seeds {TEST_SEEDS})")
print(f"Baseline (default: alpha=0.1, eps=0.2, init=1.0) shown as [*]")
print()

t0 = time.time()

DEFAULT = {"alpha": 0.10, "epsilon": 0.20, "init_delta": 1.0}

# EMA alpha sweep
print("--- Alpha sweep (epsilon=0.20, init=1.0) ---")
for alpha in [0.05, 0.10, 0.20, 0.50]:
    mean_c = run_cold(alpha, DEFAULT["epsilon"], DEFAULT["init_delta"], TEST_SEEDS, TEST_STEPS)
    marker = " [*]" if alpha == DEFAULT["alpha"] else ""
    print(f"  alpha={alpha:.2f}: {mean_c:.1f}/seed{marker}")

# Epsilon sweep
print("\n--- Epsilon sweep (alpha=0.10, init=1.0) ---")
for epsilon in [0.10, 0.20, 0.30, 0.50]:
    mean_c = run_cold(DEFAULT["alpha"], epsilon, DEFAULT["init_delta"], TEST_SEEDS, TEST_STEPS)
    marker = " [*]" if epsilon == DEFAULT["epsilon"] else ""
    print(f"  epsilon={epsilon:.2f}: {mean_c:.1f}/seed{marker}")

# Init delta sweep
print("\n--- Init delta sweep (alpha=0.10, epsilon=0.20) ---")
for init_d in [0.01, 0.10, 1.0, 10.0]:
    mean_c = run_cold(DEFAULT["alpha"], DEFAULT["epsilon"], init_d, TEST_SEEDS, TEST_STEPS)
    marker = " [*]" if init_d == DEFAULT["init_delta"] else ""
    print(f"  init_delta={init_d:.2f}: {mean_c:.1f}/seed{marker}")

print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 884 DONE")
