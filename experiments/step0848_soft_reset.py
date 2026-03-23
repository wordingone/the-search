"""
step0848_soft_reset.py -- Soft-reset delta on level_transition vs hard reset.

R3 hypothesis: does soft-resetting delta_per_action (decay by factor 0.5)
preserve more useful state than hard reset (full reset to init)?

Protocol: 3 variants:
1. Hard reset: delta = init on level_transition (current 800b)
2. Soft reset: delta *= 0.5 on level_transition (preserves partial memory)
3. No reset: delta unchanged on level_transition (full persistence)

Metric: L1 completions (cold, 25K steps, seeds 6-10). LS20.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

TEST_SEEDS = [6, 7, 8, 9, 10]
TEST_STEPS = 25_000
N_ACTIONS = 4
INIT_DELTA = 1.0
ALPHA = 0.10
EPSILON = 0.20


class SoftReset800b(BaseSubstrate):
    def __init__(self, n_actions=4, seed=0, reset_factor=0.5):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self._reset_factor = reset_factor
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._running_mean = np.zeros(256, np.float32)
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self._running_mean = (1 - alpha) * self._running_mean + alpha * enc_raw
        return enc_raw - self._running_mean

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        if self._prev_enc is not None and self._prev_action is not None:
            change = float(np.sum((enc - self._prev_enc) ** 2))
            a = self._prev_action
            self.delta_per_action[a] = (1 - ALPHA) * self.delta_per_action[a] + ALPHA * change
        if self._rng.random() < EPSILON:
            action = self._rng.randint(0, self._n_actions)
        else:
            action = int(np.argmax(self.delta_per_action))
        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._prev_enc = None; self._prev_action = None; self._last_enc = None
        self.delta_per_action = np.full(self._n_actions, INIT_DELTA, np.float32)
        self._running_mean = np.zeros(256, np.float32)
        self._n_obs = 0

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None
        if self._reset_factor < 1.0:
            self.delta_per_action *= self._reset_factor
        # If reset_factor = 1.0, no reset (full persistence)

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_cold(reset_factor, seeds, n_steps):
    results = []
    for ts in seeds:
        sub = SoftReset800b(n_actions=N_ACTIONS, seed=0, reset_factor=reset_factor)
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
        results.append(completions)
    return results


print("=" * 70)
print("STEP 848 — SOFT-RESET DELTA (hard vs soft vs no reset)")
print("=" * 70)
print(f"Metric: L1 completions cold, {TEST_STEPS} steps, seeds {TEST_SEEDS}")

t0 = time.time()

for rf, label in [(0.0, "hard reset (delta→init)"), (0.5, "soft reset (delta×0.5)"), (1.0, "no reset (full persist)")]:
    results = run_cold(rf, TEST_SEEDS, TEST_STEPS)
    mean_c = np.mean(results)
    print(f"  {label}: {mean_c:.1f}/seed  ({results})")

print()
print(f"step800b standard: 327/seed (uses implicit no-delta-reset but resets prev_enc)")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 848 DONE")
