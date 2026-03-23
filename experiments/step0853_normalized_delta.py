"""
step0853_normalized_delta.py -- Normalized delta: scale-invariant change tracking.

R3 hypothesis: does normalizing change magnitude by observation L2 norm improve
action discrimination? Raw change (||enc_t - enc_t-1||^2) conflates absolute
change with observation scale. Normalized change = change / ||enc_t||^2 is a
relative measure of "how much did this action change state (relative to current state)?"

Protocol: test 3 variants:
1. Raw delta (current 800b: ||enc - prev_enc||^2)
2. Normalized delta (change / ||enc||^2 + eps)
3. Cosine delta (1 - cosine_sim(enc, prev_enc))

Metric: L1 completions cold, 25K steps, seeds 6-10. LS20.
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


class DeltaVariant800b(BaseSubstrate):
    def __init__(self, n_actions=4, seed=0, delta_type="raw"):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self._delta_type = delta_type
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

    def _compute_change(self, enc, prev_enc):
        if self._delta_type == "raw":
            return float(np.sum((enc - prev_enc) ** 2))
        elif self._delta_type == "normalized":
            raw = float(np.sum((enc - prev_enc) ** 2))
            norm = float(np.sum(enc ** 2)) + 1e-8
            return raw / norm
        elif self._delta_type == "cosine":
            n1 = float(np.linalg.norm(enc)) + 1e-8
            n2 = float(np.linalg.norm(prev_enc)) + 1e-8
            cos = float(np.dot(enc, prev_enc)) / (n1 * n2)
            return 1.0 - cos
        return 0.0

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        if self._prev_enc is not None and self._prev_action is not None:
            change = self._compute_change(enc, self._prev_enc)
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

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_cold(delta_type, seeds, n_steps):
    results = []
    for ts in seeds:
        sub = DeltaVariant800b(n_actions=N_ACTIONS, seed=0, delta_type=delta_type)
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
print("STEP 853 — NORMALIZED DELTA (raw vs normalized vs cosine)")
print("=" * 70)
print(f"Metric: L1 completions cold, {TEST_STEPS} steps, seeds {TEST_SEEDS}")

t0 = time.time()

for delta_type in ["raw", "normalized", "cosine"]:
    results = run_cold(delta_type, TEST_SEEDS, TEST_STEPS)
    mean_c = np.mean(results)
    print(f"  {delta_type}: {mean_c:.1f}/seed  ({results})")

print()
print(f"Baseline: step800b (raw delta): 327/seed")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 853 DONE")
