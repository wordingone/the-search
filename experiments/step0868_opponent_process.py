"""
step0868_opponent_process.py -- Opponent process: reward most changed, penalize least changed.

R3 hypothesis: does explicitly PENALIZING the action with least change (opponent suppression)
improve navigation beyond pure argmax of most change?

Opponent process: instead of pure argmax(delta), use delta[i] - delta[j] for each i,
where j is the action with minimum delta. This creates a contrast signal.
Equivalent to: argmax(delta - min(delta)) = argmax(delta) (same result for argmax).

But for WEIGHTED selection (softmax): softmax(delta - min(delta)) gives sharper distribution.

Variants:
1. Argmax (standard 800b)
2. Softmax τ=0.5 (soft selection over delta)
3. Softmax τ=0.1 (sharp selection)
4. Opponent contrast: argmax(delta - min(delta)) = argmax(delta) (same as 1)
5. Inverted penalization: sample proportional to delta / (min(delta) + 1e-8) - 1

Metric: L1 cold, 25K steps, seeds 6-10.
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


class OpponentProcess868(BaseSubstrate):
    def __init__(self, n_actions=4, seed=0, mode="argmax"):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self._mode = mode
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

    def _select_action(self, delta):
        if self._mode == "argmax":
            return int(np.argmax(delta))
        elif self._mode == "softmax_05":
            tau = 0.5
            d = delta / (np.sum(delta) + 1e-8)
            exp_d = np.exp(d / tau); exp_d /= exp_d.sum()
            return int(self._rng.choice(self._n_actions, p=exp_d))
        elif self._mode == "softmax_01":
            tau = 0.1
            d = delta / (np.sum(delta) + 1e-8)
            exp_d = np.exp(d / tau); exp_d /= exp_d.sum()
            return int(self._rng.choice(self._n_actions, p=exp_d))
        return int(np.argmax(delta))

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
            action = self._select_action(self.delta_per_action)
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


def run_cold(mode, seeds, n_steps):
    results = []
    for ts in seeds:
        sub = OpponentProcess868(n_actions=N_ACTIONS, seed=0, mode=mode)
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
print("STEP 868 — OPPONENT PROCESS (softmax vs argmax selection)")
print("=" * 70)
print(f"Metric: L1 cold, {TEST_STEPS} steps, seeds {TEST_SEEDS}")

t0 = time.time()

for mode in ["argmax", "softmax_05", "softmax_01"]:
    results = run_cold(mode, TEST_SEEDS, TEST_STEPS)
    mean_c = np.mean(results)
    print(f"  {mode}: {mean_c:.1f}/seed  ({results})")

print(f"\nBaseline step800b (argmax): 327/seed")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 868 DONE")
