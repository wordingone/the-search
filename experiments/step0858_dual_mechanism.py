"""
step0858_dual_mechanism.py -- Dual mechanism: 800b navigation + prediction contrast hybrid.

R3 hypothesis: does combining 800b's action selection (EMA delta) with PredictionContrast780's
forward model provide better navigation? The forward model predicts next state per action.
800b selects based on observed change. Combined: select action with max predicted change.

Protocol: hybrid substrate using 674 encoding, forward model W, AND EMA delta.
Variants:
1. 800b (EMA observed): argmax(EMA change)
2. Pred-only (predicted change): argmax(||W*[enc,a_oh]||^2 over all actions) - max predicted norm
3. Hybrid (average): argmax(EMA_change + pred_change_norm / pred_scale)

Metric: L1 cold, 25K steps, seeds 6-10. Does prediction guidance help?
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
ETA = 0.01  # forward model learning rate


class DualMechanism858(BaseSubstrate):
    def __init__(self, n_actions=4, seed=0, mode="800b"):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self._mode = mode
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self.W = np.zeros((256, 256 + n_actions), np.float32)  # forward model
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

    def _pred_change_per_action(self, enc):
        """Predicted change norm for each action."""
        scores = np.zeros(self._n_actions, np.float32)
        for a in range(self._n_actions):
            a_oh = np.zeros(self._n_actions, np.float32); a_oh[a] = 1.0
            pred = self.W @ np.concatenate([enc, a_oh])
            scores[a] = float(np.sum((pred - enc) ** 2))
        return scores

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc

        # Update EMA delta
        if self._prev_enc is not None and self._prev_action is not None:
            change = float(np.sum((enc - self._prev_enc) ** 2))
            a = self._prev_action
            self.delta_per_action[a] = (1 - ALPHA) * self.delta_per_action[a] + ALPHA * change
            # Update forward model
            a_oh = np.zeros(self._n_actions, np.float32); a_oh[a] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            pred_err = self.W @ inp - enc
            self.W -= ETA * np.outer(pred_err, inp)

        if self._rng.random() < EPSILON:
            action = self._rng.randint(0, self._n_actions)
        else:
            if self._mode == "800b":
                action = int(np.argmax(self.delta_per_action))
            elif self._mode == "pred_only":
                scores = self._pred_change_per_action(enc)
                action = int(np.argmax(scores))
            elif self._mode == "hybrid":
                scores = self._pred_change_per_action(enc)
                # Normalize each to [0,1] range
                d = self.delta_per_action
                d_n = (d - d.min()) / (d.max() - d.min() + 1e-8)
                s_n = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                combined = d_n + s_n
                action = int(np.argmax(combined))
            else:
                action = self._rng.randint(0, self._n_actions)

        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._prev_enc = None; self._prev_action = None; self._last_enc = None
        self.delta_per_action = np.full(self._n_actions, INIT_DELTA, np.float32)
        self.W = np.zeros((256, 256 + self._n_actions), np.float32)
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
        sub = DualMechanism858(n_actions=N_ACTIONS, seed=0, mode=mode)
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
print("STEP 858 — DUAL MECHANISM (800b + prediction contrast hybrid)")
print("=" * 70)
print(f"Metric: L1 cold, {TEST_STEPS} steps, seeds {TEST_SEEDS}")

t0 = time.time()

for mode in ["800b", "pred_only", "hybrid"]:
    results = run_cold(mode, TEST_SEEDS, TEST_STEPS)
    mean_c = np.mean(results)
    print(f"  {mode}: {mean_c:.1f}/seed  ({results})")

print()
print(f"Baseline: step800b (observed EMA only): 327/seed")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 858 DONE")
