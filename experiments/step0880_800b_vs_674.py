"""
step0880_800b_vs_674.py -- 800b vs 674-encoding baselines head-to-head on LS20.

R3 hypothesis (verification): does 800b's EMA per-action tracking matter, or does
any change-seeking policy using 674 encoding beat random?

Conditions compared:
1. step800b (EMA per-action change tracking, 80/20 epsilon) — our best mechanism
2. 674-argmax-change (argmax of |enc_t - enc_t-1| by action, no EMA, no epsilon)
3. 674-random (pure random action with 674 encoding unused)
4. 674-max-change-noEMA (argmax of running total change per action, no epsilon)

Metric: L1 completions over 25K steps (cold only). Same seeds, same budget.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0800b import EpsilonActionChange800b
from substrates.step0674 import _enc_frame

TEST_SEEDS = [6, 7, 8, 9, 10]
TEST_STEPS = 25_000
N_ACTIONS = 4
INIT_DELTA = 1.0


class ArgmaxChangeInstant(BaseSubstrate):
    """Argmax of last-step observation change per action (no EMA). No epsilon."""

    def __init__(self, n_actions=4, seed=0):
        self._n_actions = n_actions
        rng = np.random.RandomState(seed)
        self._change_per_action = np.zeros(n_actions, np.float32)
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None

    def process(self, observation):
        enc = _enc_frame(np.asarray(observation, dtype=np.float32))
        self._last_enc = enc
        if self._prev_enc is not None and self._prev_action is not None:
            change = float(np.sum((enc - self._prev_enc) ** 2))
            self._change_per_action[self._prev_action] = change
        best_a = int(np.argmax(self._change_per_action))
        self._prev_enc = enc.copy(); self._prev_action = best_a
        return best_a

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._prev_enc = None; self._prev_action = None; self._last_enc = None
        self._change_per_action = np.zeros(self._n_actions, np.float32)

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def get_state(self): return {"change": self._change_per_action.copy()}
    def set_state(self, s): self._change_per_action = s["change"].copy()
    def frozen_elements(self): return []


class CumulativeArgmaxChange(BaseSubstrate):
    """Argmax of cumulative (sum) observation change per action. No epsilon."""

    def __init__(self, n_actions=4, seed=0):
        self._n_actions = n_actions
        self._cum_change = np.zeros(n_actions, np.float32)
        self._count = np.zeros(n_actions, np.int32)
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None

    def process(self, observation):
        enc = _enc_frame(np.asarray(observation, dtype=np.float32))
        self._last_enc = enc
        if self._prev_enc is not None and self._prev_action is not None:
            change = float(np.sum((enc - self._prev_enc) ** 2))
            self._cum_change[self._prev_action] += change
            self._count[self._prev_action] += 1
        # Use mean change per action (initialize to INIT_DELTA to avoid all-zeros)
        mean_change = np.where(self._count > 0,
                                self._cum_change / self._count,
                                INIT_DELTA)
        best_a = int(np.argmax(mean_change))
        self._prev_enc = enc.copy(); self._prev_action = best_a
        return best_a

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._prev_enc = None; self._prev_action = None; self._last_enc = None
        self._cum_change = np.zeros(self._n_actions, np.float32)
        self._count = np.zeros(self._n_actions, np.int32)

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def get_state(self): return {"cum": self._cum_change.copy(), "cnt": self._count.copy()}
    def set_state(self, s): self._cum_change = s["cum"].copy(); self._count = s["cnt"].copy()
    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_cold(substrate_cls, env_seeds, n_steps):
    accs = []
    for ts in env_seeds:
        sub = substrate_cls(n_actions=N_ACTIONS, seed=0)
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
        accs.append(completions)
    return accs


print("=" * 70)
print("STEP 880 — 800b vs 674-ENCODING BASELINES (LS20, cold)")
print("=" * 70)
print("Metric: L1 completions over 25K steps, mean/seed")
print(f"Random baseline: 36.4/seed\n")

t0 = time.time()

conditions = [
    ("800b (EMA+epsilon)", EpsilonActionChange800b),
    ("ArgmaxInstant (last-step, no EMA)", ArgmaxChangeInstant),
    ("CumulativeArgmax (mean, no epsilon)", CumulativeArgmaxChange),
]

for label, cls in conditions:
    results = run_cold(cls, TEST_SEEDS, TEST_STEPS)
    mean_c = np.mean(results)
    print(f"  {label}: {mean_c:.1f}/seed  (per seed: {results})")

print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 880 DONE")
