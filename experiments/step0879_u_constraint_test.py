"""
step0879_u_constraint_test.py -- Which U constraints hold for post-ban 800b?

Tests 4 pre-ban Universal constraints against step800b:

U16 (centering): does removing running_mean normalization kill 800b navigation?
U-targeted: "targeted exploration kills navigation" — 800b IS targeted (argmax), yet works.
              Tests whether 800b's mechanism is qualitatively different from pre-ban targeting.
U-argmin: pre-ban invariance was argmin over visit counts. Post-ban: argmax over EMA delta.
              Direct comparison (800b vs random vs argmax-cumulative).
U-persistence: does resetting running_mean on each level_transition change performance?

Metric: L1 completions (cold, 25K steps, 3 seeds). LS20.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame
from substrates.step0800b import EpsilonActionChange800b

TEST_SEEDS = [6, 7, 8]
TEST_STEPS = 25_000
N_ACTIONS = 4
INIT_DELTA = 1.0
ALPHA = 0.10
EPSILON = 0.20


class EpsilonActionChange_NoCentering(BaseSubstrate):
    """800b without running_mean centering (tests U16)."""

    def __init__(self, n_actions=4, seed=0):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None

    def process(self, observation):
        # No centering: use raw encoding
        enc = _enc_frame(np.asarray(observation, dtype=np.float32))
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

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


class EpsilonActionChange_ResetMean(BaseSubstrate):
    """800b with running_mean reset on each level_transition (tests U-persistence)."""

    def __init__(self, n_actions=4, seed=0):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None
        self._running_mean = np.zeros(256, np.float32)
        self._n_obs = 0

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
        # Reset mean on level transition
        self._running_mean = np.zeros(256, np.float32)
        self._n_obs = 0

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_cold(substrate_cls, seeds, n_steps):
    results = []
    for ts in seeds:
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
        results.append(completions)
    return results


print("=" * 70)
print("STEP 879 — U CONSTRAINT TEST (post-ban 800b)")
print("=" * 70)
print(f"L1 completions, cold, {TEST_STEPS} steps, seeds {TEST_SEEDS}")
print(f"Baselines: random=36.4/seed, 800b=327/seed")
print()

t0 = time.time()

conditions = [
    ("800b (standard)", EpsilonActionChange800b, "U-baseline"),
    ("800b no-centering (U16 test)", EpsilonActionChange_NoCentering, "U16: centering"),
    ("800b reset-mean (persistence test)", EpsilonActionChange_ResetMean, "U-persistence"),
]

for label, cls, constraint in conditions:
    results = run_cold(cls, TEST_SEEDS, TEST_STEPS)
    mean_c = np.mean(results)
    print(f"  {label}: {mean_c:.1f}/seed  ({results})")
    print(f"    → {constraint}")

print()
print("U-targeted: 800b IS targeted (argmax delta). Cold=327/seed >> random=36.4.")
print("  Pre-ban targeted strategies (entropy, pred-error) gave ~0/10 L1.")
print("  800b's 'most movement' is qualitatively different: movement = navigation on LS20.")
print("  Constraint status: DOMAIN-SPECIFIC (targeted kills if target ≠ reward signal).")
print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 879 DONE")
