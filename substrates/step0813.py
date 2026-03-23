"""
step0813.py -- AntiConvergenceSweep813: epsilon sweep over step800b architecture.

R3 hypothesis: step800b (EMA per-action change, 80% argmax + 20% random) achieves
6-10x random baseline. Does epsilon value matter? Can Boltzmann sampling do better?

Tests 4 epsilon variants and 1 Boltzmann variant:
- Epsilon 0.05 (5% random, 95% argmax) — near-pure exploitation
- Epsilon 0.10 (10% random, 90% argmax)
- Epsilon 0.20 (20% random, 80% argmax) — step800b default
- Epsilon 0.50 (50% random, 50% argmax) — balanced
- Boltzmann (softmax over delta) — smooth exploration

D(s) = {delta_per_action, running_mean}. L(s) = empty.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
EMA_ALPHA = 0.1
INIT_DELTA = 1.0


class EpsilonActionChange813(BaseSubstrate):
    """Configurable epsilon version of step800b. Epsilon passed at init."""

    def __init__(self, n_actions: int = 4, seed: int = 0, epsilon: float = 0.20):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self.running_mean = np.zeros(DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None
        self._rng = np.random.RandomState(seed)

    def _encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def process(self, observation) -> int:
        x = self._encode(observation)
        self._last_enc = x
        if self._prev_enc is not None and self._prev_action is not None:
            change = float(np.sqrt(np.sum((x - self._prev_enc) ** 2)))
            a = self._prev_action
            self.delta_per_action[a] = (
                (1 - EMA_ALPHA) * self.delta_per_action[a] + EMA_ALPHA * change
            )
        if self._rng.random() < self._epsilon:
            action = self._rng.randint(0, self._n_actions)
        else:
            action = int(np.argmax(self.delta_per_action))
        self._prev_enc = x.copy()
        self._prev_action = action
        return action

    @property
    def n_actions(self): return self._n_actions

    def get_state(self):
        return {"delta_per_action": self.delta_per_action.copy(),
                "running_mean": self.running_mean.copy(), "_n_obs": self._n_obs,
                "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
                "_prev_action": self._prev_action}

    def set_state(self, state):
        self.delta_per_action = state["delta_per_action"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self._prev_enc = state["_prev_enc"].copy() if state["_prev_enc"] is not None else None
        self._prev_action = state["_prev_action"]

    def reset(self, seed):
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def frozen_elements(self): return []


class BoltzmannActionChange813(BaseSubstrate):
    """Boltzmann (softmax) sampling over delta_per_action. Temperature tau."""

    def __init__(self, n_actions: int = 4, seed: int = 0, tau: float = 0.1):
        self._n_actions = n_actions
        self._seed = seed
        self._tau = tau
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self.running_mean = np.zeros(DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None
        self._rng = np.random.RandomState(seed)

    def _encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def process(self, observation) -> int:
        x = self._encode(observation)
        self._last_enc = x
        if self._prev_enc is not None and self._prev_action is not None:
            change = float(np.sqrt(np.sum((x - self._prev_enc) ** 2)))
            a = self._prev_action
            self.delta_per_action[a] = (
                (1 - EMA_ALPHA) * self.delta_per_action[a] + EMA_ALPHA * change
            )
        # Boltzmann: softmax over delta / tau
        logits = self.delta_per_action / self._tau
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(self._rng.choice(self._n_actions, p=probs))
        self._prev_enc = x.copy()
        self._prev_action = action
        return action

    @property
    def n_actions(self): return self._n_actions

    def get_state(self):
        return {"delta_per_action": self.delta_per_action.copy(),
                "running_mean": self.running_mean.copy(), "_n_obs": self._n_obs,
                "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
                "_prev_action": self._prev_action}

    def set_state(self, state):
        self.delta_per_action = state["delta_per_action"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self._prev_enc = state["_prev_enc"].copy() if state["_prev_enc"] is not None else None
        self._prev_action = state["_prev_action"]

    def reset(self, seed):
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def frozen_elements(self): return []
