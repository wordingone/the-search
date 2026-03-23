"""
baselines.py — Honest baseline substrates for PRISM comparison.

All implement BaseSubstrate. All plug into ChainRunner directly.
Not strawmen — standard implementations from the literature.

Honest baselines (Leo mail 2691, 2026-03-23):
  RandomBaseline       — random action (the absolute floor)
  CountBaseline        — global action counts, pick least-taken (non-graph: global)
  RNDBaseline          — Random Network Distillation (Burda et al. 2018)
  ICMBaseline          — Intrinsic Curiosity Module (Pathak et al. 2017, simplified)
  GraphArgminBaseline  — per-state argmin (the banned mechanism, for baseline only)
"""
import numpy as np
import hashlib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

ENC_DIM = 256


class RandomBaseline(BaseSubstrate):
    """Random action every step. The absolute floor.

    Passes R1 (no objective). Fails R2 (no adaptation). Fails R3 (no self-mod).
    Use to check that any mechanism beats chance.
    """

    def __init__(self, seed: int = 0):
        self._n_actions = 4
        self._rng = np.random.RandomState(seed)

    def set_game(self, n_actions: int):
        self._n_actions = n_actions

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def process(self, obs) -> int:
        return int(self._rng.randint(0, self._n_actions))

    def reset(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    def get_state(self) -> dict:
        return {"n_actions": self._n_actions, "rng_state": self._rng.get_state()}

    def set_state(self, state: dict) -> None:
        self._n_actions = state["n_actions"]
        self._rng.set_state(state["rng_state"])

    def frozen_elements(self) -> list:
        return [{"name": "rng", "class": "U",
                 "justification": "Random — no mechanism. U element by design."}]


class CountBaseline(BaseSubstrate):
    """Global action counts — pick least-taken action.

    Non-graph: counts are global, not per-state. Allowed post graph-ban.
    Implements a simple exploration pressure that balances action frequencies.
    Reference: count-based exploration without per-state conditioning.
    """

    def __init__(self, seed: int = 0):
        self._n_actions = 4
        self._rng = np.random.RandomState(seed)
        self._counts = np.zeros(4, dtype=np.int64)

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._counts = np.zeros(n_actions, dtype=np.int64)

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def process(self, obs) -> int:
        # Among least-taken actions, pick randomly (break ties)
        min_count = self._counts.min()
        candidates = np.where(self._counts == min_count)[0]
        action = int(self._rng.choice(candidates))
        self._counts[action] += 1
        return action

    def reset(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
        self._counts = np.zeros(self._n_actions, dtype=np.int64)

    def get_state(self) -> dict:
        return {"n_actions": self._n_actions, "counts": self._counts.copy(),
                "rng_state": self._rng.get_state()}

    def set_state(self, state: dict) -> None:
        self._n_actions = state["n_actions"]
        self._counts = state["counts"].copy()
        self._rng.set_state(state["rng_state"])

    def frozen_elements(self) -> list:
        return [
            {"name": "counts", "class": "M",
             "justification": "Action counts modified by each step."},
            {"name": "argmin_selector", "class": "U",
             "justification": "Could be softmax or epsilon-greedy — not chosen by system."},
        ]


class RNDBaseline(BaseSubstrate):
    """Random Network Distillation (Burda et al. 2018).

    Fixed random target network + trained predictor. Prediction error = novelty.
    Per-action novelty EMA — pick action with highest estimated novelty.

    Reference: Burda et al. (2018). Exploration by Random Network Distillation.
    ICLR 2019. arXiv:1810.12894.
    """

    ETA = 0.01        # predictor learning rate
    ALPHA_EMA = 0.1   # novelty EMA decay

    def __init__(self, seed: int = 0):
        self._n_actions = 4
        self._rng = np.random.RandomState(seed)
        # Fixed random target: enc_dim -> enc_dim
        rng_init = np.random.RandomState(42)  # fixed: same target across seeds
        self._W_target = (rng_init.randn(ENC_DIM, ENC_DIM) * 0.1).astype(np.float32)
        self._W_pred = np.zeros((ENC_DIM, ENC_DIM), dtype=np.float32)
        self._novelty_per_action = np.ones(4, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._novelty_per_action = np.ones(n_actions, dtype=np.float32)

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def _encode(self, obs) -> np.ndarray:
        enc = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc
        return enc - self._running_mean

    def process(self, obs) -> int:
        enc = self._encode(obs)

        # RND: target embed (fixed) vs predictor embed (trained)
        target = self._W_target @ enc
        pred = self._W_pred @ enc
        error = target - pred
        novelty = float(np.dot(error, error))

        # Update predictor (gradient descent on prediction error)
        grad = -2 * np.outer(error, enc)
        self._W_pred -= self.ETA * grad

        # EMA novelty for last action — tracked globally, not per-state
        action = int(np.argmax(self._novelty_per_action))
        self._novelty_per_action[action] = (
            (1 - self.ALPHA_EMA) * self._novelty_per_action[action]
            + self.ALPHA_EMA * novelty
        )
        # Epsilon exploration
        if self._rng.random() < 0.1:
            action = int(self._rng.randint(0, self._n_actions))
        return action

    def reset(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
        self._W_pred = np.zeros((ENC_DIM, ENC_DIM), dtype=np.float32)
        self._novelty_per_action = np.ones(self._n_actions, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

    def get_state(self) -> dict:
        return {
            "n_actions": self._n_actions,
            "W_pred": self._W_pred.copy(),
            "novelty_per_action": self._novelty_per_action.copy(),
            "running_mean": self._running_mean.copy(),
            "n_obs": self._n_obs,
        }

    def set_state(self, state: dict) -> None:
        self._n_actions = state["n_actions"]
        self._W_pred = state["W_pred"].copy()
        self._novelty_per_action = state["novelty_per_action"].copy()
        self._running_mean = state["running_mean"].copy()
        self._n_obs = state["n_obs"]

    def frozen_elements(self) -> list:
        return [
            {"name": "W_target", "class": "I",
             "justification": "Fixed random target — removing destroys RND signal."},
            {"name": "W_pred", "class": "M",
             "justification": "Predictor updated each step via gradient descent."},
            {"name": "novelty_per_action", "class": "M",
             "justification": "EMA novelty estimate per action, modified each step."},
        ]


class ICMBaseline(BaseSubstrate):
    """Intrinsic Curiosity Module (Pathak et al. 2017, simplified).

    Forward model predicts next_enc from (enc, action_onehot).
    Curiosity = forward model prediction error.
    Per-action curiosity EMA — pick action with highest curiosity.

    Simplified: no inverse model (no self-supervised action prediction).
    R1-compatible: no external reward, no labels.

    Reference: Pathak et al. (2017). Curiosity-driven Exploration by Self-Supervised
    Prediction. ICML 2017. arXiv:1705.05363.
    """

    ETA_W = 0.01
    ALPHA_EMA = 0.1

    def __init__(self, seed: int = 0):
        self._n_actions = 4
        self._rng = np.random.RandomState(seed)
        self._W_fwd = np.zeros((ENC_DIM, ENC_DIM + 4), dtype=np.float32)
        self._curiosity_per_action = np.ones(4, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._W_fwd = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self._curiosity_per_action = np.ones(n_actions, dtype=np.float32)
        self._prev_enc = None
        self._prev_action = None

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def _encode(self, obs) -> np.ndarray:
        enc = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc
        return enc - self._running_mean

    def _one_hot(self, a: int) -> np.ndarray:
        v = np.zeros(self._n_actions, dtype=np.float32)
        v[a] = 1.0
        return v

    def process(self, obs) -> int:
        enc = self._encode(obs)

        if self._prev_enc is not None and self._prev_action is not None:
            # Forward model prediction of current enc from (prev_enc, prev_action)
            inp = np.concatenate([self._prev_enc, self._one_hot(self._prev_action)])
            pred = self._W_fwd @ inp
            error = enc - pred
            curiosity = float(np.dot(error, error))

            # Clip gradient, update forward model
            en = float(np.linalg.norm(error))
            if en > 10.0:
                error = error * (10.0 / en)
            self._W_fwd -= self.ETA_W * np.outer(error, inp)

            # EMA curiosity for the action that led here
            a = self._prev_action
            self._curiosity_per_action[a] = (
                (1 - self.ALPHA_EMA) * self._curiosity_per_action[a]
                + self.ALPHA_EMA * curiosity
            )

        # Pick most curious action
        action = int(np.argmax(self._curiosity_per_action))
        if self._rng.random() < 0.1:
            action = int(self._rng.randint(0, self._n_actions))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self) -> None:
        self._prev_enc = None
        self._prev_action = None

    def reset(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
        self._W_fwd = np.zeros((ENC_DIM, ENC_DIM + self._n_actions), dtype=np.float32)
        self._curiosity_per_action = np.ones(self._n_actions, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None

    def get_state(self) -> dict:
        return {
            "n_actions": self._n_actions,
            "W_fwd": self._W_fwd.copy(),
            "curiosity_per_action": self._curiosity_per_action.copy(),
            "running_mean": self._running_mean.copy(),
            "n_obs": self._n_obs,
        }

    def set_state(self, state: dict) -> None:
        self._n_actions = state["n_actions"]
        self._W_fwd = state["W_fwd"].copy()
        self._curiosity_per_action = state["curiosity_per_action"].copy()
        self._running_mean = state["running_mean"].copy()
        self._n_obs = state["n_obs"]
        self._prev_enc = None
        self._prev_action = None

    def frozen_elements(self) -> list:
        return [
            {"name": "W_fwd", "class": "M",
             "justification": "Forward model updated each step."},
            {"name": "curiosity_per_action", "class": "M",
             "justification": "EMA curiosity per action, modified each step."},
        ]


class GraphArgminBaseline(BaseSubstrate):
    """Graph + argmin over visit counts. The pre-ban mechanism.

    Per-state visit count: G[(state_hash, action)] -> count.
    From current state, pick action with fewest visits.

    BANNED for research (graph ban, post Step 777).
    Run as BASELINE ONLY — shows ceiling of the banned approach.
    """

    MAX_STATES = 100_000  # prevent unbounded memory growth

    def __init__(self, seed: int = 0):
        self._n_actions = 4
        self._rng = np.random.RandomState(seed)
        self._G: dict = {}  # (state_hash, action) -> count
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._G = {}

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def _encode(self, obs) -> np.ndarray:
        enc = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc
        return enc - self._running_mean

    def _state_hash(self, enc: np.ndarray) -> str:
        scaled = np.clip((enc - enc.min()) / (enc.max() - enc.min() + 1e-8) * 255,
                         0, 255).astype(np.uint8)
        return hashlib.md5(scaled.tobytes()).hexdigest()[:12]

    def process(self, obs) -> int:
        enc = self._encode(obs)
        h = self._state_hash(enc)

        # Visit counts for this state
        counts = np.array([self._G.get((h, a), 0) for a in range(self._n_actions)],
                          dtype=np.int64)
        action = int(np.argmin(counts))

        # Update count (only if memory not full)
        key = (h, action)
        if key in self._G or len(self._G) < self.MAX_STATES:
            self._G[key] = self._G.get(key, 0) + 1

        return action

    def on_level_transition(self) -> None:
        pass  # graph persists across levels (same game)

    def reset(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
        self._G = {}
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

    def get_state(self) -> dict:
        return {
            "n_actions": self._n_actions,
            "G_size": len(self._G),
            "running_mean": self._running_mean.copy(),
            "n_obs": self._n_obs,
        }

    def set_state(self, state: dict) -> None:
        self._n_actions = state["n_actions"]
        self._running_mean = state["running_mean"].copy()
        self._n_obs = state["n_obs"]
        # Note: G not restored (too large to serialize) — counterfactual not supported

    def frozen_elements(self) -> list:
        return [
            {"name": "G", "class": "M",
             "justification": "Per-(state,action) visit count graph — graph-banned in research."},
            {"name": "argmin_selector", "class": "U",
             "justification": "Arbitrary choice — not chosen by system dynamics."},
        ]


# Convenience: all baselines in run order for full comparison
ALL_BASELINES = [
    ("Random", RandomBaseline),
    ("Count", CountBaseline),
    ("RND", RNDBaseline),
    ("ICM", ICMBaseline),
    ("Graph+argmin", GraphArgminBaseline),
]
