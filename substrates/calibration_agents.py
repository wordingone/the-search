"""
calibration_agents.py — R3 calibration baseline substrates (Fix 4).

Leo directive 2026-03-23: Run judge on 4 known system types to calibrate
what R3 scores mean. These form Table 1 in the paper.

Agents:
  RandomAgent:     random action, all-U. R3_static=FAIL, R3_counterfactual=N/A.
  FixedPolicyAgent: argmax of fixed hash. R3_static=FAIL, counterfactual=N/A.
  TabularQLearning: Q-table from reward. R1=FAIL, R3_counterfactual=PASS.
  (674 is the fourth, already in step0674.py)

Usage:
  from substrates.calibration_agents import RandomAgent, FixedPolicyAgent, TabularQLearning
  judge = ConstitutionalJudge()
  for cls in [RandomAgent, FixedPolicyAgent, TabularQLearning]:
      r = judge.audit(cls)
      print(cls.__name__, r['summary'])
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM


class RandomAgent(BaseSubstrate):
    """Baseline: purely random action selection.

    Expected judge scores:
    - R1: PASS (no external objectives used)
    - R2: FAIL (state never changes — G stays empty)
    - R3_static: FAIL (all elements are U)
    - R3_counterfactual: N/A (no state to restore that helps)
    - R3_dynamic: 0.0 (nothing changes)
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._t = 0

    def process(self, observation) -> int:
        self._t += 1
        return int(self._rng.randint(0, self._n_actions))

    def get_state(self) -> dict:
        return {"t": self._t}

    def set_state(self, state: dict) -> None:
        self._t = state["t"]
        # RandomAgent has no learnable state — set_state is a no-op for RNG state.
        # This means R3_counterfactual will show P_warm ≈ P_cold (no improvement).

    def frozen_elements(self) -> list:
        return [
            {"name": "random_action", "class": "U",
             "justification": "Random selection. Could be any policy. System doesn't choose."},
            {"name": "rng_seed", "class": "U",
             "justification": "Seed is set by caller. System doesn't choose."},
        ]

    def reset(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
        self._t = 0

    def on_level_transition(self) -> None:
        pass  # Random agent ignores transitions

    @property
    def n_actions(self) -> int:
        return self._n_actions


class FixedPolicyAgent(BaseSubstrate):
    """Baseline: deterministic hash-to-action mapping, never updated.

    Uses same LSH encoding as 674 but argmax of projection (not argmin of edges).
    Policy is fixed at init — no state updates during process().

    Expected judge scores:
    - R1: PASS (no external objectives)
    - R2: FAIL (state never changes)
    - R3_static: FAIL (all U — static hash, static argmax)
    - R3_counterfactual: N/A (no learning)
    - R3_dynamic: 0.0 (H_nav never changes)
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        from substrates.step0674 import K_NAV
        self._n_actions = n_actions
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self._t = 0

    def process(self, observation) -> int:
        self._t += 1
        x = _enc_frame(observation)
        # Fixed policy: action = sum of LSH bits mod n_actions
        bits = (self.H_nav @ x > 0).astype(int)
        return int(bits.sum() % self._n_actions)

    def get_state(self) -> dict:
        return {"t": self._t, "H_nav": self.H_nav.copy()}

    def set_state(self, state: dict) -> None:
        self._t = state["t"]
        if "H_nav" in state:
            self.H_nav = state["H_nav"].copy()
        # Fixed policy doesn't learn, so set_state restores exactly.
        # But since nothing changes, P_warm = P_cold = same fixed policy.

    def frozen_elements(self) -> list:
        return [
            {"name": "H_nav_planes", "class": "U",
             "justification": "Random LSH planes set at init. System never modifies them."},
            {"name": "bit_sum_policy", "class": "U",
             "justification": "Action = bit_sum % n_actions. Arbitrary deterministic function. System doesn't choose."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection for state encoding. Removing destroys any structure."},
        ]

    def reset(self, seed: int) -> None:
        from substrates.step0674 import K_NAV
        rng = np.random.RandomState(seed * 1000)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self._t = 0

    def on_level_transition(self) -> None:
        pass

    @property
    def n_actions(self) -> int:
        return self._n_actions


class TabularQLearning(BaseSubstrate):
    """Baseline: tabular Q-learning. Uses reward signal (R1=FAIL).

    epsilon-greedy Q-learning with state discretization via LSH.
    Uses reward to update Q-table (R1 violation — external objective).

    Expected judge scores:
    - R1: FAIL (uses 'reward' / 'loss' signals)
    - R3_static: M (Q-table is an M element — changes via Bellman update)
    - R3_counterfactual: PASS (Q-table learned > initial)
    - R3_dynamic: 1.0 (Q-table changes continuously)

    NOTE: This substrate REQUIRES reward to function. In game environments,
    it uses the reward parameter of env.step(). In the judge's counterfactual
    test (no reward available), it will update with reward=0 (no learning).
    """

    ALPHA = 0.1   # learning rate
    GAMMA = 0.99  # discount factor
    EPSILON = 0.1 # exploration rate

    def __init__(self, n_actions: int = 4, seed: int = 0):
        from substrates.step0674 import K_NAV
        self._n_actions = n_actions
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self._rng = np.random.RandomState(seed + 1)
        self.Q = {}          # M: state -> [q_values per action]
        self._last_state = None
        self._last_action = None
        self._last_reward = 0.0
        self._t = 0

    def _discretize(self, observation) -> int:
        x = _enc_frame(observation)
        return int(np.packbits(
            (self.H_nav @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def process(self, observation) -> int:
        """Process observation with optional reward injection.

        The judge calls process(obs) without reward. For R1 check,
        the AST scanner finds 'reward' in the variable names below.
        """
        state = self._discretize(observation)
        self._t += 1

        # Update Q from previous step (Bellman update)
        if self._last_state is not None:
            s_prev = self._last_state
            a_prev = self._last_action
            reward = self._last_reward   # R1 violation: uses external reward
            if s_prev not in self.Q:
                self.Q[s_prev] = np.zeros(self._n_actions)
            if state not in self.Q:
                self.Q[state] = np.zeros(self._n_actions)
            td_target = reward + self.GAMMA * np.max(self.Q[state])
            td_error = td_target - self.Q[s_prev][a_prev]
            self.Q[s_prev][a_prev] += self.ALPHA * td_error

        # Epsilon-greedy action selection
        if self._rng.random() < self.EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            if state not in self.Q:
                self.Q[state] = np.zeros(self._n_actions)
            action = int(np.argmax(self.Q[state]))

        self._last_state = state
        self._last_action = action
        return action

    def inject_reward(self, reward: float) -> None:
        """Inject reward from environment (called externally after env.step)."""
        self._last_reward = reward

    def get_state(self) -> dict:
        import copy
        return {
            "Q_size": len(self.Q),
            "t": self._t,
            "H_nav": self.H_nav.copy(),
            "Q": copy.deepcopy(self.Q),
            "_last_state": self._last_state,
            "_last_action": self._last_action,
            "_last_reward": self._last_reward,
        }

    def set_state(self, state: dict) -> None:
        import copy
        self._t = state["t"]
        self.H_nav = state["H_nav"].copy()
        self.Q = copy.deepcopy(state["Q"])
        self._last_state = state["_last_state"]
        self._last_action = state["_last_action"]
        self._last_reward = state["_last_reward"]

    def frozen_elements(self) -> list:
        return [
            {"name": "H_nav_planes", "class": "U",
             "justification": "Random LSH planes. System doesn't choose direction or count."},
            {"name": "Q_table", "class": "M",
             "justification": "Q-values updated by Bellman equation from reward signal. System-driven."},
            {"name": "alpha_lr", "class": "U",
             "justification": "Learning rate α=0.1. System doesn't choose this value."},
            {"name": "gamma_discount", "class": "U",
             "justification": "Discount γ=0.99. System doesn't choose this value."},
            {"name": "epsilon_explore", "class": "U",
             "justification": "Exploration ε=0.1. System doesn't choose this value."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection for state encoding. Removing destroys Q-table structure."},
            {"name": "argmax_q", "class": "I",
             "justification": "Argmax Q selection. Removing destroys greedy exploitation."},
            {"name": "reward_signal", "class": "U",
             "justification": "External reward used for Bellman update. R1 VIOLATION. System chooses to use it."},
        ]

    def reset(self, seed: int) -> None:
        from substrates.step0674 import K_NAV
        rng = np.random.RandomState(seed * 1000)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self._rng = np.random.RandomState(seed * 1000 + 1)
        self.Q = {}
        self._last_state = None
        self._last_action = None
        self._last_reward = 0.0
        self._t = 0

    def on_level_transition(self) -> None:
        self._last_state = None
        self._last_action = None
        self._last_reward = 0.0

    @property
    def n_actions(self) -> int:
        return self._n_actions
