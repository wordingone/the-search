"""
sub1175_defense_v75.py — Epsilon-greedy reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1175 --substrate experiments/sub1175_defense_v75.py

FAMILY: Epsilon-greedy reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v73 showed explore-then-exploit FAILS because 200 random
samples can't reliably discover responsive actions in a space of thousands.
Pure random works (3/5) because it explores CONTINUOUSLY for all 10K steps.
v30 works (2/5, high ARC) because it exploits efficiently.

v75 BLENDS both: epsilon-greedy reactive.
- With probability (1-epsilon): v30 reactive cycling over best known actions
- With probability epsilon: sample random action from FULL action space
- When random action produces change: ADD it to the reactive action set

This is the minimal fusion: random's continuous coverage + v30's exploitation.
Epsilon = 0.3 (30% exploration, 70% exploitation). The reactive set grows
over time as new responsive actions are discovered.

DIFFERENT from v73: v73 had a fixed explore window. v75 explores CONTINUOUSLY
throughout the entire run, discovering new responsive actions at any time.

ZERO learned parameters (defense: ℓ₁). Fixed epsilon, no learned weights.

KILL: L1 < 3/5 (worse than random).
SUCCESS: L1 ≥ 3/5 with ARC > random (0.002).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

EPSILON = 0.3              # exploration probability
RESPONSE_THRESH = 0.3      # min pixel change for responsive
MAX_RESPONSIVE = 30        # max tracked responsive actions


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class EpsilonGreedyReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._prev_action = 0

        # Responsive action tracking
        self._responsive_actions = {}  # action -> max_change
        self._action_set = []          # sorted responsive actions for cycling

        # Reactive cycling state
        self._current_idx = 0
        self._patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()
        # Start with keyboard actions as the initial action set
        n_kb = min(n_actions, N_KB)
        self._action_set = list(range(n_kb))

    def _add_responsive(self, action, change):
        """Add a newly discovered responsive action."""
        if action in self._responsive_actions:
            self._responsive_actions[action] = max(
                self._responsive_actions[action], change
            )
        else:
            self._responsive_actions[action] = change
            self.r3_updates += 1
            self.att_updates_total += 1

        # Rebuild action set: top responsive + keyboard fallback
        sorted_resp = sorted(
            self._responsive_actions.items(),
            key=lambda x: -x[1]
        )[:MAX_RESPONSIVE]
        self._action_set = [a for a, _ in sorted_resp]

        # Ensure at least keyboard actions in set
        if len(self._action_set) < N_KB:
            n_kb = min(self._n_actions_env, N_KB)
            for a in range(n_kb):
                if a not in self._action_set:
                    self._action_set.append(a)

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_action = 0
            return 0

        # Check if previous action was responsive
        delta = float(np.sum(np.abs(enc - self._prev_enc)))
        if delta > RESPONSE_THRESH:
            self._add_responsive(self._prev_action, delta)

        dist = float(np.sum(np.abs(enc - self._enc_0)))

        # Epsilon-greedy: explore or exploit
        if self._rng.random() < EPSILON:
            # EXPLORE: random action from full space
            action = int(self._rng.randint(0, self._n_actions_env))
        else:
            # EXPLOIT: v30-style reactive cycling over action set
            if dist >= self._prev_dist:
                self._current_idx = (self._current_idx + 1) % len(self._action_set)
                self._patience = 0
            else:
                self._patience += 1
                if self._patience > 10:
                    self._patience = 0
                    self._current_idx = (self._current_idx + 1) % len(self._action_set)

            action = self._action_set[self._current_idx % len(self._action_set)]

        self._prev_dist = dist
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._current_idx = 0
        self._patience = 0
        # Keep responsive actions across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "epsilon": EPSILON,
    "response_thresh": RESPONSE_THRESH,
    "max_responsive": MAX_RESPONSIVE,
    "family": "epsilon-greedy reactive",
    "tag": "defense v75 (ℓ₁ epsilon-greedy: 30% random exploration of FULL action space + 70% v30 reactive cycling. Continuous discovery of responsive actions. Combines random's coverage with v30's exploitation.)",
}

SUBSTRATE_CLASS = EpsilonGreedyReactiveSubstrate
