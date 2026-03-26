"""
sub1098_defense_v27.py — Cross-level reactive switching (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1098 --substrate experiments/sub1098_defense_v27.py

FAMILY: Reactive action switching with cross-level transfer (NEW)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v21's reactive switching solves L1 on ~40% of draws but
NEVER solves L2+ because it resets all state at level transitions. Action
meanings are consistent across levels (action 3 = "move right" in L1 AND L2).
Persisting action success counts across levels lets the substrate try the
previously-successful action FIRST in L2, reducing exploration waste.

ONE STRUCTURAL CHANGE FROM v21: persist action success counts across levels.
- _action_success[a] counts how many times action a produced progress
- On level transition: reset within-level state but NOT action success counts
- Action selection priority: try actions in order of decreasing success count
  instead of round-robin from 0

Processing rules are FIXED (distance-to-initial, patience, switching).
Only the ACTION ORDERING changes based on accumulated counts.
This is ℓ₁: fixed information processing + accumulated state.

WHY THIS IS DIFFERENT FROM PROSECUTION:
- Prosecution: W_fwd (256×263 learned matrix) modifies HOW predictions are made
- Defense: fixed rules, accumulated counts only change STARTING ORDER
- No matrix multiplication, no outer product updates, no alpha

WHY THIS TARGETS THE WIN CONDITION:
- Win = 100% ALL LEVELS. L2+ requires level transfer.
- v21 resets everything at level boundary → L2 starts from scratch
- v27 carries action preferences → L2 starts with best action from L1

KILL: L1 performance worse than v21 on same draw type.
SUCCESS: ANY L2+ solve (would be first autonomous L2 in 1050+ experiments).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 8
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS  # 64
N_KB = 7
EXPLORE_STEPS = 50
MAX_PATIENCE = 20


def _obs_to_enc(obs):
    """avgpool8: 64x64 → 8x8 = 64D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class CrossLevelReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        # Cross-level state: NEVER reset within a game
        self._action_success = np.zeros(N_KB, dtype=np.float32)
        self._init_level_state()

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def _init_level_state(self):
        """Reset within-level state only. Preserve cross-level action memory."""
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        self._action_order = None  # computed after explore phase

    def _compute_action_order(self):
        """Order actions by decreasing success count. Ties broken by index."""
        order = np.argsort(-self._action_success[:self._n_actions])
        return list(order.astype(int))

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        # Reset EVERYTHING for a new game (including cross-level counts)
        self._action_success = np.zeros(N_KB, dtype=np.float32)
        self._init_level_state()

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def _next_action_in_order(self):
        """Get next action from priority order instead of round-robin."""
        if self._action_order is None or len(self._action_order) == 0:
            return (self._current_action + 1) % self._n_actions

        # Find current action's position in priority order
        try:
            current_idx = self._action_order.index(self._current_action)
        except ValueError:
            current_idx = -1

        # Next action in priority order (wraps around)
        next_idx = (current_idx + 1) % len(self._action_order)
        return self._action_order[next_idx]

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        # Store initial encoding
        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_dist = 0.0
            # Start with the best action from previous levels (if any)
            if self._action_success.sum() > 0:
                self._current_action = int(np.argmax(
                    self._action_success[:self._n_actions]))
            else:
                self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(enc)

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Compute action priority order after exploration
        if self._action_order is None:
            self._action_order = self._compute_action_order()
            # Start with highest-priority action
            if self._action_success.sum() > 0:
                self._current_action = self._action_order[0]

        # Reactive policy
        progress = (self._prev_dist - dist) > 1e-4
        no_change = abs(self._prev_dist - dist) < 1e-6

        self._steps_on_action += 1

        if progress:
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
            # Record success for cross-level transfer
            self._action_success[self._current_action] += 1.0
        else:
            self._consecutive_progress = 0

            if self._steps_on_action >= self._patience or no_change:
                self._actions_tried_this_round += 1
                self._steps_on_action = 0
                self._patience = 3

                if self._actions_tried_this_round >= self._n_actions:
                    self._current_action = self._rng.randint(self._n_actions)
                    self._actions_tried_this_round = 0
                else:
                    # Use priority order instead of round-robin
                    self._current_action = self._next_action_in_order()

        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return self._current_action

    def on_level_transition(self):
        # Reset within-level state but KEEP action success counts
        self._init_level_state()


CONFIG = {
    "n_dims": N_DIMS,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "family": "cross-level reactive switching",
    "tag": "defense v27 (ℓ₁ reactive + cross-level action memory, zero learned params)",
}

SUBSTRATE_CLASS = CrossLevelReactiveSubstrate
