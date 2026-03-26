"""
sub1084_defense_v21.py — Reactive state-dependent action switching (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1084 --substrate experiments/sub1084_defense_v21.py

FAMILY: Reactive action switching (new architecture)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: Immediate obs-comparison-based action switching breaks oscillation
WITHOUT learned encoding. If raw reactive policy works, encoding self-modification
(prosecution's alpha) adds no value. R3 test: does the substrate need to modify
HOW it processes, or is fixed comparison sufficient?

ARCHITECTURE (fundamentally different from prosecution v19):
- enc = avgpool8 (64D) — different dimensionality than prosecution's 256D
- Track obs_0 = initial observation per episode
- NO learned parameters: no EMA, no W_pred, no alpha, no mean_delta
- Pure reactive policy:
  1. Take action, observe result
  2. Compare: did distance to obs_0 DECREASE? (progress)
  3. If YES → repeat same action (exploit)
  4. If NO → advance to next action in round-robin (explore)
  5. After trying all N_KB actions without progress → random action
- Exploitation streak: if an action works K times in a row, keep it longer
  (patience increases with consecutive successes)
- Reset patience on action switch

WHY THIS IS DIFFERENT FROM PROSECUTION:
- Prosecution: PREDICTS effect before action via learned mean_delta + alpha weighting
- Defense: REACTS to effect after action via immediate comparison, zero learning
- Prosecution: continuous EMA statistics across all steps
- Defense: only cares about the LAST step's outcome
- If defense works equally well → ℓ_π unnecessary overhead

WHY THIS ADDRESSES OSCILLATION:
- 1082: fixed strategy oscillates because same action repeated regardless of state
- This substrate: switches action when current one stops producing progress
- Natural anti-oscillation: if A→B→A→B, substrate detects "A not progressing" and switches

KILL: 0/3 responsive games → KILL.
SUCCESS: any responsive game shows L1 OR sign changes < 30/74.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 8  # avgpool8: 64/8 = 8 blocks per dim
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS  # 64
N_KB = 7
EXPLORE_STEPS = 50  # initial random exploration before reactive policy kicks in
MAX_PATIENCE = 20   # max steps to stick with one action before forced switch


def _obs_to_enc(obs):
    """avgpool8: 64x64 → 8x8 = 64D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class ReactiveActionSwitchSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None       # initial encoding
        self._prev_enc = None
        self._prev_dist = None   # distance to initial before last action
        self._current_action = 0
        self._patience = 3       # steps before switching if no progress
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0  # how many actions tried without progress
        self._last_progress = False

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

    def _dist_to_initial(self, enc):
        """L1 distance to initial observation encoding."""
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

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
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(enc)

        # Initial exploration: try random actions to build context
        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Reactive policy: did the last action make progress?
        progress = (self._prev_dist - dist) > 1e-4  # moved toward initial
        no_change = abs(self._prev_dist - dist) < 1e-6  # nothing happened

        self._steps_on_action += 1

        if progress:
            # Action is working — keep it, increase patience
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
            self._last_progress = True
        else:
            # No progress or moved away
            self._consecutive_progress = 0
            self._last_progress = False

            if self._steps_on_action >= self._patience or no_change:
                # Switch action
                self._actions_tried_this_round += 1
                self._steps_on_action = 0
                self._patience = 3

                if self._actions_tried_this_round >= self._n_actions:
                    # Tried all actions without progress — random
                    self._current_action = self._rng.randint(self._n_actions)
                    self._actions_tried_this_round = 0
                else:
                    # Next action in round-robin
                    self._current_action = (self._current_action + 1) % self._n_actions

        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return self._current_action

    def on_level_transition(self):
        # Reset episode-specific state
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0


CONFIG = {
    "n_dims": N_DIMS,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "family": "reactive action switching",
    "tag": "defense v21 (ℓ₁ reactive switching, zero learned params)",
}

SUBSTRATE_CLASS = ReactiveActionSwitchSubstrate
