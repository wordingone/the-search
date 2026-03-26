"""
sub1173_defense_v73.py — Wide-search reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1173 --substrate experiments/sub1173_defense_v73.py

FAMILY: Wide-search reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: The random baseline (Step 1170) gets 3/5 L1 while all
engineered substrates get 2/5. Why? Random samples from the FULL action
space (all n_actions including every click position). Engineered substrates
limit themselves to 7 KB + 8-16 saliency-derived clicks, MISSING responsive
click targets that aren't visually salient.

Architecture:
- Phase 1 (EXPLORE, ~200 steps): sample random actions from FULL action space.
  After each action, measure pixel change. If change > threshold, record the
  action as "responsive."
- Phase 2 (EXPLOIT): reactive cycling (v30-style) over ONLY the responsive
  actions discovered in Phase 1. If no responsive actions found, fall back to
  full keyboard cycling.

DIFFERENT from prior substrates:
- v30: limits to 7 KB + 8 saliency clicks (misses non-salient responsive targets)
- v69: systematic spatial grid search (slow — 5 hold steps per point)
- random: explores everything but NEVER exploits (wastes steps on non-responsive)
- v73: random's COVERAGE + v30's EXPLOITATION. Best of both.

ZERO learned parameters (defense: ℓ₁). Fixed protocol.

KILL: L1 ≤ random (3/5) AND ARC ≤ random (0.002).
SUCCESS: L1 ≥ 3/5 with ARC > random. Coverage + efficiency.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

EXPLORE_STEPS = 200        # steps for random exploration phase
RESPONSE_THRESH = 0.3      # min pixel change to count action as responsive
MAX_RESPONSIVE = 20        # max responsive actions to track


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class WideSearchReactiveSubstrate:
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

        # Explore phase
        self._exploring = True
        self._prev_action = 0
        self._responsive_actions = {}  # action -> max_change

        # Exploit phase
        self._exploit_actions = []   # sorted list of responsive actions
        self._exploit_idx = 0
        self._exploit_patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _transition_to_exploit(self):
        """End exploration, set up exploitation over responsive actions."""
        self._exploring = False
        self.r3_updates += 1
        self.att_updates_total += 1

        if self._responsive_actions:
            # Sort by response magnitude, take top MAX_RESPONSIVE
            sorted_actions = sorted(
                self._responsive_actions.items(),
                key=lambda x: -x[1]
            )[:MAX_RESPONSIVE]
            self._exploit_actions = [a for a, _ in sorted_actions]
        else:
            # No responsive actions found — fall back to keyboard cycling
            n_kb = min(self._n_actions_env, N_KB)
            self._exploit_actions = list(range(n_kb))

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
            # First action: random from full space
            self._prev_action = int(self._rng.randint(0, self._n_actions_env))
            return self._prev_action

        # Measure change from previous action
        delta = float(np.sum(np.abs(enc - self._prev_enc)))

        # === EXPLORE PHASE: random sampling, record responsive actions ===
        if self._exploring:
            # Record response of previous action
            if delta > RESPONSE_THRESH:
                prev = self._prev_action
                if prev in self._responsive_actions:
                    self._responsive_actions[prev] = max(
                        self._responsive_actions[prev], delta
                    )
                else:
                    self._responsive_actions[prev] = delta

            # Check if exploration is done
            if self.step_count >= EXPLORE_STEPS:
                self._transition_to_exploit()
                # Fall through to exploit
            else:
                # Sample next random action from FULL action space
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action

        # === EXPLOIT PHASE: reactive cycling over responsive actions ===
        dist = float(np.sum(np.abs(enc - self._enc_0)))

        if dist >= self._prev_dist:
            # No progress — switch action
            self._exploit_idx = (self._exploit_idx + 1) % len(self._exploit_actions)
            self._exploit_patience = 0
        else:
            # Progress — hold action
            self._exploit_patience += 1
            if self._exploit_patience > 10:
                self._exploit_patience = 0
                self._exploit_idx = (self._exploit_idx + 1) % len(self._exploit_actions)

        self._prev_dist = dist
        self._prev_enc = enc.copy()
        action = self._exploit_actions[self._exploit_idx]
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._exploit_idx = 0
        self._exploit_patience = 0
        # Keep responsive actions across levels (same game, likely same actions)
        # Don't re-explore — responsive actions persist within a game
        if self._exploring:
            # If still exploring when level transitions, keep exploring
            pass
        # enc_0 reset will be set on next process() call


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "response_thresh": RESPONSE_THRESH,
    "max_responsive": MAX_RESPONSIVE,
    "family": "wide-search reactive",
    "tag": "defense v73 (ℓ₁ wide-search: random exploration of FULL action space → reactive exploitation of responsive actions. Combines random's coverage with v30's efficiency.)",
}

SUBSTRATE_CLASS = WideSearchReactiveSubstrate
