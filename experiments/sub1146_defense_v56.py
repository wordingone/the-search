"""
sub1146_defense_v56.py — State-conditioned action model (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1146 --substrate experiments/sub1146_defense_v56.py

FAMILY: State-conditioned action (NEW, post-graph-ban-lift). Tagged: defense (ℓ₁).
R3 HYPOTHESIS: RETHINK origin. PB27 says Mode 2 games have STATE-DEPENDENT
action effects — kb1=+0.37 toward from one state, but oscillation (47/74 sign
changes) means the same action has DIFFERENT effects from different states.
All 45 prior substrates treat actions as STATE-INDEPENDENT (reactive switching
uses global progress, not state-dependent models). Graph ban was LIFTED
(Jun, 2026-03-25) — per-(state,action) memory is legal again.

This substrate discretizes state via LSH hash and tracks per-(state,action)
change magnitude. From each state, it picks the action that produces the
LARGEST ℓ₁ change (most responsive action from THIS specific state).

Architecture:
- enc = avgpool4 (256D, FIXED — ℓ₁ defense)
- state = LSH hash(enc) → 8-bit → 256 discrete states
- For each (state, action) pair: EMA of ℓ₁ change magnitude
  effect[state][action] = (1-lr) * effect[state][action] + lr * change
- Action selection: from current state, pick action with highest effect
  Ties broken by least-visited (exploration bonus)
- Click regions: top-16 salient blocks (same as v30)
- Encoding FIXED (ℓ₁). Only action POLICY adapts.

WHY DEFENSE: Encoding function (avgpool4 → LSH hash) is frozen. State
discretization is fixed. Only the action-effect table updates — this is
analogous to v30's reactive switching but STATE-CONDITIONED.

KILL: ARC ≤ v30 (0.3319).
SUCCESS: State-conditioned selection breaks oscillation on Mode 2 games.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 16
N_LSH_BITS = 8
N_STATES = 256  # 2^8
EXPLORE_STEPS = 50
EFFECT_LR = 0.1  # EMA learning rate for effect tracking
EXPLORE_BONUS = 0.01  # small bonus for unvisited (state, action) pairs


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    """Block center -> click action index."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class StateConditionedSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0

        # FROZEN random hyperplanes for LSH (ℓ₁ compliant)
        self._hyperplanes = self._rng.randn(N_LSH_BITS, N_DIMS).astype(np.float32)

        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_state = 0
        self._prev_action_idx = 0

        # Per-(state, action) effect tracking
        # effect[state][action_idx] = EMA of ℓ₁ change
        self._effect = np.zeros((N_STATES, N_KB + N_CLICK_REGIONS), dtype=np.float32)
        # Visit counts per (state, action) for exploration bonus
        self._visits = np.zeros((N_STATES, N_KB + N_CLICK_REGIONS), dtype=np.int32)

        # Click regions
        self._click_actions = []
        self._n_active = N_KB
        self._regions_set = False

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _lsh_hash(self, enc):
        """LSH: 8 random hyperplanes -> 8-bit hash -> state index."""
        bits = (self._hyperplanes @ enc > 0).astype(np.int32)
        state = 0
        for i in range(N_LSH_BITS):
            state |= (bits[i] << i)
        return state

    def _discover_regions(self, enc):
        """Find top-16 salient blocks for click targeting."""
        screen_mean = enc.mean()
        saliency = np.abs(enc - screen_mean)
        sorted_blocks = np.argsort(saliency)[::-1]
        click_regions = list(sorted_blocks[:N_CLICK_REGIONS].astype(int))
        self._click_actions = [_block_to_click_action(b) for b in click_regions]
        if self._has_clicks:
            self._n_active = N_KB + N_CLICK_REGIONS
        else:
            self._n_active = min(self._n_actions_env, N_KB)
        self._regions_set = True

    def _action_idx_to_env_action(self, idx):
        """Convert internal index to PRISM action."""
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

    def _select_action(self, state):
        """Pick best action from this state: highest effect + exploration bonus."""
        scores = np.zeros(self._n_active, dtype=np.float32)
        for a in range(self._n_active):
            effect = self._effect[state, a]
            visits = self._visits[state, a]
            # Effect + exploration bonus for unvisited pairs
            scores[a] = effect + EXPLORE_BONUS / (visits + 1)

        # Greedy with random tiebreak
        max_score = scores.max()
        candidates = np.where(np.abs(scores - max_score) < 1e-8)[0]
        return int(self._rng.choice(candidates))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)
        state = self._lsh_hash(enc)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_state = state
            self._discover_regions(enc)
            # Start with random action
            action_idx = self._rng.randint(min(self._n_active, N_KB))
            self._prev_action_idx = action_idx
            return self._action_idx_to_env_action(action_idx)

        # Compute change from previous action
        change = np.sum(np.abs(enc - self._prev_enc))

        # Update effect model for (prev_state, prev_action)
        s = self._prev_state
        a = self._prev_action_idx
        self._effect[s, a] = (1 - EFFECT_LR) * self._effect[s, a] + EFFECT_LR * change
        self._visits[s, a] += 1
        self.r3_updates += 1
        self.att_updates_total += 1

        # Initial exploration: cycle through all actions
        if self.step_count <= EXPLORE_STEPS:
            action_idx = self.step_count % self._n_active
        else:
            # Exploit: pick best action from CURRENT state
            action_idx = self._select_action(state)

        self._prev_enc = enc.copy()
        self._prev_state = state
        self._prev_action_idx = action_idx
        return self._action_idx_to_env_action(action_idx)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_state = 0
        self._prev_action_idx = 0
        # Keep effect model and visits across levels (learned)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_lsh_bits": N_LSH_BITS,
    "n_states": N_STATES,
    "n_click_regions": N_CLICK_REGIONS,
    "explore_steps": EXPLORE_STEPS,
    "effect_lr": EFFECT_LR,
    "explore_bonus": EXPLORE_BONUS,
    "family": "state-conditioned action",
    "tag": "defense v56 (ℓ₁ state-conditioned: LSH hash → 256 states, per-(state,action) effect EMA, graph ban LIFTED)",
}

SUBSTRATE_CLASS = StateConditionedSubstrate
