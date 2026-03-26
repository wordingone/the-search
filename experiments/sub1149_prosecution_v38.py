"""
sub1149_prosecution_v38.py — State-conditioned forward model (prosecution: ℓ_π)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1149 --substrate experiments/sub1149_prosecution_v38.py

FAMILY: State-conditioned (graph ban lift). Tagged: prosecution (ℓ_π).
R3 HYPOTHESIS: Per-(state,action) DIRECTION prediction enables navigation
in encoding space. Defense v56 tracks scalar change magnitude per (state,action).
Prosecution v38 tracks 256D change DIRECTION — which dims increase/decrease.
If direction info helps → ℓ_π adds value beyond ℓ₁ for state-conditioned models.

CONTROLLED COMPARISON vs defense v56:
- SAME: avgpool4 encoding, LSH hash → 256 states, exploration bonus, click regions
- DIFFERENT: v56 stores scalar magnitude per (state,action).
  v38 stores 256D direction vector per (state,action).
  v38 adds diversity tiebreaker (prefer actions with different directions).

Architecture:
- enc = avgpool4 (256D)
- LSH hash → 256 discrete states (frozen hyperplanes)
- W_dir[state, action] = EMA of (enc_next - enc_current), 256D per (state,action)
- Action selection: score[a] = ||W_dir[state, a]|| (magnitude = how much change)
- Tiebreaker: prefer action whose direction is MOST DIFFERENT from last direction
  (diversity in encoding-space trajectory)
- Exploration bonus for unvisited (state,action) pairs

NOTE ON RULE 15: Shares LSH state discretization with v56. This is a controlled
comparison — same infrastructure, different learned models (scalar vs vector).

KILL: ARC ≤ v56 → direction info doesn't help.
SUCCESS: ARC > v56 on comparable draws → direction enables better action selection.
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
MAX_ACTIONS = N_KB + N_CLICK_REGIONS  # 23
EXPLORE_STEPS = 50
EFFECT_LR = 0.1  # EMA learning rate
EXPLORE_BONUS = 0.01


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


class StateConditionedForwardSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0

        # FROZEN random hyperplanes for LSH (same seed protocol as v56)
        self._hyperplanes = self._rng.randn(N_LSH_BITS, N_DIMS).astype(np.float32)

        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_state = 0
        self._prev_action_idx = 0
        self._last_direction = np.zeros(N_DIMS, dtype=np.float32)

        # Per-(state, action) DIRECTION vectors (ℓ_π: learned)
        # W_dir[state, action, :] = EMA of (enc_next - enc_current)
        self._W_dir = np.zeros((N_STATES, MAX_ACTIONS, N_DIMS), dtype=np.float32)
        # Visit counts per (state, action)
        self._visits = np.zeros((N_STATES, MAX_ACTIONS), dtype=np.int32)

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
        """Pick best action: highest effect magnitude + diversity tiebreaker."""
        scores = np.zeros(self._n_active, dtype=np.float32)
        magnitudes = np.zeros(self._n_active, dtype=np.float32)

        for a in range(self._n_active):
            visits = self._visits[state, a]
            mag = np.sqrt(np.sum(self._W_dir[state, a] ** 2))
            magnitudes[a] = mag
            # Effect magnitude + exploration bonus
            scores[a] = mag + EXPLORE_BONUS / (visits + 1)

        # Find candidates with highest score
        max_score = scores.max()
        candidates = np.where(np.abs(scores - max_score) < 1e-6)[0]

        if len(candidates) == 1:
            return int(candidates[0])

        # Diversity tiebreaker: prefer action whose direction differs most
        # from the last direction taken (avoid repeating same trajectory)
        last_norm = np.sqrt(np.sum(self._last_direction ** 2)) + 1e-8
        best_diversity = -1.0
        best_action = int(candidates[0])

        for a in candidates:
            dir_a = self._W_dir[state, a]
            dir_norm = np.sqrt(np.sum(dir_a ** 2)) + 1e-8
            # Cosine distance (1 - cos_sim): higher = more different
            cos_sim = np.dot(dir_a, self._last_direction) / (dir_norm * last_norm)
            diversity = 1.0 - cos_sim
            if diversity > best_diversity:
                best_diversity = diversity
                best_action = int(a)

        return best_action

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
            action_idx = self._rng.randint(min(self._n_active, N_KB))
            self._prev_action_idx = action_idx
            return self._action_idx_to_env_action(action_idx)

        # Compute direction from previous action
        direction = enc - self._prev_enc  # 256D direction vector

        # Update W_dir for (prev_state, prev_action)
        s = self._prev_state
        a = self._prev_action_idx
        self._W_dir[s, a] = (1 - EFFECT_LR) * self._W_dir[s, a] + EFFECT_LR * direction
        self._visits[s, a] += 1
        self._last_direction = direction.copy()
        self.r3_updates += 1
        self.att_updates_total += 1

        # Initial exploration: cycle through all actions
        if self.step_count <= EXPLORE_STEPS:
            action_idx = self.step_count % self._n_active
        else:
            # Exploit: pick best action from CURRENT state (with diversity)
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
        self._last_direction = np.zeros(N_DIMS, dtype=np.float32)
        # Keep W_dir and visits across levels (ℓ_π: learned)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_lsh_bits": N_LSH_BITS,
    "n_states": N_STATES,
    "n_click_regions": N_CLICK_REGIONS,
    "explore_steps": EXPLORE_STEPS,
    "effect_lr": EFFECT_LR,
    "explore_bonus": EXPLORE_BONUS,
    "family": "state-conditioned forward model",
    "tag": "prosecution v38 (ℓ_π state-conditioned: LSH→256 states, per-(state,action) 256D direction EMA + diversity tiebreaker)",
}

SUBSTRATE_CLASS = StateConditionedForwardSubstrate
