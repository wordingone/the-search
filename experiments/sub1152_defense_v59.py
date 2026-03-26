"""
sub1152_defense_v59.py — Empowerment-based action selection (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1152 --substrate experiments/sub1152_defense_v59.py

FAMILY: Empowerment estimation (NEW — from information theory / sensory ecology).
Tagged: defense (ℓ₁).

R3 HYPOTHESIS: ALL 51 prior substrates select actions based on ACTIVATION signals
(did pixels change?). None use CONTRAST signals (do different actions produce
DISTINGUISHABLE outcomes?). Empowerment = mutual information I(Action; NextObs)
measures how many distinguishable futures the agent can reach. This is the
"contrast signal" identified as missing by the oscillatory kill register.

KEY DIFFERENCE FROM ALL PRIOR SUBSTRATES:
- Prior: score[a] = f(change_magnitude_of_a)
- This:  score[a] = how DIFFERENT is a's outcome distribution from other actions'?
  An action that always produces the same small change is STILL empowering if
  that change is distinguishable from what other actions produce.

Architecture:
- Phase 1 (steps 0-350): Empowerment estimation
  - For each action a ∈ {0..n_active}, execute a multiple times (50 steps each)
  - Record the distribution of next-observations (mean + variance of enc per action)
  - Compute pairwise distinguishability: ||mean_a - mean_b|| / sqrt(var_a + var_b + eps)
  - Actions with high distinguishability = high empowerment
- Phase 2 (remaining): Exploit empowered actions
  - Interleave high-empowerment actions with reactive argmin (distance-to-initial)
  - If ALL actions have zero empowerment (Mode 1 game): fall back to v30 reactive

ZERO learned parameters (defense: ℓ₁). Fixed estimation protocol.
Uses information-theoretic principle (empowerment) without learned models.

KILL: ARC ≤ v30 AND no empowerment signal detected on any game.
SUCCESS: Empowerment identifies controllable actions → better than change-magnitude.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 8
SAMPLES_PER_ACTION = 50  # observations per action during estimation
DISTINGUISHABILITY_EPS = 1e-6
TOP_EMPOWERED = 4  # exploit top-N empowered actions


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class EmpowermentSubstrate:
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

        # Empowerment estimation phase
        self._estimating = True
        self._current_action_idx = 0
        self._action_sample_count = 0

        # Per-action observation statistics (online mean + variance)
        self._n_active = N_KB
        self._action_means = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._action_vars = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._action_m2 = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)  # for Welford's
        self._action_counts = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.int32)

        # Empowerment scores
        self._empowerment = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.float32)
        self._empowered_actions = []  # sorted by empowerment

        # Exploit phase
        self._exploit_idx = 0
        self._exploit_patience = 0

        # Click regions
        self._click_actions = []
        self._regions_set = False

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _discover_regions(self, enc):
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

    def _idx_to_env_action(self, idx):
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

    def _update_action_stats(self, action_idx, enc):
        """Welford's online algorithm for mean and variance."""
        n = self._action_counts[action_idx] + 1
        self._action_counts[action_idx] = n
        delta = enc.astype(np.float64) - self._action_means[action_idx]
        self._action_means[action_idx] += delta / n
        delta2 = enc.astype(np.float64) - self._action_means[action_idx]
        self._action_m2[action_idx] += delta * delta2
        if n > 1:
            self._action_vars[action_idx] = self._action_m2[action_idx] / (n - 1)

    def _compute_empowerment(self):
        """Compute per-action empowerment as average distinguishability from other actions."""
        self._estimating = False
        self.r3_updates += 1
        self.att_updates_total += 1

        for a in range(self._n_active):
            if self._action_counts[a] < 2:
                self._empowerment[a] = 0.0
                continue

            # Distinguishability: how different is action a's outcome from all others?
            total_dist = 0.0
            n_comparisons = 0
            for b in range(self._n_active):
                if b == a or self._action_counts[b] < 2:
                    continue
                # Cohen's d analog: difference in means normalized by pooled variance
                mean_diff = self._action_means[a] - self._action_means[b]
                pooled_var = (self._action_vars[a] + self._action_vars[b]) / 2.0 + DISTINGUISHABILITY_EPS
                # Mahalanobis-like distance (sum over dimensions)
                d_sq = np.sum(mean_diff ** 2 / pooled_var)
                total_dist += np.sqrt(d_sq)
                n_comparisons += 1

            if n_comparisons > 0:
                self._empowerment[a] = total_dist / n_comparisons

        # Sort actions by empowerment
        action_scores = [(self._empowerment[a], a) for a in range(self._n_active)]
        action_scores.sort(reverse=True)
        self._empowered_actions = [a for _, a in action_scores[:TOP_EMPOWERED]]

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
            self._discover_regions(enc)
            return 0

        # Phase 1: Empowerment estimation
        if self._estimating:
            # Record observation statistics for current action
            self._update_action_stats(self._current_action_idx, enc)
            self._action_sample_count += 1

            if self._action_sample_count >= SAMPLES_PER_ACTION:
                # Move to next action
                self._action_sample_count = 0
                self._current_action_idx += 1

                if self._current_action_idx >= self._n_active:
                    # All actions sampled — compute empowerment
                    self._compute_empowerment()
                    self._prev_enc = enc.copy()

                    # Start exploit phase
                    if self._empowered_actions:
                        a = self._empowered_actions[0]
                        return self._idx_to_env_action(a)
                    else:
                        return 0

            self._prev_enc = enc.copy()
            return self._idx_to_env_action(self._current_action_idx)

        # Phase 2: Exploit empowered actions with reactive switching
        if self._empowered_actions and self._empowerment[self._empowered_actions[0]] > 0.01:
            # Interleave empowered actions
            a = self._empowered_actions[self._exploit_idx]
            self._exploit_patience += 1

            if self._exploit_patience >= 20:
                self._exploit_patience = 0
                self._exploit_idx = (self._exploit_idx + 1) % len(self._empowered_actions)

            # Reactive component: if we're far from initial, prefer actions
            # that move us back (distance-to-initial as tiebreaker)
            dist_to_initial = np.sum(np.abs(enc - self._enc_0))

            if dist_to_initial > 5.0 and self._exploit_patience % 4 == 0:
                # Every 4th step, try the action whose mean outcome is
                # closest to the initial encoding
                best_a = a
                best_dist = float('inf')
                for ea in self._empowered_actions:
                    d = np.sum(np.abs(self._action_means[ea].astype(np.float32) - self._enc_0))
                    if d < best_dist:
                        best_dist = d
                        best_a = ea
                self._prev_enc = enc.copy()
                return self._idx_to_env_action(best_a)

            self._prev_enc = enc.copy()
            return self._idx_to_env_action(a)

        # Fallback: v30-style reactive (zero empowerment = Mode 1 game)
        n_active = min(self._n_actions_env, N_KB)
        best_action = 0
        best_dist = float('inf')

        # Simple reactive: cycle through actions, prefer ones that
        # move encoding toward initial state
        dist_to_initial = np.sum(np.abs(enc - self._enc_0))
        if dist_to_initial < 0.5:
            # Near initial — explore by cycling
            action = (self.step_count // 5) % n_active
        else:
            # Away from initial — try to return (v30 behavior)
            action = (self.step_count // 3) % n_active

        self._prev_enc = enc.copy()
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._exploit_idx = 0
        self._exploit_patience = 0
        # Keep empowerment estimates across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "samples_per_action": SAMPLES_PER_ACTION,
    "top_empowered": TOP_EMPOWERED,
    "family": "empowerment estimation",
    "tag": "defense v59 (ℓ₁ empowerment: per-action outcome distributions, Welford mean+var, pairwise distinguishability → exploit most controllable actions)",
}

SUBSTRATE_CLASS = EmpowermentSubstrate
