"""
step0919_count_based.py -- Count-based exploration (Bellemare 2016) on LS20 + FT09.

Pseudo-count via observation hash. Per-observation visit counting (NOT per-(state,action)).

Graph ban check:
- obs_count[hash]: per-observation frequency, NOT per-(state,action). ALLOWED.
- Action selection: predicted next-obs novelty via W. Not a state-action table.

Implementation:
- obs_hash: hashlib.md5 on rounded enc (2 decimal places → stable discrete ID)
- obs_count[hash] = visit count
- Novelty = 1 / sqrt(obs_count + 1)  (Bellemare pseudo-count formula)
- Track novelty_per_action = EMA of novelty observed after each action
- Action: softmax over novelty_per_action

Run: LS20 + FT09, 25K, 10 seeds, cold, substrate_seed=seed.
Baseline: 895h cold LS20=268.0/seed, FT09=0/10.
"""
import sys, time, hashlib
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import defaultdict, deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
NOVELTY_EMA = 0.05
INIT_NOVELTY = 1.0
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
HASH_ROUND = 1   # round to 1 decimal for discrete hashing
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def obs_hash(enc):
    """Stable hash of rounded encoding."""
    rounded = np.round(enc, HASH_ROUND).astype(np.float16)
    return hashlib.md5(rounded.tobytes()).hexdigest()[:12]


def softmax_action(novelty, temp, rng):
    x = novelty / temp
    x = x - np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(novelty), p=probs))


class CountBased919:
    """Count-based exploration via observation hash pseudo-count."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)

        # Per-observation visit counts (NOT per-(state,action))
        self.obs_count = defaultdict(int)

        # Novelty per action (EMA)
        self.novelty_per_action = np.full(n_actions, INIT_NOVELTY, dtype=np.float32)

        # Running mean for centering
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        self._prev_action = None
        self._unique_states = set()

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def process(self, obs):
        enc = self._encode(obs)

        # Count this observation
        h = obs_hash(enc)
        self.obs_count[h] += 1
        self._unique_states.add(h)

        # Novelty = 1/sqrt(count) — Bellemare pseudo-count
        novelty = 1.0 / np.sqrt(self.obs_count[h] + 1.0)

        # Update novelty for previous action
        if self._prev_action is not None:
            a = self._prev_action
            self.novelty_per_action[a] = (
                (1 - NOVELTY_EMA) * self.novelty_per_action[a] + NOVELTY_EMA * novelty
            )

        # Action selection
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_action(self.novelty_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_action = None

    def n_unique(self):
        return len(self._unique_states)


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    for seed in seeds:
        sub = CountBased919(n_actions=n_actions, seed=seed)
        env = make_game(game_name)
        obs = env.reset(seed=seed * 1000)
        step = 0; completions = 0; current_level = 0
        while step < n_steps:
            if obs is None:
                obs = env.reset(seed=seed * 1000)
                sub.on_level_transition()
                continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
            obs, _, done, info = env.step(action)
            step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > current_level:
                completions += (cl - current_level)
                current_level = cl
                sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed * 1000)
                current_level = 0
                sub.on_level_transition()
        results.append(completions)
        print(f"    seed={seed}: L1={completions:4d}  unique_states={sub.n_unique()}")
    return results


print("=" * 70)
print("STEP 919 — COUNT-BASED EXPLORATION (Bellemare 2016) ON LS20 + FT09")
print("=" * 70)
print("Per-obs visit count (NOT per-(state,action)). Novelty=1/sqrt(count+1).")
t0 = time.time()

print("\n--- FT09 (68 actions, 25K, 10 seeds) ---")
ft09_results = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS)
ft09_mean = np.mean(ft09_results)
ft09_zeros = sum(1 for x in ft09_results if x == 0)

print("\n--- LS20 (4 actions, 25K, 10 seeds) ---")
ls20_results = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS)
ls20_mean = np.mean(ls20_results)
ls20_zeros = sum(1 for x in ls20_results if x == 0)

print(f"\n{'='*70}")
print(f"STEP 919 RESULTS (Count-based Bellemare):")
print(f"  FT09: L1={ft09_mean:.1f}/seed  std={np.std(ft09_results):.1f}  zero={ft09_zeros}/10")
print(f"  LS20: L1={ls20_mean:.1f}/seed  std={np.std(ls20_results):.1f}  zero={ls20_zeros}/10")
print(f"  {ft09_results}")
print(f"  {ls20_results}")
print(f"\nComparison:")
print(f"  895h cold LS20: 268.0/seed  FT09: 0/10")
print(f"  919 Count LS20: {ls20_mean:.1f}/seed  FT09: {ft09_mean:.1f}/seed")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 919 DONE")
