"""
step0920_rudakov_graph.py -- Graph+argmin baseline (pre-ban ceiling).

Rudakov (2025) simplified: graph-based exploration with per-(state,action) visit counts.
This INTENTIONALLY uses a graph (banned in production) to measure the pre-ban ceiling.
How much does the graph ban cost? This answers it.

Implementation:
- obs_hash: hashlib.md5 on rounded enc (same as 919)
- G[(obs_hash, action)]: visit count for each (state, action) pair
- Action: argmin over G[(obs_hash, action)] — least-visited
- Ties broken randomly (epsilon=0.05 pure exploration)
- Encoding: avgpool16+centered (same as 895h)

This is essentially Step 674 but with current game versions and 25K steps.
Gives THE PRE-BAN CEILING for LS20 and FT09.

Run: LS20 + FT09, 25K, 10 seeds, cold, substrate_seed=seed.
Baseline: 895h cold LS20=268.0/seed, FT09=0/10.
NOTE: Graph ban applies in production. This experiment is for baseline measurement only.
"""
import sys, time, hashlib
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import defaultdict
from substrates.step0674 import _enc_frame

ENC_DIM = 256
EPSILON = 0.05   # small epsilon — mostly argmin
HASH_ROUND = 1
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def obs_hash(enc):
    rounded = np.round(enc, HASH_ROUND).astype(np.float16)
    return hashlib.md5(rounded.tobytes()).hexdigest()[:12]


class GraphArgmin920:
    """Graph+argmin baseline. GRAPH BAN applies in production — for measurement only."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)

        # Per-(state,action) visit counts — GRAPH (banned in production)
        self.visit_count = defaultdict(int)

        # Running mean for centering
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_hash = None

        self._unique_states = set()

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def process(self, obs):
        enc = self._encode(obs)
        h = obs_hash(enc)
        self._unique_states.add(h)

        # Action selection: argmin visit count (least-visited action from this state)
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            counts = np.array([self.visit_count[(h, a)] for a in range(self._n_actions)],
                               dtype=np.float32)
            # Break ties randomly
            min_count = counts.min()
            candidates = np.where(counts == min_count)[0]
            action = int(self._rng.choice(candidates))

        # Record visit
        self.visit_count[(h, action)] += 1
        self._prev_hash = h
        return action

    def on_level_transition(self):
        self._prev_hash = None

    def n_unique(self):
        return len(self._unique_states)

    def n_state_action(self):
        return len(self.visit_count)


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    for seed in seeds:
        sub = GraphArgmin920(n_actions=n_actions, seed=seed)
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
        print(f"    seed={seed}: L1={completions:4d}  unique_states={sub.n_unique()}  state_action_pairs={sub.n_state_action()}")
    return results


print("=" * 70)
print("STEP 920 — GRAPH+ARGMIN BASELINE (pre-ban ceiling)")
print("=" * 70)
print("Per-(state,action) visit counts + argmin. GRAPH BAN applies in production.")
print("This measures the pre-ban ceiling: how much does graph ban cost?")
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
print(f"STEP 920 RESULTS (Graph+argmin pre-ban ceiling):")
print(f"  FT09: L1={ft09_mean:.1f}/seed  std={np.std(ft09_results):.1f}  zero={ft09_zeros}/10")
print(f"  LS20: L1={ls20_mean:.1f}/seed  std={np.std(ls20_results):.1f}  zero={ls20_zeros}/10")
print(f"  {ft09_results}")
print(f"  {ls20_results}")
print(f"\nComparison table:")
print(f"  Method              LS20 L1/seed   FT09 L1/seed   Graph?")
print(f"  895h cold (ours)    268.0          0.0            No")
print(f"  920 Graph+argmin    {ls20_mean:.1f}          {ft09_mean:.1f}            YES (banned)")
print(f"  Graph ban cost LS20: {ls20_mean - 268.0:+.1f}/seed")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 920 DONE")
