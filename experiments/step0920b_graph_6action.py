"""
step0920b_graph_6action.py -- Graph+argmin with N_ACTIONS=6 (baseline actions only).

Step 920 (68 actions) got L1=0/10 on FT09. Does restricting to 6 baseline actions help?
FT09 baseline_actions = [17, 19, 15, 21, 65, 26] (6 puzzle-relevant click positions).
6^7 = 279K sequences — tractable for graph+argmin. Pre-ban 674 solved FT09 somehow.

R3 question: is 6^7 tractable for simple argmin, or does FT09 require a more sophisticated traversal?

Action mapping: substrate uses 0-5, env receives baseline_actions[action].
Graph: per-(hash, action_idx) where action_idx ∈ 0-5.

Run: FT09 only, 25K, 10 seeds cold. Compare to 920 (68 actions, L1=0/10).
"""
import sys, time, hashlib
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import defaultdict
from substrates.step0674 import _enc_frame

ENC_DIM = 256
EPSILON = 0.05
HASH_ROUND = 1
BASELINE_ACTIONS = [17, 19, 15, 21, 65, 26]  # FT09 puzzle tiles
N_ACTIONS = len(BASELINE_ACTIONS)  # 6
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def obs_hash(enc):
    rounded = np.round(enc, HASH_ROUND).astype(np.float16)
    return hashlib.md5(rounded.tobytes()).hexdigest()[:12]


class GraphArgmin920b:
    """Graph+argmin on 6 baseline actions only."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        self.visit_count = defaultdict(int)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._unique_states = set()

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def process(self, obs):
        """Returns action_idx ∈ 0-5. Caller maps to BASELINE_ACTIONS."""
        enc = self._encode(obs)
        h = obs_hash(enc)
        self._unique_states.add(h)

        if self._rng.random() < EPSILON:
            action_idx = int(self._rng.randint(0, N_ACTIONS))
        else:
            counts = np.array([self.visit_count[(h, a)] for a in range(N_ACTIONS)])
            min_count = counts.min()
            candidates = np.where(counts == min_count)[0]
            action_idx = int(self._rng.choice(candidates))

        self.visit_count[(h, action_idx)] += 1
        return action_idx

    def on_level_transition(self): pass

    def n_unique(self): return len(self._unique_states)


def make_game():
    try:
        import arcagi3; return arcagi3.make("FT09")
    except:
        import util_arcagi3; return util_arcagi3.make("FT09")


results = []
t0 = time.time()
print("=" * 70)
print("STEP 920b — GRAPH+ARGMIN WITH 6 BASELINE ACTIONS (FT09)")
print("=" * 70)
print(f"Baseline actions: {BASELINE_ACTIONS}")
print(f"6^7 = {6**7:,} sequences (tractable?). Compare: 68^7 ≈ 10^12 (920 got 0/10)")

for seed in TEST_SEEDS:
    sub = GraphArgmin920b(seed=seed)
    env = make_game()
    obs = env.reset(seed=seed * 1000)
    step = 0; completions = 0; current_level = 0
    while step < TEST_STEPS:
        if obs is None:
            obs = env.reset(seed=seed * 1000); sub.on_level_transition(); continue
        action_idx = sub.process(np.asarray(obs, dtype=np.float32))
        env_action = BASELINE_ACTIONS[action_idx]
        obs, _, done, info = env.step(env_action)
        step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=seed * 1000); current_level = 0
            sub.on_level_transition()
    results.append(completions)
    print(f"  seed={seed}: L1={completions:4d}  unique_states={sub.n_unique()}")

mean = np.mean(results)
zeros = sum(1 for x in results if x == 0)
print(f"\n{'='*70}")
print(f"STEP 920b RESULTS (graph+argmin, 6 baseline actions):")
print(f"  FT09: L1={mean:.1f}/seed  std={np.std(results):.1f}  zero={zeros}/10")
print(f"  {results}")
print(f"\nComparison:")
print(f"  920  graph+argmin 68 actions: L1=0.0/seed  10/10 zeros")
print(f"  920b graph+argmin  6 actions: L1={mean:.1f}/seed  {zeros}/10 zeros")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 920b DONE")
