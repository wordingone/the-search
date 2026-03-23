"""
step0817_ft09_null.py -- FT09 null hypothesis with varied substrate seeds.

Establishes proper FT09 floor: random action with substrate_seed=ts for each
test seed. Fixes the degenerate case (substrate_seed=0 for all seeds = n_eff=1).

FT09 floor from step807_ft09 (substrate_seed=0 for all): L1=0.
This confirms whether L1=0 holds with proper varied seeds.

R3 null hypothesis: no action selection mechanism beats random on FT09 at 25K steps.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np

N_ACTIONS = 68
PRETRAIN_SEEDS = list(range(1, 6))
PRETRAIN_STEPS = 5_000
TEST_SEEDS = list(range(6, 11))
TEST_STEPS = 25_000


def _make_ft09():
    try:
        import arcagi3
        return arcagi3.make("FT09")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("FT09")


def run_random_ft09(env_seed, n_steps, substrate_seed):
    """Pure random action on FT09 with given substrate_seed (for varied RNG)."""
    rng = np.random.RandomState(substrate_seed)
    env = _make_ft09()
    obs = env.reset(seed=env_seed)
    level_completions = 0
    current_level = 0
    step = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed)
            current_level = 0
            continue
        action = rng.randint(0, N_ACTIONS)
        obs, reward, done, info = env.step(action)
        step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            level_completions += (cl - current_level)
            current_level = cl
        if done:
            obs = env.reset(seed=env_seed)
            current_level = 0
    return level_completions


print("=" * 70)
print("STEP 817 FT09 NULL HYPOTHESIS — random action, varied substrate seeds")
print("=" * 70)
print(f"n_actions={N_ACTIONS}, n_steps={TEST_STEPS}, substrate_seed=ts per seed")
print("Null: L1=0 regardless of random seed")
print("=" * 70)

t0 = time.time()
totals = []
for ts in TEST_SEEDS:
    env_seed = ts * 1000
    c = run_random_ft09(env_seed, TEST_STEPS, substrate_seed=ts)
    totals.append(c)
    print(f"  seed={ts}  completions={c}")

total = sum(totals)
print()
print(f"Total: {total}  Mean: {np.mean(totals):.2f}/seed")
print(f"FT09 floor (varied seeds): {'0' if total == 0 else total} level completions at 25K steps")
print(f"Elapsed: {time.time()-t0:.1f}s")
print()
print("STEP 817 FT09 NULL DONE")
