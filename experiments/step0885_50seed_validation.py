"""
step0885_50seed_validation.py -- step800b on 50 seeds for robust cold navigation estimate.

R3 hypothesis (validation): does 800b cold navigation generalize beyond seeds 6-10?
Prior result: 327/seed on seeds 6-10 (substrate_seed=0, n_eff=1 artifact).
This test uses 50 env_seeds with substrate_seeds varied (ss = seed % 4) to get
a population estimate with n_eff > 1.

Protocol: 50 env seeds (1-50) × substrate_seed = seed % 4. 25K steps each.
Metric: mean L1/seed, 5th/95th percentile.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0800b import EpsilonActionChange800b

N_SEEDS = 50
TEST_STEPS = 25_000
N_ACTIONS = 4

# Random baseline: 36.4/seed confirmed.

def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_one(env_seed, substrate_seed, n_steps):
    sub = EpsilonActionChange800b(n_actions=N_ACTIONS, seed=substrate_seed)
    sub.reset(substrate_seed)
    env = make_game(); obs = env.reset(seed=env_seed)
    completions = 0; current_level = 0; step = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); current_level = 0
            sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=env_seed); current_level = 0
            sub.on_level_transition()
    return completions


print("=" * 70)
print("STEP 885 — 800b ON 50 SEEDS")
print("=" * 70)
print(f"n_seeds={N_SEEDS}, steps={TEST_STEPS}, substrate_seed = seed % 4")

t0 = time.time()
results = []

for s in range(1, N_SEEDS + 1):
    env_seed = s * 1000
    substrate_seed = s % 4
    c = run_one(env_seed, substrate_seed, TEST_STEPS)
    results.append(c)
    if s % 10 == 0:
        so_far = results
        print(f"  seed={s:3d}: c={c:4d}  running_mean={np.mean(so_far):.1f}/seed  elapsed={time.time()-t0:.1f}s")

print()
print(f"Results (50 seeds):")
print(f"  Mean:   {np.mean(results):.1f}/seed")
print(f"  Median: {np.median(results):.1f}/seed")
print(f"  Std:    {np.std(results):.1f}")
print(f"  P5:     {np.percentile(results, 5):.1f}")
print(f"  P95:    {np.percentile(results, 95):.1f}")
print(f"  Min:    {np.min(results)}")
print(f"  Max:    {np.max(results)}")
print(f"  >0:     {sum(1 for c in results if c > 0)}/{N_SEEDS}")
print(f"  >100:   {sum(1 for c in results if c > 100)}/{N_SEEDS}")
print(f"  >300:   {sum(1 for c in results if c > 300)}/{N_SEEDS}")
print()
print(f"Random baseline: 36.4/seed")
print(f"Improvement factor: {np.mean(results)/36.4:.1f}×")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 885 DONE")
