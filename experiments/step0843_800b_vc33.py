"""
step0843_800b_vc33.py -- step800b on VC33 with 68 actions.

R3 hypothesis: does 800b's "most visual change" strategy generalize to VC33?
VC33 requires finding a magic pixel at (62,26) or (62,34). These are specific
click positions. If the right click produces more visual change than others,
800b would identify it.

From step874 characterization: LS20 actions have low CV (similar change magnitudes).
FT09 has high CV (action 5 dominates). VC33 is unknown.

68-action test: 4 directions + 64 grid clicks.
Protocol: 3 seeds × 25K steps. Report L1 completions.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0800b import EpsilonActionChange800b

TEST_SEEDS = [0, 1, 2]
TEST_STEPS = 25_000
N_ACTIONS = 68  # 4 dirs + 64 grid clicks


def make_vc33():
    try:
        import arcagi3; return arcagi3.make("VC33")
    except:
        import util_arcagi3; return util_arcagi3.make("VC33")


def run_cold(env_seed, n_steps):
    sub = EpsilonActionChange800b(n_actions=N_ACTIONS, seed=0)
    sub.reset(0)
    env = make_vc33(); obs = env.reset(seed=env_seed)
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
    # Report delta_per_action for first 8 actions
    d = sub.delta_per_action[:8] if hasattr(sub, 'delta_per_action') else None
    return completions, d


print("=" * 70)
print("STEP 843 — 800b ON VC33 (68 actions: 4 dirs + 64 grid clicks)")
print("=" * 70)
print(f"Protocol: cold, {TEST_STEPS} steps, seeds {TEST_SEEDS}")

t0 = time.time()
results = []

for ts in TEST_SEEDS:
    c, d = run_cold(ts, TEST_STEPS)
    results.append(c)
    print(f"  seed={ts}: L1={c}  delta[:8]={d}")

print()
print(f"Mean: {np.mean(results):.1f}/seed")
print(f"Random baseline: ~0 (VC33 magic pixel requires 3-action targeted discovery)")
print(f"674+running-mean+argmin (3-action): 20/20 at 25K steps (Step 705)")
print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 843 DONE")
