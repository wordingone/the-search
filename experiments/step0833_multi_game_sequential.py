"""
step0833_multi_game_sequential.py -- Multi-game sequential: LS20 → FT09 → LS20.

R3 hypothesis: does delta_per_action from LS20 interfere with FT09 adaptation?
Does LS20 performance recover after FT09 exposure?

Protocol: step800b on LS20 (10K) → FT09 (10K) → LS20 again (10K).
Tracks:
- Level completions per phase
- delta_per_action values at end of each phase
- Whether LS20 recovery matches fresh cold run on LS20

Key question: is 800b's per-action state portable across game contexts?
FT09 has 68 actions. LS20 has 4. Mismatch exposes adaptation robustness.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0800b import EpsilonActionChange800b

LS20_ACTIONS = 4
FT09_ACTIONS = 68
STEPS_PER_PHASE = 10_000
ENV_SEED = 6000  # fixed for reproducibility


def make_ls20():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def make_ft09():
    try:
        import arcagi3; return arcagi3.make("FT09")
    except:
        import util_arcagi3; return util_arcagi3.make("FT09")


def run_phase(substrate, env_fn, n_actions, env_seed, n_steps):
    """Run one phase, return (completions, final_delta_per_action)."""
    env = env_fn()
    obs = env.reset(seed=env_seed)
    completions = 0; current_level = 0; step = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition(); continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr) % n_actions
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
    delta_snapshot = substrate.delta_per_action.copy() if hasattr(substrate, 'delta_per_action') else None
    return completions, delta_snapshot


print("=" * 70)
print("STEP 833 — MULTI-GAME SEQUENTIAL (LS20 → FT09 → LS20)")
print("=" * 70)
print(f"Phase steps: {STEPS_PER_PHASE} each. env_seed={ENV_SEED}.")

t0 = time.time()

# Substrate: 4-action (LS20 compatible). FT09 will use modulo.
# Note: delta_per_action is per action index. For FT09 with 68 actions, we'll
# use a separate 68-action substrate.
print("\n--- Phase 1: LS20 (fresh 800b, 4 actions) ---")
sub_ls20 = EpsilonActionChange800b(n_actions=LS20_ACTIONS, seed=0)
sub_ls20.reset(0)
c1, d1 = run_phase(sub_ls20, make_ls20, LS20_ACTIONS, ENV_SEED, STEPS_PER_PHASE)
print(f"  L1={c1}  delta_per_action={d1}")

print("\n--- Phase 2: FT09 (fresh 800b, 68 actions, cold start) ---")
sub_ft09 = EpsilonActionChange800b(n_actions=FT09_ACTIONS, seed=0)
sub_ft09.reset(0)
c2, d2 = run_phase(sub_ft09, make_ft09, FT09_ACTIONS, ENV_SEED, STEPS_PER_PHASE)
print(f"  L1={c2}  delta_per_action[:8]={d2[:8] if d2 is not None else None}")

print("\n--- Phase 3: LS20 again (fresh 800b, 4 actions, cold start) ---")
sub_ls20_2 = EpsilonActionChange800b(n_actions=LS20_ACTIONS, seed=0)
sub_ls20_2.reset(0)
c3, d3 = run_phase(sub_ls20_2, make_ls20, LS20_ACTIONS, ENV_SEED, STEPS_PER_PHASE)
print(f"  L1={c3}  delta_per_action={d3}")

# Baseline: pure cold LS20 run
print("\n--- Baseline: LS20 cold (independent 800b) ---")
sub_base = EpsilonActionChange800b(n_actions=LS20_ACTIONS, seed=0)
sub_base.reset(0)
c_base, _ = run_phase(sub_base, make_ls20, LS20_ACTIONS, ENV_SEED, STEPS_PER_PHASE)
print(f"  L1={c_base}")

print()
print("Summary:")
print(f"  LS20 Phase 1: {c1}  |  LS20 Phase 3 (after FT09): {c3}  |  LS20 baseline: {c_base}")
print(f"  FT09 Phase 2: {c2} (expected 0)")
print(f"  Recovery ratio: {c3}/{c_base if c_base > 0 else 1:.2f} = {c3/(c_base if c_base > 0 else 1):.2f}")
print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 833 DONE")
