"""
Step 726 (C2): R3 dynamics — 674 on FT09.

R3 hypothesis: FT09 has fewer aliased cells (1-4) than LS20. R3 changes
should be smaller in magnitude and faster-stabilizing (aliasing detected
earlier, refinement completes sooner).

Same protocol as C1 (step725) but on FT09.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674, REFINE_EVERY
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 726 (C2) — R3 DYNAMICS PROFILE: 674 ON FT09")
print("=" * 65)
print(f"  REFINE_EVERY={REFINE_EVERY}")

SEED = 0
N_STEPS = 10_000
CHECKPOINT_EVERY = 1_000

try:
    try:
        import arcagi3
        env = arcagi3.make("FT09")
    except ImportError:
        import util_arcagi3
        env = util_arcagi3.make("FT09")

    n_valid = len(env._action_space) if hasattr(env, '_action_space') else 4
    print(f"  FT09 action space: {n_valid}")

    sub = TransitionTriggered674(n_actions=n_valid, seed=SEED)
    sub.reset(SEED)
    obs = env.reset(seed=SEED)
    level = 0
    l1_step = l2_step = None
    steps = 0
    fresh = True
    t_start = time.time()
    checkpoints = []
    prev_state = None

    while steps < N_STEPS and (time.time() - t_start) < 290:
        if obs is None:
            obs = env.reset(seed=SEED)
            sub.on_level_transition()
            fresh = True
            continue

        obs_arr = np.array(obs, dtype=np.float32)
        action = sub.process(obs_arr)
        obs, reward, done, info = env.step(action % n_valid)
        steps += 1

        if fresh:
            fresh = False
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1_step is None:
                l1_step = steps
            if cl == 2 and l2_step is None:
                l2_step = steps
            level = cl
            sub.on_level_transition()

        if done:
            obs = env.reset(seed=SEED)
            sub.on_level_transition()
            fresh = True

        if steps % CHECKPOINT_EVERY == 0:
            state = sub.get_state()
            if prev_state is not None:
                changed = [k for k in ["G_size", "aliased_count", "live_count", "ref_count"]
                           if state.get(k, 0) != prev_state.get(k, 0)]
            else:
                changed = []
            checkpoints.append({
                "step": steps,
                "aliased_count": state.get("aliased_count", 0),
                "live_count": state.get("live_count", 0),
                "G_size": state.get("G_size", 0),
                "ref_count": state.get("ref_count", 0),
                "changed_keys": changed,
            })
            prev_state = {k: state.get(k, 0)
                         for k in ["G_size", "aliased_count", "live_count", "ref_count"]}

    elapsed = time.time() - t_start
    print(f"\n  Completed: {steps} steps in {elapsed:.1f}s")
    print(f"  l1={l1_step} l2={l2_step} level={level}")

    print(f"\n  {'Step':>6} | {'Aliased':>8} | {'Live':>6} | {'G':>7} | {'Ref':>5} | Changed")
    print("  " + "-" * 65)
    for ckpt in checkpoints:
        print(f"  {ckpt['step']:>6} | {ckpt['aliased_count']:>8} | "
              f"{ckpt['live_count']:>6} | {ckpt['G_size']:>7} | {ckpt['ref_count']:>5} | "
              f"{ckpt['changed_keys']}")

    # Compare to LS20 hypothesis
    max_aliased = max((c["aliased_count"] for c in checkpoints), default=0)
    active_steps = [c["step"] for c in checkpoints if c["changed_keys"]]
    print(f"\n  FT09 max aliased cells: {max_aliased}")
    print(f"  R3 active checkpoints: {len(active_steps)}/{len(checkpoints)}")
    print(f"  Active steps: {active_steps}")

    # C1 hypothesis: FT09 aliased < LS20 aliased
    print(f"\n  C2 hypothesis check:")
    print(f"    FT09 max_aliased={max_aliased} (expected 1-4, less than LS20)")
    if max_aliased <= 4:
        print(f"    SUPPORT: FT09 aliasing ≤ 4 (hypothesis holds)")
    else:
        print(f"    REJECT: FT09 aliasing > 4 (hypothesis fails)")

    # Cross-check with judge
    print(f"\n  measure_r3_dynamics cross-check (2K, 10 ckpts)...")
    judge = ConstitutionalJudge()
    class _674_ft09(TransitionTriggered674):
        def __init__(self): super().__init__(n_actions=n_valid, seed=0)
    r3_check = judge.measure_r3_dynamics(_674_ft09, n_steps=2000, n_checkpoints=10)
    print(f"  R3 score: {r3_check.get('r3_dynamic_score')} profile: {r3_check.get('dynamics_profile')}")
    print(f"  Verified M: {r3_check.get('verified_M_elements')}")

except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 65)
print("STEP 726 DONE")
print("=" * 65)
