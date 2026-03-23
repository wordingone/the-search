"""
Step 725 (C1): R3 dynamics profile — 674 on LS20 (1000-step resolution).

R3 hypothesis: refinement produces R3 concentrated in first 5K steps, near-zero
after aliased cells stabilize. Once all ambiguous cells are detected and refined,
the system's M elements stop changing.

Note: Leo spec says 25K steps. Jun directive caps LS20 at 10K steps (signal
shows by 10K or it doesn't exist). Running 10K with 10 checkpoints at 1000-step
intervals. This is sufficient to test the 5K concentration hypothesis.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674, REFINE_EVERY
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 725 (C1) — R3 DYNAMICS PROFILE: 674 ON LS20")
print("=" * 65)
print(f"  REFINE_EVERY={REFINE_EVERY} (refinement fires at multiples of this)")

SEED = 0
N_STEPS = 10_000
CHECKPOINT_EVERY = 1_000

try:
    try:
        import arcagi3
        env = arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        env = util_arcagi3.make("LS20")

    n_valid = len(env._action_space) if hasattr(env, '_action_space') else 4
    print(f"  LS20 action space: {n_valid}")
    print(f"  Running {N_STEPS} steps with checkpoints every {CHECKPOINT_EVERY}...")

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
            # Compute R3 change rate from previous checkpoint
            if prev_state is not None:
                changed = []
                for k in ["G_size", "aliased_count", "live_count", "ref_count"]:
                    if state.get(k, 0) != prev_state.get(k, 0):
                        changed.append(k)
                r3_active = len(changed) > 0
            else:
                changed = []
                r3_active = False

            checkpoints.append({
                "step": steps,
                "aliased_count": state.get("aliased_count", 0),
                "live_count": state.get("live_count", 0),
                "G_size": state.get("G_size", 0),
                "ref_count": state.get("ref_count", 0),
                "changed_keys": changed,
                "r3_active": r3_active,
            })
            prev_state = {k: state.get(k, 0)
                         for k in ["G_size", "aliased_count", "live_count", "ref_count"]}

    elapsed = time.time() - t_start
    print(f"\n  Completed: {steps} steps in {elapsed:.1f}s")
    print(f"  l1={l1_step} l2={l2_step} level={level}")

    # Print checkpoint table
    print(f"\n  {'Step':>6} | {'Aliased':>8} | {'Live':>6} | {'G':>7} | {'Ref':>5} | {'R3_active':>10} | Changed")
    print("  " + "-" * 75)
    for ckpt in checkpoints:
        print(f"  {ckpt['step']:>6} | {ckpt['aliased_count']:>8} | "
              f"{ckpt['live_count']:>6} | {ckpt['G_size']:>7} | {ckpt['ref_count']:>5} | "
              f"{'YES' if ckpt['r3_active'] else 'no':>10} | {ckpt['changed_keys']}")

    # R3 concentration analysis
    active_steps = [c["step"] for c in checkpoints if c["r3_active"]]
    early_active = [s for s in active_steps if s <= 5000]
    late_active = [s for s in active_steps if s > 5000]
    print(f"\n  R3 active checkpoints: {len(active_steps)}/{len(checkpoints)}")
    print(f"  Early (<=5K): {early_active}")
    print(f"  Late (>5K):   {late_active}")

    if late_active:
        print(f"  HYPOTHESIS REJECT: R3 changes continue past 5K steps")
    elif early_active:
        print(f"  HYPOTHESIS SUPPORT: R3 concentrated in first 5K steps")
    else:
        print(f"  NOTE: No R3 changes detected (static profile)")

    # Full measure_r3_dynamics for cross-check
    print(f"\n  Running measure_r3_dynamics (2K steps, 10 ckpts) for cross-check...")
    judge = ConstitutionalJudge()
    class _674_ls20(TransitionTriggered674):
        def __init__(self): super().__init__(n_actions=n_valid, seed=0)

    r3_check = judge.measure_r3_dynamics(_674_ls20, n_steps=2000, n_checkpoints=10)
    print(f"  R3 dynamic score: {r3_check.get('r3_dynamic_score')}")
    print(f"  Profile: {r3_check.get('dynamics_profile')}")
    print(f"  Verified M: {r3_check.get('verified_M_elements')}")

except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 65)
print("STEP 725 DONE")
print("=" * 65)
