"""
Step 770 - SOTA chain: LS20 pretraining → Split-CIFAR-100.

R3 hypothesis: 674's M elements (ref dict, G, aliased) accumulate during LS20
navigation and persist as the substrate transitions to CIFAR classification.
Refinement hyperplanes (ref dict) created during LS20 apply globally — if a
CIFAR image falls in a cell that was refined during LS20, the finer split is
used. This is passive cross-domain structure transfer.

Protocol:
1. Run 674 on LS20 for 10K steps (pre-train representation)
2. Without resetting substrate, run on Split-CIFAR-100 (20 tasks × 500 images)
3. Compare: Step 760 (cold start 674 on CIFAR): 20.21%, BWT=5.6%

Key metrics:
- ref_count before CIFAR (how many refinements from LS20 persist)
- accuracy vs step760 baseline (cross-domain transfer?)
- BWT with vs without LS20 pretraining

If accuracy > 20.21%: LS20 navigation structure helps CIFAR classification.
If accuracy ≈ 20.21%: refinements don't overlap with CIFAR cell structure.
If accuracy < 20.21%: LS20 navigation corrupts CIFAR representation.

3 seeds. 10K LS20 steps per seed. Full 20-task CIFAR protocol.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674
from substrates.chain import SplitCIFAR100Wrapper

print("=" * 65)
print("STEP 770 - SOTA CHAIN: LS20 → CIFAR-100 CROSS-DOMAIN")
print("=" * 65)

N_SEEDS = 3
LS20_STEPS = 10_000
N_IMAGES_PER_TASK = 500
PER_SEED_TIME = 60


def _make_ls20():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("LS20")


wrapper = SplitCIFAR100Wrapper(n_images_per_task=N_IMAGES_PER_TASK,
                                per_seed_time=PER_SEED_TIME)
if not wrapper._load():
    print("ERROR: CIFAR-100 not available.")
    sys.exit(1)

print(f"CIFAR-100 loaded: {wrapper.N_TASKS} tasks × {wrapper.CLASSES_PER_TASK} classes/task")

all_results = []

for seed in range(N_SEEDS):
    print(f"\n-- Seed {seed} --")
    sub = TransitionTriggered674(n_actions=7, seed=seed)
    sub.reset(seed)

    # Phase 1: LS20 pretraining
    print(f"  Phase 1: LS20 pretraining ({LS20_STEPS} steps)...")
    t_ls20 = time.time()
    try:
        env = _make_ls20()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        obs = env.reset(seed=seed * 100)
        steps = 0
        while steps < LS20_STEPS:
            if obs is None:
                obs = env.reset(seed=seed * 100)
                sub.on_level_transition()
                continue
            obs_arr = np.array(obs, dtype=np.float32)
            action = sub.process(obs_arr)
            obs, reward, done, info = env.step(action % n_valid)
            steps += 1
            if done:
                obs = env.reset(seed=seed * 100)
                sub.on_level_transition()
        state_before = sub.get_state()
        print(f"  LS20 done: {steps} steps in {time.time()-t_ls20:.1f}s")
        print(f"  State: live={state_before.get('live_count', 0)} "
              f"G={state_before.get('G_size', 0)} "
              f"ref={state_before.get('ref_count', 0)} "
              f"aliased={state_before.get('aliased_count', 0)}")
    except Exception as e:
        print(f"  LS20 error: {e}")
        import traceback; traceback.print_exc()
        all_results.append({"seed": seed, "error": str(e), "phase": "ls20"})
        continue

    # Phase 2: CIFAR classification (substrate NOT reset)
    print(f"  Phase 2: CIFAR classification (no substrate reset)...")
    result = wrapper.run_seed(sub, seed=seed)

    if result.get("error"):
        print(f"  CIFAR ERROR: {result['error']}")
    else:
        accs = result.get("task_accuracies", [])
        avg = result.get("avg_accuracy")
        bwt = result.get("backward_transfer")
        state_after = sub.get_state()
        print(f"  tasks_completed={result.get('tasks_completed')}/20")
        if accs:
            print(f"  per-task acc: {[f'{a:.2f}' for a in accs]}")
        print(f"  avg_accuracy={avg}  BWT={bwt}  elapsed={result.get('elapsed')}s")
        print(f"  ref_count after CIFAR: {state_after.get('ref_count', 0)}")

    result["seed"] = seed
    result["ls20_ref_before"] = state_before.get("ref_count", 0)
    result["ls20_live_before"] = state_before.get("live_count", 0)
    all_results.append(result)

valid = [r for r in all_results if r.get("avg_accuracy") is not None]
avg_accs = [r["avg_accuracy"] for r in valid]
bwts = [r["backward_transfer"] for r in valid if r.get("backward_transfer") is not None]
ref_counts = [r.get("ls20_ref_before", 0) for r in valid]

print("\n" + "=" * 65)
print("STEP 770 SUMMARY - LS20 → CIFAR CROSS-DOMAIN CHAIN")
print("=" * 65)
if avg_accs:
    print(f"Avg accuracy (mean over seeds): {float(np.mean(avg_accs)):.4f}")
    print(f"Individual seeds: {[f'{a:.4f}' for a in avg_accs]}")
    if bwts:
        print(f"BWT (mean): {float(np.mean(bwts)):.4f}")
    print(f"LS20 ref_count before CIFAR: {ref_counts}")
    print(f"")
    print(f"Baseline (Step 760 — cold start 674 on CIFAR): 20.21%, BWT=5.6%")
    print(f"Step 762 (D1+D3 cold start):                  19.65%, BWT=1.4%")
    print(f"")
    print(f"If accuracy > 20.21%: LS20 nav structure transfers to CIFAR.")
    print(f"If accuracy ≈ 20.21%: cross-domain structure transfer is zero.")
    print(f"If accuracy < 20.21%: LS20 contamination.")
    print(f"BWT change indicates whether nav experience affected memory retention.")
else:
    print("No valid results.")
print("=" * 65)
print("STEP 770 DONE")
print("=" * 65)
