"""
Step 761 - PlainLSH on Split-CIFAR-100 (control baseline).

Control experiment: plain k=12 LSH, no refinement, no aliasing, no fine graph.
Same protocol as Step 760 (674). Shows what plain hashing achieves without
674's refinement mechanism.

Expected: similar to 674 (~20% chance). If lower: refinement helps marginally.
If higher: 674's refinement actually hurts on CIFAR (unlikely).
5 seeds, 20-task protocol.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.plain_lsh import PlainLSH
from substrates.chain import SplitCIFAR100Wrapper

print("=" * 65)
print("STEP 761 - PLAIN LSH ON SPLIT-CIFAR-100")
print("=" * 65)

N_SEEDS = 5
N_IMAGES_PER_TASK = 500
PER_SEED_TIME = 60

wrapper = SplitCIFAR100Wrapper(n_images_per_task=N_IMAGES_PER_TASK,
                                per_seed_time=PER_SEED_TIME)

if not wrapper._load():
    print("ERROR: CIFAR-100 not available.")
    sys.exit(1)

print(f"CIFAR-100 loaded: {wrapper.N_TASKS} tasks × {wrapper.CLASSES_PER_TASK} classes/task")

all_results = []

for seed in range(N_SEEDS):
    print(f"\n-- Seed {seed} --")
    sub = PlainLSH(n_actions=7, seed=seed)
    result = wrapper.run_seed(sub, seed=seed)

    if result.get("error"):
        print(f"  ERROR: {result['error']}")
    else:
        accs = result.get("task_accuracies", [])
        avg = result.get("avg_accuracy")
        bwt = result.get("backward_transfer")
        print(f"  tasks_completed={result.get('tasks_completed')}/20")
        if accs:
            print(f"  per-task acc: {[f'{a:.2f}' for a in accs]}")
        print(f"  avg_accuracy={avg}  BWT={bwt}  elapsed={result.get('elapsed')}s")
    all_results.append(result)

valid = [r for r in all_results if r.get("avg_accuracy") is not None]
avg_accs = [r["avg_accuracy"] for r in valid]
bwts = [r["backward_transfer"] for r in valid if r.get("backward_transfer") is not None]

print("\n" + "=" * 65)
print("STEP 761 SUMMARY - PLAIN LSH ON SPLIT-CIFAR-100")
print("=" * 65)
if avg_accs:
    print(f"Avg accuracy (mean over seeds): {float(np.mean(avg_accs)):.4f}")
    print(f"Individual seeds: {[f'{a:.4f}' for a in avg_accs]}")
    if bwts:
        print(f"BWT (mean): {float(np.mean(bwts)):.4f}")
    print(f"Chance: 20.0% | Step 760 (674): 20.21%")
    print(f"If PlainLSH ≈ 674: refinement doesn't help CIFAR accuracy.")
    print(f"If PlainLSH < 674: refinement improves clustering marginally.")
else:
    print("No valid results.")
print("=" * 65)
print("STEP 761 DONE")
print("=" * 65)
