"""
Step 760 - 674+running-mean on Split-CIFAR-100 (SOTA baseline).

Establishes 674 performance on the standard continual learning benchmark.
R1 mode: substrate gets images, picks actions (0-4), no labels or reward.
External judge measures accuracy = P(action % 5 == true_label).

Split: 20 tasks × 5 classes/task (500 images/task). Our split is more
fine-grained than published DER++/iCaRL (10 tasks × 10 classes).
Chance = 20% for our split. Published chance = 10% for their split.

Metrics:
- Per-task accuracy (should we get above 20% = chance for our split?)
- Avg accuracy across 20 tasks
- BWT (backward transfer = task0 accuracy before vs after other tasks)

Published reference (10-task, with labels+replay):
  DER++: 29.56%, iCaRL: 47.55%, chance: 10%
We're in R1 mode (no labels, no replay). Expect ~20% (chance).
Point: establish the floor. Self-organization alone achieves ~20%.
5 seeds, full 20-task protocol.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674
from substrates.chain import SplitCIFAR100Wrapper

print("=" * 65)
print("STEP 760 - 674 ON SPLIT-CIFAR-100")
print("=" * 65)

N_SEEDS = 5
N_IMAGES_PER_TASK = 500
PER_SEED_TIME = 60   # 60s per seed for full 20 tasks

wrapper = SplitCIFAR100Wrapper(n_images_per_task=N_IMAGES_PER_TASK,
                                per_seed_time=PER_SEED_TIME)

# Check data availability
if not wrapper._load():
    print("ERROR: CIFAR-100 data not available. Check torchvision installation.")
    print("Install: pip install torchvision")
    sys.exit(1)

print(f"CIFAR-100 loaded: {len(wrapper._data)} tasks, {N_IMAGES_PER_TASK} images/task")
print(f"Task structure: {wrapper.N_TASKS} tasks × {wrapper.CLASSES_PER_TASK} classes/task")
print(f"Chance = {1/wrapper.CLASSES_PER_TASK:.0%}")

all_results = []

for seed in range(N_SEEDS):
    print(f"\n-- Seed {seed} --")
    sub = TransitionTriggered674(n_actions=7, seed=seed)
    result = wrapper.run_seed(sub, seed=seed)

    if result.get("error"):
        print(f"  ERROR: {result['error']}")
    else:
        accs = result.get("task_accuracies", [])
        avg = result.get("avg_accuracy")
        bwt = result.get("backward_transfer")
        n_tasks = result.get("tasks_completed", 0)
        print(f"  tasks_completed={n_tasks}/20")
        if accs:
            print(f"  per-task acc: {[f'{a:.2f}' for a in accs]}")
        print(f"  avg_accuracy={avg}  BWT={bwt}  elapsed={result.get('elapsed')}s")

    all_results.append(result)

# Summary
valid = [r for r in all_results if r.get("avg_accuracy") is not None]
avg_accs = [r["avg_accuracy"] for r in valid]
bwts = [r["backward_transfer"] for r in valid if r.get("backward_transfer") is not None]

print("\n" + "=" * 65)
print("STEP 760 SUMMARY - 674 ON SPLIT-CIFAR-100")
print("=" * 65)
if avg_accs:
    print(f"Avg accuracy (mean over seeds): {float(np.mean(avg_accs)):.4f}")
    print(f"Avg accuracy (individual seeds): {[f'{a:.4f}' for a in avg_accs]}")
    if bwts:
        print(f"BWT (mean): {float(np.mean(bwts)):.4f}")
    print(f"")
    print(f"Chance (5 classes): 20.0%")
    print(f"Published DER++ (10-task, with labels+replay): 29.56%")
    print(f"Published iCaRL (10-task, with labels+replay): 47.55%")
    print(f"Note: Published results on 10-task split (10 classes, chance=10%).")
    print(f"Our split is harder (20 tasks, 5 classes). R1 mode (no labels).")
    print(f"This is the FLOOR — where self-organization alone lands.")
else:
    print("No valid results — CIFAR data issue.")
print("=" * 65)
print("STEP 760 DONE")
print("=" * 65)
