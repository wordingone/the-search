"""
Step 721 (A2): 674+running-mean Split-CIFAR-100. Per-task NMI, forgetting, aliasing.

R3 hypothesis: greyscale encoding loses color. D1 (color-aware encoding) should
improve clustering NMI if color is diagnostic for CIFAR classes. This experiment
measures the BASELINE NMI with current channel-0-only encoding.

Metrics per task:
  - Accuracy (action%5 == label)
  - NMI(true_labels, actions%5) — measures clustering quality
  - aliased_count — how many cells have ambiguous transitions at task end
  - live_count — number of active cells
  - nodes_per_task — new cells created during this task

Published Split-CIFAR-100:
  DER++: 49.8%  EWC: 33.2%  Random: 20%
  674 (no CL mechanism): expected ~20% (random floor)
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from substrates.step0674 import TransitionTriggered674
from substrates.chain import SplitCIFAR100Wrapper

print("=" * 65)
print("STEP 721 (A2) — SPLIT-CIFAR-100 NMI + FORGETTING + ALIASING")
print("=" * 65)


def nmi_score(true_labels, pred_labels, n_classes=5):
    """Normalized Mutual Information (symmetric)."""
    n = len(true_labels)
    if n == 0:
        return 0.0
    joint = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(true_labels, pred_labels):
        joint[int(t) % n_classes][int(p) % n_classes] += 1
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    def h(probs):
        return -sum(float(p) * np.log2(float(p) + 1e-15)
                    for p in probs if p > 1e-15)

    hx = h(px)
    hy = h(py)
    hxy = -sum(float(joint[i, j]) * np.log2(float(joint[i, j]) + 1e-15)
               for i in range(n_classes) for j in range(n_classes)
               if joint[i, j] > 1e-15)
    mi = hx + hy - hxy
    denom = (hx + hy) / 2
    return mi / denom if denom > 1e-12 else 0.0


N_TASKS = 20
CLASSES_PER_TASK = 5
N_IMG = 500
SEED = 0

wrapper = SplitCIFAR100Wrapper(n_images_per_task=N_IMG)
if not wrapper._load():
    print("ERROR: CIFAR-100 not available. Install torchvision.")
    sys.exit(1)

print(f"  CIFAR-100 loaded. Running 674 on {N_TASKS} tasks x {N_IMG} images...")

sub = TransitionTriggered674(n_actions=5, seed=SEED)
sub.reset(SEED)
rng = np.random.RandomState(SEED)

task_results = []
t_start = time.time()

for task_id in range(N_TASKS):
    if (time.time() - t_start) >= 280:
        print(f"  WARNING: time limit hit at task {task_id}")
        break

    task_images, task_labels = wrapper._data[task_id]
    live_before = sub.get_state().get("live_count", 0)

    idx = rng.choice(len(task_images), min(N_IMG, len(task_images)), replace=False)
    true_labels_task = []
    pred_labels_task = []

    for i in idx:
        obs = task_images[i].astype(np.float32) / 255.0
        action = sub.process(obs)
        true_labels_task.append(int(task_labels[i]))
        pred_labels_task.append(action % CLASSES_PER_TASK)

    acc = sum(t == p for t, p in zip(true_labels_task, pred_labels_task)) / len(idx)
    nmi = nmi_score(true_labels_task, pred_labels_task)

    state = sub.get_state()
    live_after = state.get("live_count", 0)
    aliased = state.get("aliased_count", 0)

    task_results.append({
        "task": task_id,
        "acc": round(acc, 4),
        "nmi": round(nmi, 4),
        "live_before": live_before,
        "live_after": live_after,
        "new_cells": live_after - live_before,
        "aliased": aliased,
        "G_size": state.get("G_size", 0),
    })

    sub.on_level_transition()

# Backward transfer: re-eval task 0 after all tasks
task0_images, task0_labels = wrapper._data[0]
idx0 = rng.choice(len(task0_images), min(N_IMG, len(task0_images)), replace=False)
correct0 = sum(
    int(sub.process(task0_images[i].astype(np.float32) / 255.0) % CLASSES_PER_TASK
        == int(task0_labels[i]))
    for i in idx0
)
task0_after_acc = correct0 / len(idx0)
bwt = task0_after_acc - task_results[0]["acc"]

elapsed = time.time() - t_start

# --- Print results ---
print(f"\n  {'Task':>5} | {'Acc':>6} | {'NMI':>6} | {'NewCells':>9} | {'Aliased':>8} | {'G_size':>7}")
print("  " + "-" * 55)
for r in task_results:
    print(f"  {r['task']:>5} | {r['acc']:>6.3f} | {r['nmi']:>6.4f} | "
          f"{r['new_cells']:>9} | {r['aliased']:>8} | {r['G_size']:>7}")

avg_acc = np.mean([r["acc"] for r in task_results])
avg_nmi = np.mean([r["nmi"] for r in task_results])

print(f"\n  Tasks completed: {len(task_results)}")
print(f"  Avg accuracy:    {avg_acc:.4f}  (random floor: 0.200)")
print(f"  Avg NMI:         {avg_nmi:.4f}  (random baseline ~0.0)")
print(f"  Backward transfer (BWT): {bwt:.4f}  (task0: {task_results[0]['acc']:.3f} -> {task0_after_acc:.3f})")
print(f"  Elapsed: {elapsed:.1f}s")

# Aliasing profile
aliased_counts = [r["aliased"] for r in task_results]
print(f"\n  Aliasing profile: {aliased_counts}")
print(f"  Aliased cells at end: {task_results[-1]['aliased']}")

# Cell growth profile
new_cells = [r["new_cells"] for r in task_results]
print(f"\n  New cells per task: {new_cells}")
print(f"  Total live cells at end: {task_results[-1]['live_after']}")

print("\n  Published Split-CIFAR-100 baselines:")
print(f"    DER++: 49.8%  EWC: 33.2%  iCaRL: 49.0%  Random: ~20%")
print(f"    674 (no CL): {avg_acc:.1%} — {'at random floor' if avg_acc < 0.22 else 'above random floor'}")

if avg_nmi > 0.05:
    print(f"\n  NMI={avg_nmi:.4f} > 0.05: encoding has structure — D1 may help")
else:
    print(f"\n  NMI={avg_nmi:.4f} ≈ 0: encoding is random w.r.t. CIFAR classes — D1 likely needed")

print("\n" + "=" * 65)
print("STEP 721 DONE")
print("=" * 65)
