"""
Step 720: Baseline measurements — Split-CIFAR-100 + Dynamic R3.

1. Split-CIFAR-100: 674 substrate vs random baseline (1 seed, 50 img/task)
   - Per-task accuracy, cell growth after each task, backward transfer
   - NOTE: 674 has NO classification mechanism. Expected accuracy ≈ 1/5 = 20% (same as random).
     This is not a failure — it establishes the floor for R3 substrates on CIFAR.

2. Dynamic R3 with real LS20 data (1200 steps collected, 1000 used for measurement)
   - Which M elements actually change, when they first change
   - Dynamics profile (static / slow / fast / chaotic)
   - R3 dynamic score: fraction of declared-M elements that actually change

Published baselines for comparison (from literature):
  Split-CIFAR-100 avg accuracy:
    DER++:  49.8% (Buzzega et al. 2020)
    EWC:    33.2% (Kirkpatrick et al. 2017)
    iCaRL:  49.0% (Rebuffi et al. 2017)
    A-GEM:  36.1% (Chaudhry et al. 2019)
    Random: ~20%  (1/5 for 5-class tasks)
  Our substrate has no CL mechanism → expected ~20% = random floor.
  R3 substrate target: approach DER++ without replay buffer (novel).
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates import BaseSubstrate, ConstitutionalJudge
from substrates.step0674 import TransitionTriggered674
from substrates.chain import SplitCIFAR100Wrapper, ArcGameWrapper

print("=" * 65)
print("STEP 720 — SPLIT-CIFAR-100 + DYNAMIC R3 BASELINES")
print("=" * 65)


# -- Random baseline substrate --------------------------------------
class RandomSubstrate(BaseSubstrate):
    """Simplest possible baseline: uniform random actions."""
    def __init__(self, n_actions: int = 68, seed: int = 0):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self._t = 0

    def process(self, obs) -> int:
        self._t += 1
        return int(self._rng.randint(0, self._n_actions))

    def reset(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
        self._t = 0

    def get_state(self) -> dict:
        return {"t": self._t}

    def frozen_elements(self) -> list:
        return []

    @property
    def n_actions(self) -> int:
        return self._n_actions


# -- Custom CIFAR runner: also tracks cell growth -------------------
def run_cifar_with_cell_tracking(substrate, wrapper, seed=0):
    """Like SplitCIFAR100Wrapper.run_seed() but also records cell counts."""
    if not wrapper._load():
        return {"error": "CIFAR-100 not available"}

    rng = np.random.RandomState(seed)
    N_TASKS = 20
    CLASSES_PER_TASK = 5
    N_IMG = 50  # images per task (fast)

    task_accuracies = []
    task_cell_counts = []
    t_start = time.time()
    substrate.reset(seed)

    for task_id in range(N_TASKS):
        if (time.time() - t_start) >= 240:  # 4-min safety
            break

        task_images, task_labels = wrapper._data[task_id]
        idx = rng.choice(len(task_images), min(N_IMG, len(task_images)), replace=False)

        correct = 0
        for i in idx:
            obs = task_images[i].astype(np.float32) / 255.0
            action = substrate.process(obs) % CLASSES_PER_TASK
            correct += int(action == task_labels[i])

        acc = correct / len(idx)
        task_accuracies.append(round(acc, 4))

        # Record cell count from substrate state
        state = substrate.get_state()
        cell_count = state.get("live_count", state.get("G_size", state.get("t", "N/A")))
        task_cell_counts.append(cell_count)

        substrate.on_level_transition()

    # Backward transfer: re-eval task 0
    bwt = None
    if len(task_accuracies) >= 2:
        task0_images, task0_labels = wrapper._data[0]
        idx0 = rng.choice(len(task0_images), min(N_IMG, len(task0_images)), replace=False)
        correct0 = sum(
            int(substrate.process(task0_images[i].astype(np.float32) / 255.0) % CLASSES_PER_TASK
                == task0_labels[i])
            for i in idx0
        )
        task0_after = correct0 / len(idx0)
        bwt = round(task0_after - task_accuracies[0], 4)

    elapsed = time.time() - t_start
    return {
        "task_accuracies": task_accuracies,
        "task_cell_counts": task_cell_counts,
        "avg_accuracy": round(float(np.mean(task_accuracies)), 4) if task_accuracies else None,
        "backward_transfer": bwt,
        "tasks_completed": len(task_accuracies),
        "elapsed": round(elapsed, 2),
    }


# -- Part 1: Split-CIFAR-100 ----------------------------------------
print("\n-- PART 1: Split-CIFAR-100 --")
print("20 tasks × 5 classes × 50 images/task")

wrapper = SplitCIFAR100Wrapper(n_images_per_task=50)
wrapper._load()  # preload

if wrapper._data is None:
    print("  ERROR: CIFAR-100 not available (torchvision required)")
    cifar_674 = cifar_rnd = None
else:
    print("\n  Running 674 substrate...")
    sub674 = TransitionTriggered674(n_actions=5, seed=0)
    cifar_674 = run_cifar_with_cell_tracking(sub674, wrapper, seed=0)

    print("  Running random baseline...")
    sub_rnd = RandomSubstrate(n_actions=5, seed=0)
    cifar_rnd = run_cifar_with_cell_tracking(sub_rnd, wrapper, seed=0)

    print(f"\n  {'Task':>6} | {'674 acc':>8} | {'Random acc':>10} | {'674 cells':>10}")
    print("  " + "-" * 45)
    for i, (a674, arnd) in enumerate(
            zip(cifar_674["task_accuracies"], cifar_rnd["task_accuracies"])):
        cells = cifar_674["task_cell_counts"][i] if i < len(cifar_674["task_cell_counts"]) else "?"
        print(f"  {i:>6} | {a674:>8.3f} | {arnd:>10.3f} | {str(cells):>10}")

    print(f"\n  674 avg accuracy:    {cifar_674['avg_accuracy']:.4f}  ({cifar_674['elapsed']}s)")
    print(f"  Random avg accuracy: {cifar_rnd['avg_accuracy']:.4f}  ({cifar_rnd['elapsed']}s)")
    print(f"  674 backward transfer (BWT): {cifar_674['backward_transfer']}")
    print(f"  Random BWT:                  {cifar_rnd['backward_transfer']}")
    print(f"\n  Published baselines (Split-CIFAR-100 with replay/EWC):")
    print(f"    DER++: 49.8%  EWC: 33.2%  iCaRL: 49.0%  A-GEM: 36.1%  Random: ~20%")
    print(f"  674 is ~{cifar_674['avg_accuracy']:.0%} — expected (no CL mechanism)")


# -- Part 2: Dynamic R3 with real LS20 data -------------------------
print("\n-- PART 2: Dynamic R3 — real LS20 observations --")

try:
    try:
        import arcagi3
        ls20_env = arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        ls20_env = util_arcagi3.make("LS20")

    # Collect 1200 real LS20 frames using 674
    print("  Collecting 1200 real LS20 frames...")
    ls20_obs = []
    obs = ls20_env.reset(seed=0)
    n_valid = len(ls20_env._action_space) if hasattr(ls20_env, '_action_space') else 4
    sub_collect = TransitionTriggered674(n_actions=n_valid, seed=0)
    sub_collect.reset(0)
    fresh_episode = True
    steps_collected = 0

    t_collect_start = time.time()
    while steps_collected < 1200 and (time.time() - t_collect_start) < 90:
        if obs is None:
            obs = ls20_env.reset(seed=0)
            fresh_episode = True
            continue
        obs_arr = np.array(obs, dtype=np.float32)
        ls20_obs.append(obs_arr)
        action = sub_collect.process(obs_arr)
        obs, reward, done, info = ls20_env.step(action % n_valid)
        steps_collected += 1
        if fresh_episode:
            fresh_episode = False
        if done:
            obs = ls20_env.reset(seed=0)
            fresh_episode = True

    print(f"  Collected {len(ls20_obs)} frames in {time.time()-t_collect_start:.1f}s")

    # Run dynamic R3 measurement
    print(f"  Running measure_r3_dynamics (1000 steps, 10 checkpoints)...")
    judge = ConstitutionalJudge()
    t_r3_start = time.time()
    # Wrap TransitionTriggered674 with n_actions=4 for dynamic measurement
    class _674_4act(TransitionTriggered674):
        def __init__(self): super().__init__(n_actions=n_valid, seed=0)

    r3_result = judge.measure_r3_dynamics(
        _674_4act,
        obs_sequence=ls20_obs[:1000],
        n_steps=1000,
        n_checkpoints=10
    )
    t_r3_elapsed = time.time() - t_r3_start

    print(f"\n  R3 dynamic score: {r3_result.get('r3_dynamic_score', 'N/A'):.3f}")
    print(f"  Dynamics profile: {r3_result.get('dynamics_profile', 'N/A')}")
    print(f"  Elapsed: {t_r3_elapsed:.1f}s")

    # Which components changed and when
    change_times = r3_result.get("component_change_times", {})
    if change_times:
        print(f"\n  Component first-change times (out of 1000 steps):")
        for comp, step in sorted(change_times.items(), key=lambda x: x[1]):
            print(f"    {comp:<30} first changed at step {step}")
    else:
        print(f"\n  No components changed (static profile)")

    # Checkpoint summary
    print(f"\n  Checkpoints (changed_keys per interval):")
    for ckpt in r3_result.get("checkpoints", []):
        changed = ckpt.get("changed_from_prev", [])
        mags = ckpt.get("change_magnitudes", {})
        mag_str = " ".join(f"{k}:{v:.2f}" for k, v in list(mags.items())[:3])
        print(f"    step={ckpt['step']:>5}: changed={changed[:4]} {mag_str}")

    # M element verification
    print(f"\n  M-element verification:")
    print(f"  Declared M elements: edge_count_update, aliased_set, ref_hyperplanes")
    print(f"  Actually-changed keys: {sorted(change_times.keys())}")

except Exception as e:
    import traceback
    print(f"  ERROR in Dynamic R3: {e}")
    traceback.print_exc()


# -- Summary ---------------------------------------------------------
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
if cifar_674:
    print(f"Split-CIFAR-100 674:    avg_acc={cifar_674['avg_accuracy']:.3f}  BWT={cifar_674['backward_transfer']}")
    print(f"Split-CIFAR-100 random: avg_acc={cifar_rnd['avg_accuracy']:.3f}  BWT={cifar_rnd['backward_transfer']}")
    print(f"Gap vs DER++: {0.498 - cifar_674['avg_accuracy']:.3f} (DER++ uses replay, we don't)")
if 'r3_result' in dir() and r3_result:
    print(f"Dynamic R3 score: {r3_result.get('r3_dynamic_score', 'N/A')}")
    print(f"Dynamics profile: {r3_result.get('dynamics_profile', 'N/A')}")
print("=" * 65)
