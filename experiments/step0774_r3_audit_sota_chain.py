"""
Step 774 - R3 audit on SOTA chain (LS20 → CIFAR domain transition).

R3 hypothesis: The substrate's M element change rates shift at domain boundaries.
During LS20 navigation: high G growth, high aliasing, some refinement.
During CIFAR classification: low G growth (each image seen once), no aliasing.

R3 audit measures:
- dG/dt: transition graph growth rate (steps per new edge)
- d_live/dt: new cell discovery rate
- d_aliased/dt: aliasing rate (aliased cells / live cells)
- d_ref/dt: refinement rate (new refinements per 1K steps)

If rates change at LS20 → CIFAR boundary: R3 is detecting domain change.
If rates stay constant: domain transition is transparent to substrate.

Logs state every 1K LS20 steps and every 100 CIFAR images.
Compare rates in LS20 phase vs CIFAR phase vs boundary window.
3 seeds for reproducibility.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674
from substrates.chain import SplitCIFAR100Wrapper

print("=" * 65)
print("STEP 774 - R3 AUDIT: LS20 → CIFAR DOMAIN TRANSITION")
print("=" * 65)


def _make_ls20():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3; return util_arcagi3.make("LS20")


N_SEEDS = 3
LS20_STEPS = 10_000
LOG_EVERY_LS20 = 1_000    # Log state every 1K LS20 steps
LOG_EVERY_CIFAR = 100     # Log state every 100 CIFAR images
N_IMAGES_PER_TASK = 500

wrapper = SplitCIFAR100Wrapper(n_images_per_task=N_IMAGES_PER_TASK, per_seed_time=60)
if not wrapper._load():
    print("ERROR: CIFAR-100 not available."); sys.exit(1)
print(f"CIFAR-100 loaded: {wrapper.N_TASKS} tasks × {wrapper.CLASSES_PER_TASK} classes/task")


def get_metrics(sub):
    state = sub.get_state()
    return {
        "G": state.get("G_size", 0),
        "live": state.get("live_count", 0),
        "aliased": state.get("aliased_count", 0),
        "ref": state.get("ref_count", 0),
    }


all_results = []

for seed in range(N_SEEDS):
    print(f"\n-- Seed {seed} --")
    sub = TransitionTriggered674(n_actions=7, seed=seed)
    sub.reset(seed)

    ls20_timeline = []  # (step, G, live, aliased, ref)
    cifar_timeline = [] # (img_num, G, live, aliased, ref, acc)

    # Phase 1: LS20
    print(f"  LS20 phase ({LS20_STEPS} steps, logging every {LOG_EVERY_LS20})...")
    try:
        env = _make_ls20()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        obs = env.reset(seed=seed * 100); steps = 0
        m0 = get_metrics(sub)
        ls20_timeline.append((0, m0))

        while steps < LS20_STEPS:
            if obs is None:
                obs = env.reset(seed=seed * 100); sub.on_level_transition(); continue
            action = sub.process(np.array(obs, dtype=np.float32))
            obs, _, done, _ = env.step(action % n_valid); steps += 1
            if done:
                obs = env.reset(seed=seed * 100); sub.on_level_transition()
            if steps % LOG_EVERY_LS20 == 0:
                m = get_metrics(sub)
                ls20_timeline.append((steps, m))

        m_end = get_metrics(sub)
        print(f"  LS20 end: G={m_end['G']} live={m_end['live']} "
              f"aliased={m_end['aliased']} ref={m_end['ref']}")
    except Exception as e:
        print(f"  LS20 error: {e}")
        import traceback; traceback.print_exc()
        continue

    # Compute LS20 rates
    if len(ls20_timeline) >= 2:
        g_start = ls20_timeline[0][1]["G"]; g_end = ls20_timeline[-1][1]["G"]
        live_start = ls20_timeline[0][1]["live"]; live_end = ls20_timeline[-1][1]["live"]
        ali_rate = m_end['aliased'] / max(m_end['live'], 1)
        g_rate_ls20 = (g_end - g_start) / LS20_STEPS
        live_rate_ls20 = (live_end - live_start) / LS20_STEPS
        print(f"  LS20 rates: dG/step={g_rate_ls20:.3f} d_live/step={live_rate_ls20:.4f} "
              f"aliased_frac={ali_rate:.3f}")

    # Phase 2: CIFAR (no reset, but using wrapper's data directly)
    print(f"  CIFAR phase ({wrapper.N_TASKS} tasks, logging every {LOG_EVERY_CIFAR} images)...")
    rng = np.random.RandomState(seed)
    total_correct = 0; total_images = 0
    task_accuracies = []
    img_counter = 0
    m_cifar_start = get_metrics(sub)
    cifar_timeline.append((0, m_cifar_start, None))

    for task_id in range(wrapper.N_TASKS):
        task_images, task_labels = wrapper._data[task_id]
        idx = rng.choice(len(task_images), min(N_IMAGES_PER_TASK, len(task_images)), replace=False)
        correct = 0
        for i in idx:
            obs = task_images[i].astype(np.float32) / 255.0
            action = sub.process(obs)
            correct += int(action == task_labels[i])
            img_counter += 1
            if img_counter % LOG_EVERY_CIFAR == 0:
                m = get_metrics(sub)
                cifar_timeline.append((img_counter, m, None))
        task_acc = correct / len(idx)
        task_accuracies.append(task_acc)
        sub.on_level_transition()

    # Backward transfer: re-evaluate task 0
    task0_images, task0_labels = wrapper._data[0]
    idx0 = rng.choice(len(task0_images), min(N_IMAGES_PER_TASK, len(task0_labels)), replace=False)
    t0_correct = sum(1 for i in idx0 if sub.process(task0_images[i].astype(np.float32)/255.0) == task0_labels[i])
    bwt = task0_labels[idx0[0]] if False else round(t0_correct/len(idx0) - task_accuracies[0], 4)

    avg_acc = float(np.mean(task_accuracies))
    m_cifar_end = get_metrics(sub)
    g_rate_cifar = (m_cifar_end['G'] - m_cifar_start['G']) / max(img_counter, 1)
    live_rate_cifar = (m_cifar_end['live'] - m_cifar_start['live']) / max(img_counter, 1)
    print(f"  CIFAR: avg_acc={avg_acc:.4f} BWT={bwt}")
    print(f"  CIFAR rates: dG/img={g_rate_cifar:.3f} d_live/img={live_rate_cifar:.4f}")
    print(f"  R3 transition: G {m_end['G']}→{m_cifar_end['G']} "
          f"ref {m_end['ref']}→{m_cifar_end['ref']}")

    # Rate ratio: LS20 vs CIFAR
    rate_ratio_g = g_rate_cifar / max(g_rate_ls20, 1e-8)
    rate_ratio_live = live_rate_cifar / max(live_rate_ls20, 1e-8)
    print(f"  Rate change at transition: G×{rate_ratio_g:.2f} live×{rate_ratio_live:.2f}")
    print(f"  (>1: CIFAR more active; <1: CIFAR less active than LS20)")

    all_results.append({
        "seed": seed,
        "avg_acc": avg_acc, "bwt": bwt,
        "g_rate_ls20": g_rate_ls20, "live_rate_ls20": live_rate_ls20,
        "g_rate_cifar": g_rate_cifar, "live_rate_cifar": live_rate_cifar,
        "rate_ratio_g": rate_ratio_g, "rate_ratio_live": rate_ratio_live,
        "ref_ls20": m_end['ref'], "ref_cifar": m_cifar_end['ref'],
        "ls20_timeline": ls20_timeline,
    })

print("\n" + "=" * 65)
print("STEP 774 SUMMARY - R3 AUDIT: SOTA CHAIN TRANSITION")
print("=" * 65)
valid = [r for r in all_results if "avg_acc" in r]
if valid:
    accs = [r["avg_acc"] for r in valid]
    bwts = [r["bwt"] for r in valid]
    g_ratios = [r["rate_ratio_g"] for r in valid]
    live_ratios = [r["rate_ratio_live"] for r in valid]
    print(f"CIFAR avg accuracy: {float(np.mean(accs)):.4f} (step760 cold: 20.21%)")
    print(f"BWT: {float(np.mean(bwts)):.4f}")
    print(f"")
    print(f"R3 DYNAMICS AT LS20 → CIFAR TRANSITION:")
    print(f"  G growth rate:    LS20={float(np.mean([r['g_rate_ls20'] for r in valid])):.3f}/step  "
          f"CIFAR={float(np.mean([r['g_rate_cifar'] for r in valid])):.3f}/img")
    print(f"  live growth rate: LS20={float(np.mean([r['live_rate_ls20'] for r in valid])):.4f}/step  "
          f"CIFAR={float(np.mean([r['live_rate_cifar'] for r in valid])):.4f}/img")
    print(f"  Rate ratio (CIFAR/LS20): G×{float(np.mean(g_ratios)):.2f} live×{float(np.mean(live_ratios)):.2f}")
    print(f"  ref_count: LS20={float(np.mean([r['ref_ls20'] for r in valid])):.1f} "
          f"→ CIFAR={float(np.mean([r['ref_cifar'] for r in valid])):.1f}")
    print(f"")
    print(f"If G rate ratio >> 1: CIFAR generates more transitions per step → different dynamics.")
    print(f"If rate ratio ≈ 1: domain boundary is invisible to the substrate.")
    print(f"This is the R3 audit: does the substrate DETECT the domain change?")
else:
    print("No valid results.")
print("=" * 65)
print("STEP 774 DONE")
print("=" * 65)
