"""
Step 773 - Reversed SOTA chain: CIFAR-100 pretraining → LS20.

R3 hypothesis: Does CIFAR-100 classification experience help LS20 navigation?
Tests directionality of cross-domain transfer. Reversed protocol vs Step 770.

Step 770: LS20 (10K) → CIFAR-100. No transfer in accuracy (20.13% ≈ 20.21%).
Step 773: CIFAR-100 (task 0, 500 images) → LS20 (25s). Does CIFAR pretraining help?

Key: the substrate accumulates G structure from CIFAR image→action mappings.
When transitioning to LS20, CIFAR cells won't overlap LS20 navigation cells.
BUT: if CIFAR triggered refinements (ref dict), those hyperplanes persist and
may split LS20 cells more finely. Refinement is the only possible transfer channel.

Compare to 674 baseline on LS20: 10 seeds × 25s = ~7-8/10 (standard).
3 seeds × 25s on LS20 after CIFAR pretraining. Report L1 rate.

If L1 rate ≈ 7-8/10: CIFAR pretraining doesn't help LS20. Transfer is zero.
If L1 rate > 8/10: CIFAR image structure helps navigate LS20. Surprising.
If L1 rate < 7/10: CIFAR pretraining pollutes LS20 graph. Contamination.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674
from substrates.chain import SplitCIFAR100Wrapper

print("=" * 65)
print("STEP 773 - REVERSED CHAIN: CIFAR-100 → LS20")
print("=" * 65)


def _make_ls20():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3; return util_arcagi3.make("LS20")


N_SEEDS = 10        # More seeds to compare against 674 baseline properly
N_CIFAR_TASKS = 1   # Just task 0 as pretraining (500 images)
LS20_TIME = 25      # Same as 674 baseline (25s per seed)
N_IMAGES_PER_TASK = 500

wrapper = SplitCIFAR100Wrapper(n_images_per_task=N_IMAGES_PER_TASK, per_seed_time=60)
if not wrapper._load():
    print("ERROR: CIFAR-100 not available."); sys.exit(1)
print(f"CIFAR-100 loaded. Will use task 0 as pretraining ({N_IMAGES_PER_TASK} images).")

all_results = []

for seed in range(N_SEEDS):
    print(f"\n-- Seed {seed} --")
    sub = TransitionTriggered674(n_actions=7, seed=seed)
    sub.reset(seed)

    # Phase 1: CIFAR task 0 pretraining
    print(f"  Phase 1: CIFAR task 0 pretraining ({N_IMAGES_PER_TASK} images)...")
    try:
        # wrapper._data is a list of (images, labels) tuples, one per task
        task_images, task_labels = wrapper._data[0]
        n_img = min(N_IMAGES_PER_TASK, len(task_images))
        for i in range(n_img):
            obs = task_images[i].astype(np.float32) / 255.0
            sub.process(obs)
        state_cifar = sub.get_state()
        print(f"  CIFAR: live={state_cifar.get('live_count',0)} "
              f"ref={state_cifar.get('ref_count',0)} "
              f"G={state_cifar.get('G_size',0)}")
    except Exception as e:
        print(f"  CIFAR error: {e}")
        import traceback; traceback.print_exc()
        all_results.append({"seed": seed, "error": str(e)}); continue

    # Phase 2: LS20 (no reset)
    print(f"  Phase 2: LS20 ({LS20_TIME}s, no reset)...")
    try:
        env = _make_ls20()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        obs = env.reset(seed=seed * 100)
        l1_step = None; steps = 0; level = 0; fresh = True
        t_start = time.time()
        while (time.time() - t_start) < LS20_TIME:
            if obs is None:
                obs = env.reset(seed=seed * 100); sub.on_level_transition(); fresh = True; continue
            obs_arr = np.array(obs, dtype=np.float32)
            action = sub.process(obs_arr)
            obs, _, done, info = env.step(action % n_valid)
            steps += 1
            if fresh: fresh = False; continue
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                if cl == 1 and l1_step is None: l1_step = steps
                level = cl; sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed * 100); sub.on_level_transition(); fresh = True

        status = "L1" if l1_step else "  "
        print(f"  seed={seed} {status} steps={steps} l1={l1_step}")
        all_results.append({"seed": seed, "l1": l1_step, "steps": steps,
                            "cifar_ref": state_cifar.get('ref_count', 0)})
    except Exception as e:
        print(f"  LS20 error: {e}")
        all_results.append({"seed": seed, "error": str(e)})

l1_count = sum(1 for r in all_results if r.get("l1"))
valid = [r for r in all_results if "l1" in r]
verdict = "PASS" if l1_count >= 7 else ("KILL" if l1_count < 5 else "MARGINAL")

print("\n" + "=" * 65)
print("STEP 773 SUMMARY - REVERSED CHAIN: CIFAR → LS20")
print("=" * 65)
print(f"LS20 after CIFAR pretraining: L1={l1_count}/{N_SEEDS} {verdict}")
print(f"CIFAR ref_counts: {[r.get('cifar_ref', 0) for r in valid]}")
print(f"")
print(f"674 cold baseline (LS20, 25s): ~7-8/10 (from steps 752, 755, etc.)")
print(f"If L1 rate ≈ 7-8: CIFAR pretraining = zero transfer to LS20.")
print(f"If L1 rate > 8: CIFAR image structure helps LS20 navigation.")
print(f"If L1 rate < 7: CIFAR pretraining contaminates LS20 graph.")
print(f"")
print(f"Cross-domain transfer chain summary:")
print(f"  Step 770: LS20 → CIFAR: accuracy = 20.13% (cold: 20.21%) — zero transfer")
print(f"  Step 773: CIFAR → LS20: L1={l1_count}/{N_SEEDS} (cold: ~7-8/10)")
print("=" * 65)
print("STEP 773 DONE")
print("=" * 65)
