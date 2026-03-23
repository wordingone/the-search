"""
Step 772 - SOTA chain with PlainLSH: LS20 pretraining → Split-CIFAR-100.

Control variant of Step 770 using PlainLSH (no refinement, no aliasing, no fine graph).
Same protocol: 10K LS20 steps → CIFAR-100 (no substrate reset).

R3 hypothesis: With no M elements (no refinement, no aliasing), does LS20 pretraining
still help CIFAR? Tests whether cross-domain transfer requires refinement or whether
simple graph accumulation is enough.

Compare:
- Step 770 (674 LS20→CIFAR): 20.13%, BWT=6.5%
- Step 761 (PlainLSH cold CIFAR): 20.04%, BWT=4.3%

If PlainLSH chain ≈ PlainLSH cold: cross-domain transfer requires refinement (M elements).
If PlainLSH chain > PlainLSH cold: simple graph accumulation enables cross-domain transfer.

3 seeds, same protocol as steps 770-771.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.plain_lsh import PlainLSH
from substrates.chain import SplitCIFAR100Wrapper

print("=" * 65)
print("STEP 772 - PLAIN LSH SOTA CHAIN: LS20 → CIFAR-100")
print("=" * 65)


def _make_ls20():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3; return util_arcagi3.make("LS20")


N_SEEDS = 3; LS20_STEPS = 10_000; N_IMAGES_PER_TASK = 500; PER_SEED_TIME = 60

wrapper = SplitCIFAR100Wrapper(n_images_per_task=N_IMAGES_PER_TASK, per_seed_time=PER_SEED_TIME)
if not wrapper._load():
    print("ERROR: CIFAR-100 not available."); sys.exit(1)
print(f"CIFAR-100 loaded: {wrapper.N_TASKS} tasks × {wrapper.CLASSES_PER_TASK} classes/task")

all_results = []

for seed in range(N_SEEDS):
    print(f"\n-- Seed {seed} --")
    sub = PlainLSH(n_actions=7, seed=seed)

    # Phase 1: LS20
    print(f"  Phase 1: LS20 pretraining ({LS20_STEPS} steps)...")
    try:
        env = _make_ls20()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        obs = env.reset(seed=seed * 100); steps = 0
        while steps < LS20_STEPS:
            if obs is None:
                obs = env.reset(seed=seed * 100)
                if hasattr(sub, 'on_level_transition'): sub.on_level_transition()
                continue
            action = sub.process(np.array(obs, dtype=np.float32))
            obs, _, done, _ = env.step(action % n_valid); steps += 1
            if done:
                obs = env.reset(seed=seed * 100)
                if hasattr(sub, 'on_level_transition'): sub.on_level_transition()
        # Get state before CIFAR
        state_ls20 = {}
        if hasattr(sub, 'get_state'): state_ls20 = sub.get_state()
        elif hasattr(sub, 'G'): state_ls20 = {"G_size": len(sub.G), "live": len(getattr(sub, 'live', set()))}
        print(f"  LS20: {steps} steps, state={state_ls20}")
    except Exception as e:
        print(f"  LS20 error: {e}")
        all_results.append({"seed": seed, "error": str(e)}); continue

    # Phase 2: CIFAR (no reset)
    print(f"  Phase 2: CIFAR (no reset)...")
    result = wrapper.run_seed(sub, seed=seed)
    if result.get("error"):
        print(f"  CIFAR ERROR: {result['error']}")
    else:
        print(f"  tasks={result.get('tasks_completed')}/20 avg={result.get('avg_accuracy')} BWT={result.get('backward_transfer')}")
    result["seed"] = seed
    all_results.append(result)

valid = [r for r in all_results if r.get("avg_accuracy") is not None]
avg_accs = [r["avg_accuracy"] for r in valid]
bwts = [r["backward_transfer"] for r in valid if r.get("backward_transfer") is not None]

print("\n" + "=" * 65)
print("STEP 772 SUMMARY - PLAIN LSH SOTA CHAIN: LS20 → CIFAR")
print("=" * 65)
if avg_accs:
    print(f"Avg accuracy: {float(np.mean(avg_accs)):.4f} (seeds: {[f'{a:.4f}' for a in avg_accs]})")
    if bwts: print(f"BWT: {float(np.mean(bwts)):.4f}")
    print(f"\nBaseline (Step 761 PlainLSH cold CIFAR): 20.04%, BWT=4.3%")
    print(f"Baseline (Step 770 674 LS20→CIFAR):     20.13%, BWT=6.5%")
    print(f"If PlainLSH chain ≈ cold: LS20 pretraining requires M elements (refinement) to transfer.")
    print(f"If PlainLSH chain > cold: simple accumulation enables cross-domain transfer.")
else:
    print("No valid results.")
print("=" * 65)
print("STEP 772 DONE")
print("=" * 65)
