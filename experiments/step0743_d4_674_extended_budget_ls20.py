"""
Step 743 (D4): 674+running-mean at extended budget (LS20 L2 exploration).

R3 hypothesis: L2 may be budget-limited. 674 gives faster L1. With extended budget,
L2 might be reachable. (300s requires Jun approval; using 60s as intermediate.)

674 baseline, 5 seeds, 60s each.
Measure: L1 time, L2 status, growth curve (aliased cells over time).
Success: any L2 progress. Kill: L2=0/5, no new states post-L1.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674

print("=" * 65)
print("STEP 743 (D4) — 674 EXTENDED BUDGET LS20 L2")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 5
PER_SEED_TIME = 60


def _make_env():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("LS20")


print(f"\n-- LS20 {N_SEEDS} seeds x {PER_SEED_TIME}s (674 baseline) --")
results = []

for seed_i in range(N_SEEDS):
    seed = SEED_BASE + seed_i * 100
    try:
        env = _make_env()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        sub = TransitionTriggered674(n_actions=n_valid, seed=seed)
        sub.reset(seed)
        obs = env.reset(seed=seed)
        level = 0
        l1_step = l2_step = None
        steps = 0
        fresh = True
        t_start = time.time()
        aliased_log = []

        while (time.time() - t_start) < PER_SEED_TIME:
            if obs is None:
                obs = env.reset(seed=seed)
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
                obs = env.reset(seed=seed)
                sub.on_level_transition()
                fresh = True
            if steps % 20000 == 0:
                state = sub.get_state()
                aliased_log.append((steps, state.get("aliased_count", 0)))

        state = sub.get_state()
        results.append({"seed": seed, "l1": l1_step, "l2": l2_step, "steps": steps,
                        "G": state.get("G_size", 0), "aliased": state.get("aliased_count", 0)})
        status = f"L{level}" if level > 0 else "  "
        print(f"  seed={seed:>4} {status} steps={steps:>6} G={state.get('G_size',0):>5} "
              f"aliased={state.get('aliased_count',0):>3} l1={l1_step} l2={l2_step}")
        if aliased_log:
            print(f"    aliased growth: {aliased_log}")
    except Exception as e:
        print(f"  seed={seed:>4} ERROR: {e}")
        results.append({"seed": seed, "l1": None, "l2": None, "steps": 0, "error": str(e)})

l1_count = sum(1 for r in results if r.get("l1"))
l2_count = sum(1 for r in results if r.get("l2"))

print("\n" + "=" * 65)
print("D4 SUMMARY (674 EXTENDED BUDGET)")
print("=" * 65)
print(f"LS20 L1: {l1_count}/{N_SEEDS}")
print(f"LS20 L2: {l2_count}/{N_SEEDS}")
print(f"Compare D1 (retention): L1=4/5, L2=0/5 at 60s")
print("If L2=0/5: L2 not just budget-limited — requires different mechanism")
print("=" * 65)
print("STEP 743 DONE")
print("=" * 65)
