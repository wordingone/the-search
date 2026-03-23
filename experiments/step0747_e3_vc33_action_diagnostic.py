"""
Step 747 (E3): Per-cell action discrimination diagnostic on VC33.

R3 hypothesis: VC33 bottleneck is action decomposition. G dict action profiles
encode zone structure. 674 on VC33, 5 min, 3 seeds.
Extract G dict, compute per-cell action entropy.
Diagnostic: do action profile clusters match zone boundaries?

VC33 has only 1 effective action (ACTION6 = index 5 in new action space).
This experiment documents whether the substrate can distinguish game zones.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674

print("=" * 65)
print("STEP 747 (E3) — VC33 ACTION DIAGNOSTIC")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 3
PER_SEED_TIME = 60   # 3 seeds x 60s = 3 min


def _make_env():
    try:
        import arcagi3
        return arcagi3.make("VC33")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("VC33")


for seed_i in range(N_SEEDS):
    seed = SEED_BASE + seed_i
    print(f"\n-- Seed {seed} --")
    try:
        env = _make_env()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        sub = TransitionTriggered674(n_actions=n_valid, seed=seed)
        sub.reset(seed)
        obs = env.reset(seed=seed)
        level = 0; steps = 0; fresh = True
        t_start = time.time()

        while (time.time() - t_start) < PER_SEED_TIME:
            if obs is None:
                obs = env.reset(seed=seed); sub.on_level_transition(); fresh = True; continue
            obs_arr = np.array(obs, dtype=np.float32)
            action = sub.process(obs_arr)
            obs, reward, done, info = env.step(action % n_valid)
            steps += 1
            if fresh:
                fresh = False; continue
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl; sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed); sub.on_level_transition(); fresh = True

        state = sub.get_state()
        print(f"  steps={steps} level={level} G={state.get('G_size',0)} "
              f"live={state.get('live_count',0)} aliased={state.get('aliased_count',0)}")

        # Analyze G dict for per-cell action entropy
        G = sub.G
        if G:
            action_entropies = []
            action_dominant = []
            for (cell, action), successors in G.items():
                total = sum(successors.values())
                if total < 3:
                    continue
                # Entropy of successors from this (cell,action)
                v = np.array(list(successors.values()), np.float64)
                p = v / v.sum()
                h = float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))
                action_entropies.append((cell, action, h, total))
                action_dominant.append(action)

            if action_entropies:
                actions_used = [a for _, a, _, _ in action_entropies]
                from collections import Counter
                action_counts = Counter(actions_used)
                print(f"  Action distribution in G (action: count):")
                for a, cnt in sorted(action_counts.items()):
                    print(f"    action {a}: {cnt} edges")
                avg_h = np.mean([h for _, _, h, _ in action_entropies])
                print(f"  Avg edge entropy: {avg_h:.3f}")

                # Check if any cell has differentiated action profiles
                cell_actions = {}
                for (cell, action), succ in G.items():
                    if cell not in cell_actions:
                        cell_actions[cell] = {}
                    cell_actions[cell][action] = sum(succ.values())

                multi_action_cells = {c: d for c, d in cell_actions.items() if len(d) > 1}
                print(f"  Cells with >1 action explored: {len(multi_action_cells)}/{len(cell_actions)}")

                # ACTION6 = index 5 in ACTION1..ACTION7 mapping
                action6_edges = sum(1 for (c, a) in G if a == 5)
                print(f"  Edges from ACTION6 (index 5): {action6_edges}")
            else:
                print("  Not enough data for analysis (G too small)")
        else:
            print("  G is empty — VC33 returning identical frames?")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print("\n" + "=" * 65)
print("E3 SUMMARY (VC33 ACTION DIAGNOSTIC)")
print("=" * 65)
print("VC33 diagnostic complete. Key question answered:")
print("Does 674 build a graph with diverse action profiles on VC33?")
print("ACTION6 (index 5) is the only effective action per probe.")
print("=" * 65)
print("STEP 747 DONE")
print("=" * 65)
