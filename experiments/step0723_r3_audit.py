"""
Step 723 (A4): ConstitutionalJudge static + dynamic R3 audit on 674.

R3 hypothesis: audit() confirms 3M, 3I, 9U per frozen_elements().
Dynamic R3 on real LS20 data: edge_count_update and aliased_set should
change (score >= 2/3). ref_hyperplanes requires REFINE_EVERY=5000 steps;
with 3K obs, may not trigger — expected score = 2/3 or 3/3 if refinement
fires early.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates import BaseSubstrate, ConstitutionalJudge
from substrates.step0674 import TransitionTriggered674

print("=" * 65)
print("STEP 723 (A4) — CONSTITUTIONAL JUDGE AUDIT ON 674")
print("=" * 65)

judge = ConstitutionalJudge()

# --- Part 1: Static audit ---
print("\n-- Part 1: Static audit (no chain results) --")
t_start = time.time()
audit = judge.audit(TransitionTriggered674, chain_results=None,
                    game_name="LS20", seed=0, n_audit_steps=500)
elapsed_audit = time.time() - t_start
print(f"  Elapsed: {elapsed_audit:.1f}s")

for check in ["R1", "R2", "R3", "R5", "R6"]:
    r = audit.get(check, {})
    print(f"  {check}: pass={r.get('pass')} — {str(r.get('detail',''))[:70]}")

summary = audit.get("summary", {})
print(f"\n  Summary: {summary.get('verdict')}")
print(f"  Score: {summary.get('score')}")
r3 = audit.get("R3", {})
print(f"  R3: M={r3.get('M_count')} I={r3.get('I_count')} U={r3.get('U_count')}")

print("\n  R3 element classification:")
for e in r3.get("elements", []):
    print(f"    [{e['class']}] {e['name']}")

# --- Part 2: Dynamic R3 on LS20 ---
print("\n-- Part 2: Dynamic R3 — real LS20 observations (3K steps) --")

try:
    try:
        import arcagi3
        env = arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        env = util_arcagi3.make("LS20")

    n_valid = len(env._action_space) if hasattr(env, '_action_space') else 4
    print(f"  LS20 action space: {n_valid} actions")
    print(f"  Game version hash: {getattr(env, '_hash', 'unknown')}")

    # Collect 3K LS20 observations
    sub_collect = TransitionTriggered674(n_actions=n_valid, seed=0)
    sub_collect.reset(0)
    ls20_obs = []
    obs = env.reset(seed=0)
    fresh = True
    t_collect = time.time()

    while len(ls20_obs) < 3000 and (time.time() - t_collect) < 90:
        if obs is None:
            obs = env.reset(seed=0); fresh = True; continue
        obs_arr = np.array(obs, dtype=np.float32)
        ls20_obs.append(obs_arr)
        action = sub_collect.process(obs_arr)
        obs, reward, done, info = env.step(action % n_valid)
        fresh = False
        if done:
            obs = env.reset(seed=0); fresh = True

    print(f"  Collected {len(ls20_obs)} frames in {time.time()-t_collect:.1f}s")

    # State at end of collection
    final_state = sub_collect.get_state()
    print(f"  Collection substrate: G_size={final_state['G_size']} "
          f"live={final_state['live_count']} aliased={final_state.get('aliased_count',0)} "
          f"ref={final_state.get('ref_count',0)}")

    # Dynamic R3 measurement with fresh substrate
    print(f"\n  Running measure_r3_dynamics (3K steps, 10 checkpoints)...")

    class _674_ls20(TransitionTriggered674):
        def __init__(self): super().__init__(n_actions=n_valid, seed=0)

    t_r3 = time.time()
    r3_dyn = judge.measure_r3_dynamics(
        _674_ls20,
        obs_sequence=ls20_obs[:3000],
        n_steps=3000,
        n_checkpoints=10
    )
    print(f"  Elapsed: {time.time()-t_r3:.1f}s")

    print(f"\n  R3 dynamic score: {r3_dyn.get('r3_dynamic_score')}")
    print(f"  Dynamics profile: {r3_dyn.get('dynamics_profile')}")
    print(f"  Declared M: {r3_dyn.get('declared_M_elements')}")
    print(f"  Verified M: {r3_dyn.get('verified_M_elements')}")
    print(f"  Detail: {r3_dyn.get('detail')}")

    change_times = r3_dyn.get("component_change_times", {})
    print(f"\n  Components first changed (of {len(change_times)} total):")
    for k, v in sorted(change_times.items(), key=lambda x: x[1])[:10]:
        print(f"    {k:<30} step {v}")

    if not change_times:
        print("    (no components changed)")

    print(f"\n  Checkpoint summary:")
    for ckpt in r3_dyn.get("checkpoints", []):
        changed = ckpt.get("changed_from_prev", [])
        print(f"    step={ckpt['step']:>5}: changed={len(changed)} keys {changed[:3]}")

except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 65)
print("STEP 723 DONE")
print("=" * 65)
