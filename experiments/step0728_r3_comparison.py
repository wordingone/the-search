"""
Step 728 (C4): R3 dynamics comparison — 674 vs PlainLSH on LS20.

R3 hypothesis: 674 R3 dynamic score > PlainLSH R3 dynamic score because
674 has ℓ_π refinement (ref_hyperplanes M element) that PlainLSH lacks.

PlainLSH only has edge_count_update (M). 674 has edge_count_update +
aliased_set + ref_hyperplanes (all M). Expected:
  PlainLSH: R3_score = 1/1 (edge_count only)
  674: R3_score = 2/3 or 3/3 (more M elements verified)

Same seed, same LS20 observations.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674
from substrates.plain_lsh import PlainLSH
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 728 (C4) — R3 COMPARISON: 674 vs PLAIN LSH ON LS20")
print("=" * 65)

SEED = 0
N_STEPS = 5_000
judge = ConstitutionalJudge()

# --- Static R3 audit both substrates ---
print("\n-- Static R3 audit --")
for cls, name in [(TransitionTriggered674, "674"), (PlainLSH, "PlainLSH")]:
    sub = cls()
    elements = sub.frozen_elements()
    m = [e for e in elements if e["class"] == "M"]
    i = [e for e in elements if e["class"] == "I"]
    u = [e for e in elements if e["class"] == "U"]
    print(f"  {name}: M={len(m)} I={len(i)} U={len(u)}")
    print(f"    M elements: {[e['name'] for e in m]}")
    print(f"    U elements: {[e['name'] for e in u]}")

# --- Collect shared LS20 observations ---
print(f"\n-- Collecting {N_STEPS} LS20 observations (shared) --")
try:
    try:
        import arcagi3
        env = arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        env = util_arcagi3.make("LS20")

    n_valid = len(env._action_space) if hasattr(env, '_action_space') else 4
    print(f"  Action space: {n_valid}")

    # Collect with neutral policy (674, seed=0)
    sub_col = TransitionTriggered674(n_actions=n_valid, seed=SEED)
    sub_col.reset(SEED)
    ls20_obs = []
    obs = env.reset(seed=SEED)
    fresh = True
    t_start = time.time()
    l1_674 = l2_674 = None
    level = 0
    steps_col = 0

    while len(ls20_obs) < N_STEPS and (time.time() - t_start) < 150:
        if obs is None:
            obs = env.reset(seed=SEED)
            sub_col.on_level_transition()
            fresh = True
            continue
        obs_arr = np.array(obs, dtype=np.float32)
        ls20_obs.append(obs_arr)
        action = sub_col.process(obs_arr)
        obs, reward, done, info = env.step(action % n_valid)
        steps_col += 1
        if fresh:
            fresh = False
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1_674 is None:
                l1_674 = steps_col
            if cl == 2 and l2_674 is None:
                l2_674 = steps_col
            level = cl
            sub_col.on_level_transition()
        if done:
            obs = env.reset(seed=SEED)
            sub_col.on_level_transition()
            fresh = True

    print(f"  Collected {len(ls20_obs)} frames in {time.time()-t_start:.1f}s")
    print(f"  674 collection: l1={l1_674} l2={l2_674}")

    # --- Dynamic R3 for 674 ---
    print(f"\n-- Dynamic R3: 674 ({N_STEPS} obs, 10 ckpts) --")
    class _674_ls20(TransitionTriggered674):
        def __init__(self): super().__init__(n_actions=n_valid, seed=0)

    t = time.time()
    r3_674 = judge.measure_r3_dynamics(
        _674_ls20, obs_sequence=ls20_obs,
        n_steps=len(ls20_obs), n_checkpoints=10
    )
    print(f"  Elapsed: {time.time()-t:.1f}s")
    print(f"  R3 score: {r3_674.get('r3_dynamic_score')}")
    print(f"  Profile: {r3_674.get('dynamics_profile')}")
    print(f"  Declared M: {r3_674.get('declared_M_elements')}")
    print(f"  Verified M: {r3_674.get('verified_M_elements')}")
    change_times_674 = r3_674.get("component_change_times", {})
    for k, v in sorted(change_times_674.items(), key=lambda x: x[1])[:5]:
        print(f"    {k}: first changed at step {v}")

    # --- Dynamic R3 for PlainLSH ---
    print(f"\n-- Dynamic R3: PlainLSH ({N_STEPS} obs, 10 ckpts) --")
    class _plain_ls20(PlainLSH):
        def __init__(self): super().__init__(n_actions=n_valid, seed=0)

    t = time.time()
    r3_plain = judge.measure_r3_dynamics(
        _plain_ls20, obs_sequence=ls20_obs,
        n_steps=len(ls20_obs), n_checkpoints=10
    )
    print(f"  Elapsed: {time.time()-t:.1f}s")
    print(f"  R3 score: {r3_plain.get('r3_dynamic_score')}")
    print(f"  Profile: {r3_plain.get('dynamics_profile')}")
    print(f"  Declared M: {r3_plain.get('declared_M_elements')}")
    print(f"  Verified M: {r3_plain.get('verified_M_elements')}")
    change_times_plain = r3_plain.get("component_change_times", {})
    for k, v in sorted(change_times_plain.items(), key=lambda x: x[1])[:5]:
        print(f"    {k}: first changed at step {v}")

    # --- Also run PlainLSH game ---
    print(f"\n-- PlainLSH game run (same obs, measure l1) --")
    sub_plain = PlainLSH(n_actions=n_valid, seed=SEED)
    sub_plain.reset(SEED)
    obs = env.reset(seed=SEED)
    fresh = True
    l1_plain = l2_plain = None
    level_plain = 0
    steps_plain = 0
    t_start = time.time()

    while steps_plain < N_STEPS and (time.time() - t_start) < 150:
        if obs is None:
            obs = env.reset(seed=SEED)
            sub_plain.on_level_transition()
            fresh = True
            continue
        obs_arr = np.array(obs, dtype=np.float32)
        action = sub_plain.process(obs_arr)
        obs, reward, done, info = env.step(action % n_valid)
        steps_plain += 1
        if fresh:
            fresh = False
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level_plain:
            if cl == 1 and l1_plain is None:
                l1_plain = steps_plain
            if cl == 2 and l2_plain is None:
                l2_plain = steps_plain
            level_plain = cl
            sub_plain.on_level_transition()
        if done:
            obs = env.reset(seed=SEED)
            sub_plain.on_level_transition()
            fresh = True

    print(f"  PlainLSH: l1={l1_plain} l2={l2_plain} steps={steps_plain}")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  674:      R3_score={r3_674.get('r3_dynamic_score')} "
          f"profile={r3_674.get('dynamics_profile')} "
          f"verified_M={r3_674.get('verified_M_elements')} "
          f"l1={l1_674}")
    print(f"  PlainLSH: R3_score={r3_plain.get('r3_dynamic_score')} "
          f"profile={r3_plain.get('dynamics_profile')} "
          f"verified_M={r3_plain.get('verified_M_elements')} "
          f"l1={l1_plain}")

    delta = (r3_674.get("r3_dynamic_score") or 0) - (r3_plain.get("r3_dynamic_score") or 0)
    print(f"\n  R3 delta (674 - PlainLSH): {delta:.3f}")
    if delta > 0:
        print(f"  SUPPORT: 674 R3 > PlainLSH R3 — ℓ_π adds M dynamics")
    elif delta == 0:
        print(f"  NEUTRAL: same R3 score — ℓ_π not measured by dynamic R3")
    else:
        print(f"  REJECT: PlainLSH R3 > 674 R3 — unexpected")

except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 65)
print("STEP 728 DONE")
print("=" * 65)
