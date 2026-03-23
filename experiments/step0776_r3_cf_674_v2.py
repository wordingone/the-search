"""
step0776_r3_cf_674_v2.py — R3 Counterfactual on 674 (proper, n=20, v2).

Fixes v1 metric bug: done=True in LS20 = level timeout (failure), NOT
a level completion. The correct L1 metric is info.get('level', 0) > prev_level,
i.e., the agent advances from level 0 → 1 (completing the first level).

R3 hypothesis: 674's M elements (G edge counts, live cells, ref dict) accumulate
useful navigational structure during 5K pretrain steps. If it transfers,
warm-start substrate completes more levels in the 5K test window than cold-start.

Protocol (per Leo spec, mail 2552):
  1. Pretrain: run 674 for 5K steps on LS20 seed → save state S_5K
  2. Cold: fresh 674 → run 5K steps on SAME game sequence → count L1 advances
  3. Warm: load S_5K → run 5K steps on SAME game sequence → count L1 advances
  4. R3_cf = Fisher exact test on total cold vs warm L1 completions (20 seeds)

L1 advance = info['level'] increments (e.g., 0→1, 1→2, etc.) during test phase.

20 seeds. 10K steps budget per seed. LS20 game (5K pretrain + 5K test).
"""
import sys, os, time, copy
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from substrates.step0674 import TransitionTriggered674

print("=" * 70)
print("STEP 776 v2 — R3 COUNTERFACTUAL ON 674 (PROPER, n=20)")
print("=" * 70)
print()

N_SEEDS = 20
PRETRAIN_STEPS = 5_000
TEST_STEPS = 5_000
N_ACTIONS = 4  # LS20 uses 4 actions (1,2,3,4)


def _make_ls20():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("LS20")


def run_phase(env_seed, substrate, n_steps):
    """Run substrate on LS20 for n_steps. Returns level_completions count.

    Level completion = info['level'] advances (agent solves a level).
    done=True = timeout (failure), not a completion — ignore for metric.
    """
    env = _make_ls20()
    obs = env.reset(seed=env_seed)
    current_level = 0
    level_completions = 0
    step = 0

    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed)
            substrate.on_level_transition()
            continue

        obs_arr = np.array(obs, dtype=np.float32)
        action = substrate.process(obs_arr)
        # LS20 actions are 1-4; action from substrate is 0-3; map +1
        obs, reward, done, info = env.step((action % N_ACTIONS) + 1)
        step += 1

        # Check for level advancement (actual success metric)
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            level_completions += (cl - current_level)
            current_level = cl
            substrate.on_level_transition()

        if done:
            # Timeout/failure — reset level counter, continue from level 0
            obs = env.reset(seed=env_seed)
            current_level = 0
            substrate.on_level_transition()

    return level_completions, step


print(f"Protocol: {PRETRAIN_STEPS}K pretrain → save state → cold 5K / warm 5K")
print(f"Seeds: {N_SEEDS}  |  Game: LS20  |  Metric: level completions (info['level'] advances)")
print()

results = []

for seed in range(N_SEEDS):
    t0 = time.time()
    env_pretrain_seed = seed * 1000
    env_test_seed = seed * 1000 + 500  # Different game sequence for test

    # --- Phase 1: Pretrain ---
    sub_pretrain = TransitionTriggered674(n_actions=N_ACTIONS, seed=seed)
    sub_pretrain.reset(seed)
    l1_pre, _ = run_phase(env_pretrain_seed, sub_pretrain, PRETRAIN_STEPS)
    saved_state = sub_pretrain.get_state()

    # --- Phase 2: Cold eval (fresh substrate, test sequence) ---
    sub_cold = TransitionTriggered674(n_actions=N_ACTIONS, seed=seed)
    sub_cold.reset(seed)
    l1_cold, _ = run_phase(env_test_seed, sub_cold, TEST_STEPS)

    # --- Phase 3: Warm eval (pretrained state, same test sequence) ---
    sub_warm = TransitionTriggered674(n_actions=N_ACTIONS, seed=seed)
    sub_warm.reset(seed)
    sub_warm.set_state(saved_state)
    l1_warm, _ = run_phase(env_test_seed, sub_warm, TEST_STEPS)

    elapsed = time.time() - t0
    results.append({
        "seed": seed,
        "l1_pretrain": l1_pre,
        "l1_cold": l1_cold,
        "l1_warm": l1_warm,
        "elapsed": elapsed,
    })
    print(f"  seed={seed:02d}  pretrain={l1_pre}  cold={l1_cold}  warm={l1_warm}  "
          f"delta={l1_warm-l1_cold:+d}  ({elapsed:.1f}s)")

# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 776 v2 RESULTS — R3 COUNTERFACTUAL ON 674")
print("=" * 70)
print()

cold_totals = [r["l1_cold"] for r in results]
warm_totals = [r["l1_warm"] for r in results]

total_cold = sum(cold_totals)
total_warm = sum(warm_totals)
n_warm_better = sum(1 for r in results if r["l1_warm"] > r["l1_cold"])
n_cold_better = sum(1 for r in results if r["l1_cold"] > r["l1_warm"])
n_tied = sum(1 for r in results if r["l1_cold"] == r["l1_warm"])

print(f"  Total L1 completions (cold): {total_cold}  across {N_SEEDS} seeds × {TEST_STEPS} steps")
print(f"  Total L1 completions (warm): {total_warm}  across {N_SEEDS} seeds × {TEST_STEPS} steps")
print()
print(f"  Mean cold: {np.mean(cold_totals):.3f} ± {np.std(cold_totals):.3f}")
print(f"  Mean warm: {np.mean(warm_totals):.3f} ± {np.std(warm_totals):.3f}")
print()
print(f"  warm > cold: {n_warm_better}/20 seeds")
print(f"  cold > warm: {n_cold_better}/20 seeds")
print(f"  tied:        {n_tied}/20 seeds")
print()

# Fisher exact test on totals (warm L1 completions vs cold L1 completions)
try:
    from scipy.stats import fisher_exact
    total_steps = N_SEEDS * TEST_STEPS
    warm_no = total_steps - total_warm
    cold_no = total_steps - total_cold
    if total_warm > 0 or total_cold > 0:
        odds, p = fisher_exact([[total_warm, warm_no], [total_cold, cold_no]])
        print(f"  Fisher exact: OR={odds:.3f}  p={p:.4f}")
        r3_cf_pass = (p < 0.05 and total_warm > total_cold)
        print(f"  R3_counterfactual: {'PASS' if r3_cf_pass else 'FAIL'} "
              f"(warm {'>' if total_warm > total_cold else '<='} cold, p={p:.4f})")
    else:
        print("  No L1 completions observed in either phase.")
        print("  R3_counterfactual: INCONCLUSIVE (no level completions in 5K steps)")
        r3_cf_pass = None
except ImportError:
    from math import comb
    print(f"  scipy not available — sign test only")
    p_sign = sum(comb(20, k) * 0.5**20 for k in range(n_warm_better, 21))
    print(f"  p (one-tailed sign test): {p_sign:.4f}")
    r3_cf_pass = (p_sign < 0.05 and n_warm_better > n_cold_better)
    print(f"  R3_counterfactual: {'PASS' if r3_cf_pass else 'FAIL'}")

print()
if r3_cf_pass is True:
    print("FINDING: 674's M elements (G, cells, refinements) TRANSFER.")
    print("Pretraining gives the substrate useful structure. R3 mechanism confirmed.")
elif r3_cf_pass is False:
    print("FINDING: 674's M elements do NOT transfer between LS20 episodes.")
    print("The substrate's self-modification is navigational bookkeeping, not a")
    print("generalizable internal model. The R3 mechanism is NOT confirmed.")
else:
    print("FINDING: INCONCLUSIVE — 674 does not solve LS20 levels in 5K steps.")
    print("Need longer test window or easier game to measure R3 transfer.")

print()
print("STEP 776 v2 DONE")
