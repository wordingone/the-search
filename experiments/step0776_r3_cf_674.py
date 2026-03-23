"""
step0776_r3_cf_674.py — R3 Counterfactual on 674 (proper, n=20).

R3 hypothesis: 674's M elements (G edge counts, live cells, ref dict) accumulate
useful navigational structure during 5K steps of LS20. If this structure
transfers, the warm-start substrate (pretrained 5K) should complete more levels
in the 5K test window than a cold-start substrate on the same game sequence.

Protocol (per Leo spec, mail 2552):
  1. Pretrain: run 674 for 5K steps on LS20 seed → save state S_5K
  2. Cold: fresh 674 → run 5K steps on SAME game sequence → count L1 completions
  3. Warm: load S_5K → run 5K steps on SAME game sequence → count L1 completions
  4. R3_cf = Fisher exact test on total cold vs warm L1 completions (20 seeds)

If P_warm > P_cold: substrate's self-modification IS the R3 mechanism.
If P_warm ≈ P_cold: M elements are navigational noise, not transferable structure.

20 seeds. 10K steps budget per seed. LS20 game (5K pretrain + 5K test each phase).
"""
import sys, os, time, copy
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from substrates.step0674 import TransitionTriggered674

print("=" * 70)
print("STEP 776 — R3 COUNTERFACTUAL ON 674 (PROPER, n=20)")
print("=" * 70)
print()

N_SEEDS = 20
PRETRAIN_STEPS = 5_000
TEST_STEPS = 5_000
N_ACTIONS = 7


def _make_ls20():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("LS20")


def run_phase(env_seed, substrate, n_steps, phase_name):
    """Run substrate on LS20 for n_steps. Returns (l1_count, obs_count)."""
    env = _make_ls20()
    obs = env.reset(seed=env_seed)
    l1_count = 0
    step = 0
    fresh_first = True  # ignore first step's cl signal (LS20 false-positive)

    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed + step)
            substrate.on_level_transition()
            fresh_first = True
            continue

        obs_arr = np.array(obs, dtype=np.float32)
        action = substrate.process(obs_arr)
        obs, reward, done, info = env.step(action % N_ACTIONS)
        step += 1

        if done:
            if not fresh_first:
                l1_count += 1
            fresh_first = False
            obs = env.reset(seed=env_seed + step)
            substrate.on_level_transition()
            fresh_first = True
        else:
            fresh_first = False

    return l1_count, step


print(f"Protocol: {PRETRAIN_STEPS}K pretrain → save state → cold 5K / warm 5K")
print(f"Seeds: {N_SEEDS}  |  Game: LS20")
print()

results = []

for seed in range(N_SEEDS):
    t0 = time.time()
    env_pretrain_seed = seed * 1000
    env_test_seed = seed * 1000 + 500  # Different sequence for test

    # --- Phase 1: Pretrain ---
    sub_pretrain = TransitionTriggered674(n_actions=N_ACTIONS, seed=seed)
    sub_pretrain.reset(seed)
    l1_pre, _ = run_phase(env_pretrain_seed, sub_pretrain, PRETRAIN_STEPS, "pretrain")
    saved_state = sub_pretrain.get_state()

    # --- Phase 2: Cold eval (fresh substrate, test sequence) ---
    sub_cold = TransitionTriggered674(n_actions=N_ACTIONS, seed=seed)
    sub_cold.reset(seed)
    l1_cold, _ = run_phase(env_test_seed, sub_cold, TEST_STEPS, "cold")

    # --- Phase 3: Warm eval (pretrained state, same test sequence) ---
    sub_warm = TransitionTriggered674(n_actions=N_ACTIONS, seed=seed)
    sub_warm.reset(seed)
    sub_warm.set_state(saved_state)
    l1_warm, _ = run_phase(env_test_seed, sub_warm, TEST_STEPS, "warm")

    elapsed = time.time() - t0
    results.append({
        "seed": seed,
        "l1_pretrain": l1_pre,
        "l1_cold": l1_cold,
        "l1_warm": l1_warm,
        "elapsed": elapsed,
    })
    print(f"  seed={seed:02d}  pretrain_l1={l1_pre}  cold={l1_cold}  warm={l1_warm}  "
          f"delta={l1_warm-l1_cold:+d}  ({elapsed:.1f}s)")

# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 776 RESULTS — R3 COUNTERFACTUAL ON 674")
print("=" * 70)
print()

cold_totals = [r["l1_cold"] for r in results]
warm_totals = [r["l1_warm"] for r in results]

total_cold = sum(cold_totals)
total_warm = sum(warm_totals)
n_warm_better = sum(1 for r in results if r["l1_warm"] > r["l1_cold"])
n_cold_better = sum(1 for r in results if r["l1_cold"] > r["l1_warm"])
n_tied = sum(1 for r in results if r["l1_cold"] == r["l1_warm"])

print(f"  Total L1 (cold): {total_cold}  across {N_SEEDS} seeds × {TEST_STEPS} steps")
print(f"  Total L1 (warm): {total_warm}  across {N_SEEDS} seeds × {TEST_STEPS} steps")
print()
print(f"  Mean cold: {np.mean(cold_totals):.3f} ± {np.std(cold_totals):.3f}")
print(f"  Mean warm: {np.mean(warm_totals):.3f} ± {np.std(warm_totals):.3f}")
print()
print(f"  warm > cold: {n_warm_better}/20 seeds")
print(f"  cold > warm: {n_cold_better}/20 seeds")
print(f"  tied:        {n_tied}/20 seeds")
print()

# Fisher exact test on totals
try:
    from scipy.stats import fisher_exact
    # 2x2: [[warm_l1, warm_no_l1], [cold_l1, cold_no_l1]]
    warm_no = N_SEEDS * TEST_STEPS - total_warm
    cold_no = N_SEEDS * TEST_STEPS - total_cold
    odds, p = fisher_exact([[total_warm, warm_no], [total_cold, cold_no]])
    print(f"  Fisher exact: OR={odds:.3f}  p={p:.4f}")
    r3_cf_pass = (p < 0.05 and total_warm > total_cold)
    print(f"  R3_counterfactual: {'PASS' if r3_cf_pass else 'FAIL'} "
          f"(warm {'>' if total_warm > total_cold else '<='} cold, p={p:.4f})")
except ImportError:
    # Manual sign test fallback
    print(f"  scipy not available — sign test only")
    print(f"  Sign test: warm better in {n_warm_better}/20 seeds")
    from math import comb
    p_sign = sum(comb(20, k) * 0.5**20 for k in range(n_warm_better, 21))
    print(f"  p (one-tailed sign test): {p_sign:.4f}")
    r3_cf_pass = (p_sign < 0.05 and n_warm_better > n_cold_better)
    print(f"  R3_counterfactual: {'PASS' if r3_cf_pass else 'FAIL'}")

print()
if r3_cf_pass:
    print("FINDING: 674's M elements (G, cells, refinements) TRANSFER.")
    print("Pretraining gives the substrate useful structure. R3 mechanism confirmed.")
else:
    print("FINDING: 674's M elements do NOT transfer between LS20 episodes.")
    print("The substrate's self-modification is navigational bookkeeping, not a")
    print("generalizable internal model. The R3 mechanism is NOT confirmed.")

print()
print("STEP 776 DONE")
