"""
step0776_r3_cf_674_v3.py — R3 Counterfactual on 674 (proper, n=20, v3).

Fixes v2 budget issue: 5K steps insufficient for 674 to complete LS20 level 0.
v3 uses 25K pretrain + 25K test steps. Game steps are fast (~0.1ms each);
the ~3s overhead is API initialization, not step computation. So 25K vs 5K
adds <1s per phase.

From step705: 674 achieves L1 in 17/20 seeds within 25s (≈25K-250K steps).
25K steps should give L1 for several seeds, making the comparison informative.

R3 hypothesis: 674's M elements (G, cells, ref dict) pretrained on LS20 sequence A
transfer to LS20 sequence B. Warm (pretrained on A, tested on B) > Cold (tested on B).

Protocol:
  1. Pretrain: run 674 for 25K steps on seed*1000 → save state S_25K
  2. Cold: fresh 674 → run 25K steps on seed*1000+500 → count L1 advances
  3. Warm: load S_25K → run 25K steps on seed*1000+500 → count L1 advances
  4. R3_cf = Fisher exact on total L1 completions (20 seeds)

20 seeds. LS20 game (25K pretrain + 25K test).
"""
import sys, os, time, copy
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from substrates.step0674 import TransitionTriggered674

print("=" * 70)
print("STEP 776 v3 — R3 COUNTERFACTUAL ON 674 (25K STEPS, n=20)")
print("=" * 70)
print()

N_SEEDS = 20
PRETRAIN_STEPS = 25_000
TEST_STEPS = 25_000
N_ACTIONS = 4


def _make_ls20():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("LS20")


def run_phase(env_seed, substrate, n_steps, label=""):
    """Run substrate on LS20 for n_steps. Returns level completions (info['level'] advances)."""
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
        obs, reward, done, info = env.step(action % N_ACTIONS)
        step += 1

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            level_completions += (cl - current_level)
            current_level = cl
            substrate.on_level_transition()

        if done:
            obs = env.reset(seed=env_seed)
            current_level = 0
            substrate.on_level_transition()

    return level_completions, step


print(f"Protocol: {PRETRAIN_STEPS//1000}K pretrain → save → cold {TEST_STEPS//1000}K / warm {TEST_STEPS//1000}K")
print(f"Seeds: {N_SEEDS}  |  Game: LS20  |  Metric: level advances (info['level'])")
print()

results = []

for seed in range(N_SEEDS):
    t0 = time.time()
    env_pretrain_seed = seed * 1000
    env_test_seed = seed * 1000 + 500

    sub_pretrain = TransitionTriggered674(n_actions=N_ACTIONS, seed=seed)
    sub_pretrain.reset(seed)
    l1_pre, _ = run_phase(env_pretrain_seed, sub_pretrain, PRETRAIN_STEPS, "pretrain")
    saved_state = sub_pretrain.get_state()

    sub_cold = TransitionTriggered674(n_actions=N_ACTIONS, seed=seed)
    sub_cold.reset(seed)
    l1_cold, _ = run_phase(env_test_seed, sub_cold, TEST_STEPS, "cold")

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
    print(f"  seed={seed:02d}  pretrain={l1_pre}  cold={l1_cold}  warm={l1_warm}  "
          f"delta={l1_warm-l1_cold:+d}  ({elapsed:.1f}s)")

print()
print("=" * 70)
print("STEP 776 v3 RESULTS — R3 COUNTERFACTUAL ON 674")
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

try:
    from scipy.stats import fisher_exact
    total_steps = N_SEEDS * TEST_STEPS
    if total_warm > 0 or total_cold > 0:
        warm_no = total_steps - total_warm
        cold_no = total_steps - total_cold
        odds, p = fisher_exact([[total_warm, warm_no], [total_cold, cold_no]])
        print(f"  Fisher exact: OR={odds:.3f}  p={p:.4f}")
        r3_cf_pass = (p < 0.05 and total_warm > total_cold)
        print(f"  R3_counterfactual: {'PASS' if r3_cf_pass else 'FAIL'} "
              f"(warm {'>' if total_warm > total_cold else '<='} cold, p={p:.4f})")
    else:
        print("  INCONCLUSIVE: no level completions in 25K steps per phase.")
        print("  674 does not complete LS20 level 0 in this budget.")
        r3_cf_pass = None
except ImportError:
    from math import comb
    p_sign = sum(comb(20, k) * 0.5**20 for k in range(n_warm_better, 21))
    r3_cf_pass = (p_sign < 0.05 and n_warm_better > n_cold_better)
    print(f"  p (sign test): {p_sign:.4f}  R3_cf: {'PASS' if r3_cf_pass else 'FAIL'}")

print()
if r3_cf_pass is True:
    print("FINDING: 674's M elements TRANSFER across LS20 episodes. R3 confirmed.")
elif r3_cf_pass is False:
    print("FINDING: 674's M elements do NOT transfer. R3 mechanism NOT confirmed.")
else:
    print("FINDING: INCONCLUSIVE — 674 cannot solve LS20 in 25K steps.")
    print("The level timeout is shorter than L1 solve time for 674.")
    print("R3 counterfactual cannot be measured with the episode-completion metric.")

print()
print("STEP 776 v3 DONE")
