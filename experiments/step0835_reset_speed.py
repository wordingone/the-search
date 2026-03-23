"""
step0835_reset_speed.py -- Substrate reset speed: how fast does warm substrate re-learn?

R3 hypothesis: a substrate that has seen the game (warm W) should re-learn faster
after a full weight reset than a fresh cold substrate. If true → W learned something
that reduces future learning time (meta-learning signal).

Protocol:
1. Pretrain substrate on seeds 1-5 (25K total). Save W.
2. Reset substrate to s_0 (cold weights, but use same running_mean from pretrain).
3. Run on seed 6. Measure pred accuracy at checkpoints: 1K, 5K, 10K, 25K steps.
4. Compare cold (fresh) vs reset (cold-weights-warm-mean) learning curves.

Uses PredictionContrast780.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0780 import PredictionContrast780

PRETRAIN_SEEDS = [1, 2, 3, 4, 5]
PRETRAIN_STEPS = 5_000
TEST_SEED = 6
N_ACTIONS = 4
CHECKPOINTS = [1_000, 5_000, 10_000, 25_000]


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_with_checkpoints(substrate, env_seed, checkpoints):
    """Run substrate, return pred accuracy at each checkpoint."""
    env = make_game()
    obs = env.reset(seed=env_seed)
    step = 0; pred_errors = []; prev_enc = None
    results = {}
    cp_set = set(checkpoints)

    while step < max(checkpoints):
        if obs is None:
            obs = env.reset(seed=env_seed); substrate.on_level_transition(); continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr)
        obs_next, _, done, _ = env.step(action % N_ACTIONS); step += 1
        if done:
            obs_next = env.reset(seed=env_seed); substrate.on_level_transition()
        if prev_enc is not None and obs_next is not None and substrate._last_enc is not None:
            next_enc = substrate._encode_for_pred(np.asarray(obs_next, dtype=np.float32))
            pred = substrate.predict_next(prev_enc, action % N_ACTIONS)
            err = float(np.sum((pred - next_enc) ** 2))
            norm = float(np.sum(next_enc ** 2)) + 1e-8
            pred_errors.append((err, norm))
        if substrate._last_enc is not None:
            prev_enc = substrate._last_enc.copy()
        obs = obs_next
        if step in cp_set:
            if pred_errors:
                te = sum(e for e, n in pred_errors)
                tn = sum(n for e, n in pred_errors)
                results[step] = float(1.0 - te / tn) * 100.0
            else:
                results[step] = None
    return results


print("=" * 70)
print("STEP 835 — SUBSTRATE RESET SPEED")
print("=" * 70)

t0 = time.time()

# Pretrain on seeds 1-5
sub_p = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
sub_p.reset(0)
for ps in PRETRAIN_SEEDS:
    sub_p.on_level_transition()
    env = make_game(); obs = env.reset(seed=ps * 1000); s = 0
    while s < PRETRAIN_STEPS:
        if obs is None:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition(); continue
        action = sub_p.process(np.asarray(obs, dtype=np.float32))
        obs, _, done, _ = env.step(action % N_ACTIONS); s += 1
        if done:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition()
saved = sub_p.get_state()
print(f"Pretrain done ({time.time()-t0:.1f}s).")

# Condition 1: Full cold (no pretrain, no running_mean)
print("\nRunning: COLD (fresh weights + fresh running_mean)...")
sub_cold = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
sub_cold.reset(0)
cold_results = run_with_checkpoints(sub_cold, TEST_SEED * 1000, CHECKPOINTS)

# Condition 2: Reset (fresh W + warm running_mean)
print("Running: RESET (fresh W + warm running_mean)...")
sub_reset = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
sub_reset.reset(0)
state = sub_reset.get_state()
state["running_mean"] = saved["running_mean"].copy()
state["_n_obs"] = saved["_n_obs"]
# Leave W cold (fresh/zeros)
sub_reset.set_state(state)
reset_results = run_with_checkpoints(sub_reset, TEST_SEED * 1000, CHECKPOINTS)

# Condition 3: Full warm (W + running_mean transferred)
print("Running: WARM (warm W + warm running_mean)...")
sub_warm = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
sub_warm.reset(0); sub_warm.set_state(saved)
warm_results = run_with_checkpoints(sub_warm, TEST_SEED * 1000, CHECKPOINTS)

print()
print(f"{'step':>8}  {'cold':>8}  {'reset':>8}  {'warm':>8}")
for cp in CHECKPOINTS:
    c = cold_results.get(cp)
    r = reset_results.get(cp)
    w = warm_results.get(cp)
    cs = f"{c:.2f}%" if c is not None else "N/A"
    rs = f"{r:.2f}%" if r is not None else "N/A"
    ws = f"{w:.2f}%" if w is not None else "N/A"
    print(f"{cp:>8}  {cs:>8}  {rs:>8}  {ws:>8}")

print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 835 DONE")
