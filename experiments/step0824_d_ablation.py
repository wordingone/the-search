"""
step0824_d_ablation.py -- D component ablation: W only vs running_mean only transfer.

R3_cf PASS confirmed (5/7 substrates, pred accuracy). Which component carries the
transfer? This test isolates W (dynamics model) from running_mean (obs distribution).

Protocol: pretrain full substrate on seeds 1-5. Then test:
1. Full D transfer: warm W + warm running_mean (standard R3_cf PASS)
2. W only: warm W + cold running_mean (reset running_mean)
3. mean only: cold W + warm running_mean (reset W)
4. Cold: fresh W + fresh running_mean (standard cold)

Uses step780v5 (PredictionContrast + delta rule W) — cleanest D(s) substrate.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0780 import PredictionContrast780

# Reuse r3cf_runner infrastructure
PRETRAIN_SEEDS = [1, 2, 3, 4, 5]
TEST_SEEDS = [6, 7, 8, 9, 10]
PRETRAIN_STEPS = 5_000
TEST_STEPS = 25_000
DIM = 256


def make_game():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except:
        import util_arcagi3
        return util_arcagi3.make("LS20")


def run_test_seed(substrate, env_seed, n_steps, collect_pred=True):
    """Run substrate for n_steps, return (completions, mean_pred_acc)."""
    env = make_game()
    obs = env.reset(seed=env_seed)
    completions = 0; current_level = 0; step = 0
    pred_errors = []
    prev_enc = None

    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); substrate.on_level_transition(); continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr)
        obs_next, _, done, info = env.step(action % 4)
        step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs_next = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
        if collect_pred and hasattr(substrate, 'predict_next') and prev_enc is not None:
            if obs_next is not None and substrate._last_enc is not None:
                next_arr = np.asarray(obs_next, dtype=np.float32)
                next_enc = substrate._encode_for_pred(next_arr) if hasattr(substrate, '_encode_for_pred') else substrate._encode(next_arr)
                pred = substrate.predict_next(prev_enc, action % 4)
                if pred is not None:
                    err = float(np.sum((pred - next_enc) ** 2))
                    norm = float(np.sum(next_enc ** 2)) + 1e-8
                    pred_errors.append((err, norm))
        if hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
            prev_enc = substrate._last_enc.copy()
        obs = obs_next

    pred_acc = None
    if pred_errors:
        te = sum(e for e, n in pred_errors)
        tn = sum(n for e, n in pred_errors)
        pred_acc = float(1.0 - te / tn) * 100.0
    return completions, pred_acc


print("=" * 70)
print("STEP 824 — D COMPONENT ABLATION")
print("=" * 70)
print("Testing which D(s) component carries R3_cf transfer.")
print()

t0 = time.time()

# Pretrain
sub_pretrain = PredictionContrast780(n_actions=4, seed=0)
sub_pretrain.reset(0)
pre_c = 0
for ps in PRETRAIN_SEEDS:
    sub_pretrain.on_level_transition()
    env = make_game()
    obs = env.reset(seed=ps * 1000)
    s = 0
    while s < PRETRAIN_STEPS:
        if obs is None:
            obs = env.reset(seed=ps * 1000); sub_pretrain.on_level_transition(); continue
        action = sub_pretrain.process(np.asarray(obs, dtype=np.float32))
        obs, _, done, info = env.step(action % 4)
        s += 1
        if done:
            obs = env.reset(seed=ps * 1000); sub_pretrain.on_level_transition()
saved = sub_pretrain.get_state()
print(f"Pretrain done ({time.time()-t0:.1f}s). W norm: {float(np.linalg.norm(saved['W'])):.3f}")

# Test 4 conditions
conditions = {
    "cold": {"use_W": False, "use_mean": False},
    "W_only": {"use_W": True, "use_mean": False},
    "mean_only": {"use_W": False, "use_mean": True},
    "full_D": {"use_W": True, "use_mean": True},
}

print()
for cond_name, flags in conditions.items():
    accs = []; comps = []
    for ts in TEST_SEEDS:
        sub = PredictionContrast780(n_actions=4, seed=0)
        sub.reset(0)
        # Apply selected components
        state = sub.get_state()
        if flags["use_W"]:
            state["W"] = saved["W"].copy()
        if flags["use_mean"]:
            state["running_mean"] = saved["running_mean"].copy()
            state["_n_obs"] = saved["_n_obs"]
        sub.set_state(state)
        c, pa = run_test_seed(sub, ts * 1000, TEST_STEPS)
        comps.append(c); accs.append(pa)
    mean_c = np.mean(comps)
    valid_acc = [a for a in accs if a is not None]
    mean_acc = np.mean(valid_acc) if valid_acc else None
    acc_str = f"{mean_acc:.2f}%" if mean_acc is not None else "N/A"
    print(f"  {cond_name:12s}: L1={mean_c:.0f}/seed  pred_acc={acc_str}")

print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 824 DONE")
