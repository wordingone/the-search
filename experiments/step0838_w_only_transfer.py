"""
step0838_w_only_transfer.py -- W-only transfer protocol: corrected R3_cf.

Step 824 finding: W_only (warm W + cold mean) > full_D (warm W + warm mean).
This is because frozen running_mean mismatches test distribution.

R3 hypothesis: does W-only transfer consistently PASS across multiple substrates?
Tests PredictionContrast780 with W-only protocol on LS20.
Compares: W-only warm vs full_D warm vs cold.

If W-only > full_D consistently → the standard R3_cf protocol underestimates transfer.
If W-only ≈ full_D → step824 finding was noise.

Protocol: same as R3_cf but warm condition = W only (fresh running_mean).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0780 import PredictionContrast780

PRETRAIN_SEEDS = [1, 2, 3, 4, 5]
TEST_SEEDS = [6, 7, 8, 9, 10]
PRETRAIN_STEPS = 5_000
TEST_STEPS = 25_000
N_ACTIONS = 4


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_phase(substrate, env_seed, n_steps):
    env = make_game()
    obs = env.reset(seed=env_seed)
    step = 0; pred_errors = []; prev_enc = None
    while step < n_steps:
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
    if not pred_errors: return None
    return float(1.0 - sum(e for e,n in pred_errors) / sum(n for e,n in pred_errors)) * 100.0


print("=" * 70)
print("STEP 838 — W-ONLY TRANSFER PROTOCOL (corrected R3_cf)")
print("=" * 70)
print("Conditions: cold | W_only (warm W + cold mean) | full_D (warm W + warm mean)")

t0 = time.time()

# Pretrain
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
print(f"Pretrain done ({time.time()-t0:.1f}s). W norm={float(np.linalg.norm(saved['W'])):.3f}")

cold_accs = []; w_only_accs = []; full_d_accs = []

for ts in TEST_SEEDS:
    # Cold
    sub_c = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
    sub_c.reset(0)
    acc_c = run_phase(sub_c, ts * 1000, TEST_STEPS)
    cold_accs.append(acc_c)

    # W-only: warm W, cold running_mean (fresh n_obs=0)
    sub_w = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
    sub_w.reset(0)
    state = sub_w.get_state()
    state["W"] = saved["W"].copy()
    # Keep running_mean fresh (n_obs=0, mean=0)
    sub_w.set_state(state)
    acc_w = run_phase(sub_w, ts * 1000, TEST_STEPS)
    w_only_accs.append(acc_w)

    # Full D: warm W + warm mean
    sub_d = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
    sub_d.reset(0); sub_d.set_state(saved)
    acc_d = run_phase(sub_d, ts * 1000, TEST_STEPS)
    full_d_accs.append(acc_d)

def mean_safe(vals):
    v = [x for x in vals if x is not None]
    return np.mean(v) if v else None

mc = mean_safe(cold_accs)
mw = mean_safe(w_only_accs)
md = mean_safe(full_d_accs)

print()
print(f"  cold:    {mc:.2f}%" if mc else "  cold: N/A")
print(f"  W_only:  {mw:.2f}%  {'PASS' if (mw and mc and mw > mc) else 'FAIL'}  ({mw-mc:+.2f}%)" if (mw and mc) else "  W_only: N/A")
print(f"  full_D:  {md:.2f}%  {'PASS' if (md and mc and md > mc) else 'FAIL'}  ({md-mc:+.2f}%)" if (md and mc) else "  full_D: N/A")

if mw and md:
    print(f"\n  W_only vs full_D: {mw-md:+.2f}%  {'W_only wins' if mw > md else 'full_D wins'}")

print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 838 DONE")
