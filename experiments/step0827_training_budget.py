"""
step0827_training_budget.py -- Training budget sweep: N=1K,5K,10K,25K,50K pretrain steps.

R3 hypothesis: does more pretraining improve D(s) prediction accuracy transfer?
If transfer improves monotonically with N → W learns transferable dynamics.
If transfer saturates early → low-budget regime is sufficient.

Uses PredictionContrast780 (delta rule W). Pretrain on seeds 1-5 × N steps each.
Test cold vs warm pred_acc on seeds 6-10 (fixed 25K test steps).
Metric: R3_cf pred accuracy at each N.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0780 import PredictionContrast780

PRETRAIN_SEEDS = [1, 2, 3, 4, 5]
TEST_SEEDS = [6, 7, 8, 9, 10]
TEST_STEPS = 25_000
N_ACTIONS = 4
BUDGETS = [1_000, 5_000, 10_000, 25_000, 50_000]


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
print("STEP 827 — TRAINING BUDGET SWEEP (N=1K,5K,10K,25K,50K)")
print("=" * 70)

t0 = time.time()

for N in BUDGETS:
    # Pretrain on seeds 1-5 × N steps each
    sub_p = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
    sub_p.reset(0)
    for ps in PRETRAIN_SEEDS:
        sub_p.on_level_transition()
        env = make_game(); obs = env.reset(seed=ps * 1000); s = 0
        while s < N:
            if obs is None:
                obs = env.reset(seed=ps * 1000); sub_p.on_level_transition(); continue
            action = sub_p.process(np.asarray(obs, dtype=np.float32))
            obs, _, done, _ = env.step(action % N_ACTIONS); s += 1
            if done:
                obs = env.reset(seed=ps * 1000); sub_p.on_level_transition()
    saved = sub_p.get_state()

    cold_accs = []; warm_accs = []
    for ts in TEST_SEEDS:
        sub_c = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
        sub_c.reset(0)
        acc_c = run_phase(sub_c, ts * 1000, TEST_STEPS)
        cold_accs.append(acc_c)

        sub_w = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
        sub_w.reset(0); sub_w.set_state(saved)
        acc_w = run_phase(sub_w, ts * 1000, TEST_STEPS)
        warm_accs.append(acc_w)

    vc = [a for a in cold_accs if a is not None]
    vw = [a for a in warm_accs if a is not None]
    mc = np.mean(vc) if vc else None
    mw = np.mean(vw) if vw else None
    r3_pass = mc is not None and mw is not None and mw > mc
    diff_str = f"+{mw-mc:.2f}%" if (mc is not None and mw is not None) else "N/A"
    if mc is not None and mw is not None:
        print(f"  N={N:6d}: cold={mc:.2f}%  warm={mw:.2f}%  {'PASS' if r3_pass else 'FAIL'}  ({diff_str})")
    else:
        print(f"  N={N:6d}: N/A")

print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 827 DONE")
