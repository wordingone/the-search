"""
step0829_within_game_transfer.py -- Weaker R3_cf: within-game seed-to-seed transfer.

R3 hypothesis: does D(s) transfer hold for same-game adjacent seeds?
Tests 10 seed pairs: train on seed s, test on seed s+1.
If transfer holds at this fine-grained level → W encodes game mechanics, not
seed-specific patterns.

Uses PredictionContrast780. Single-seed pretrain (5K steps), single-seed test (25K steps).
Metric: pred accuracy warm vs cold for each pair.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0780 import PredictionContrast780

PRETRAIN_STEPS = 5_000
TEST_STEPS = 25_000
N_ACTIONS = 4
SEED_PAIRS = [(s, s + 1) for s in range(1, 11)]  # (1,2), (2,3), ..., (10,11)


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
    return float(1.0 - sum(e for e, n in pred_errors) / sum(n for e, n in pred_errors)) * 100.0


print("=" * 70)
print("STEP 829 — WITHIN-GAME SEED-TO-SEED TRANSFER")
print("=" * 70)
print(f"10 seed pairs: (train_seed, test_seed). 5K pretrain, 25K test.")

t0 = time.time()
passes = 0; totals = 0

for (train_s, test_s) in SEED_PAIRS:
    # Pretrain on train_s
    sub_p = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
    sub_p.reset(0)
    sub_p.on_level_transition()
    env = make_game(); obs = env.reset(seed=train_s * 1000); s = 0
    while s < PRETRAIN_STEPS:
        if obs is None:
            obs = env.reset(seed=train_s * 1000); sub_p.on_level_transition(); continue
        action = sub_p.process(np.asarray(obs, dtype=np.float32))
        obs, _, done, _ = env.step(action % N_ACTIONS); s += 1
        if done:
            obs = env.reset(seed=train_s * 1000); sub_p.on_level_transition()
    saved = sub_p.get_state()

    # Cold test on test_s
    sub_c = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
    sub_c.reset(0)
    acc_c = run_phase(sub_c, test_s * 1000, TEST_STEPS)

    # Warm test on test_s
    sub_w = PredictionContrast780(n_actions=N_ACTIONS, seed=0)
    sub_w.reset(0); sub_w.set_state(saved)
    acc_w = run_phase(sub_w, test_s * 1000, TEST_STEPS)

    if acc_c is not None and acc_w is not None:
        r3_pass = acc_w > acc_c
        passes += r3_pass; totals += 1
        diff = acc_w - acc_c
        print(f"  ({train_s:2d}→{test_s:2d}): cold={acc_c:.2f}%  warm={acc_w:.2f}%  {'PASS' if r3_pass else 'FAIL'}  ({diff:+.2f}%)")
    else:
        print(f"  ({train_s:2d}→{test_s:2d}): N/A")

print()
print(f"R3_cf within-game: {passes}/{totals} PASS")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 829 DONE")
