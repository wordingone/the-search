"""
step0874_obs_diversity.py -- Observation diversity characterization across games.

R3 hypothesis (characterization): does LS20 have higher observation diversity than
FT09? Lower diversity → static background → trivially predictable → pred accuracy
near 100% even for cold substrate.

Metrics:
1. Hash collision rate: fraction of repeated observation hashes in 1K steps
2. Running_mean L2 norm convergence speed (at t=100,500,1000,5000)
3. Mean prediction error (untrained W) on 1K steps
4. Per-action observation change magnitude (confirms 800b's differentiability signal)

Games: LS20 and FT09. Seed=6.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

N_STEPS = 5_000
N_ACTIONS_LS20 = 4
N_ACTIONS_FT09 = 68
ENV_SEED = 6000
DIM = 256


def make_ls20():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def make_ft09():
    try:
        import arcagi3; return arcagi3.make("FT09")
    except:
        import util_arcagi3; return util_arcagi3.make("FT09")


def characterize_game(env_fn, n_actions, env_seed, n_steps):
    env = env_fn()
    obs = env.reset(seed=env_seed)
    rng = np.random.RandomState(0)

    obs_hashes = []
    running_mean = np.zeros(DIM, np.float32)
    n_obs = 0

    W = np.zeros((DIM, DIM + n_actions), np.float32)
    pred_errors = []
    change_per_action = np.zeros(n_actions, np.float32)
    count_per_action = np.zeros(n_actions, np.int32)

    prev_enc = None
    prev_action = None

    mean_norms = {}
    checkpoints = {100, 500, 1000, 5000}

    step = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        enc = _enc_frame(obs_arr)

        # Hash
        obs_hashes.append(hash(obs_arr.tobytes()))

        # Running mean
        n_obs += 1
        alpha = 1.0 / n_obs
        running_mean = (1 - alpha) * running_mean + alpha * enc
        norm_enc = enc - running_mean

        # Prediction error (untrained W)
        if prev_enc is not None:
            a_oh = np.zeros(n_actions, np.float32); a_oh[prev_action] = 1.0
            pred = W @ np.concatenate([prev_enc, a_oh])
            err = float(np.sum((pred - norm_enc) ** 2))
            norm = float(np.sum(norm_enc ** 2)) + 1e-8
            pred_errors.append(err / norm)

        # Per-action change magnitude
        if prev_enc is not None:
            change = float(np.sum((norm_enc - prev_enc) ** 2))
            count_per_action[prev_action] += 1
            change_per_action[prev_action] += change

        # Checkpoints
        if step in checkpoints or step + 1 in checkpoints:
            mean_norms[step + 1] = float(np.linalg.norm(running_mean))

        action = rng.randint(0, n_actions)
        obs_next, _, done, _ = env.step(action); step += 1
        if done:
            obs = env.reset(seed=env_seed)
        else:
            obs = obs_next
        prev_enc = norm_enc; prev_action = action

    # Hash collision rate
    unique_hashes = len(set(obs_hashes))
    collision_rate = 1.0 - unique_hashes / len(obs_hashes)

    # Per-action change (mean)
    mean_change_per_action = np.where(count_per_action > 0,
                                       change_per_action / (count_per_action + 1e-8),
                                       0.0)
    change_differentiability = float(np.std(mean_change_per_action) / (np.mean(mean_change_per_action) + 1e-8))

    return {
        "collision_rate": collision_rate,
        "unique_obs": unique_hashes,
        "total_obs": len(obs_hashes),
        "mean_pred_error": float(np.mean(pred_errors)) if pred_errors else None,
        "mean_change_per_action": mean_change_per_action,
        "change_differentiability_cv": change_differentiability,
        "mean_norms": mean_norms,
    }


print("=" * 70)
print("STEP 874 — OBSERVATION DIVERSITY (LS20 vs FT09)")
print("=" * 70)

t0 = time.time()

print("\n--- LS20 ---")
ls20_stats = characterize_game(make_ls20, N_ACTIONS_LS20, ENV_SEED, N_STEPS)
print(f"  Hash collision rate: {ls20_stats['collision_rate']:.4f}")
print(f"  Unique obs: {ls20_stats['unique_obs']} / {ls20_stats['total_obs']}")
print(f"  Mean pred error (untrained W): {ls20_stats['mean_pred_error']:.4f}" if ls20_stats['mean_pred_error'] else "  Mean pred error: N/A")
print(f"  Change CV (action differentiability): {ls20_stats['change_differentiability_cv']:.4f}")
print(f"  mean_change_per_action: {ls20_stats['mean_change_per_action']}")

print("\n--- FT09 ---")
ft09_stats = characterize_game(make_ft09, N_ACTIONS_FT09, ENV_SEED, N_STEPS)
print(f"  Hash collision rate: {ft09_stats['collision_rate']:.4f}")
print(f"  Unique obs: {ft09_stats['unique_obs']} / {ft09_stats['total_obs']}")
print(f"  Mean pred error (untrained W): {ft09_stats['mean_pred_error']:.4f}" if ft09_stats['mean_pred_error'] else "  Mean pred error: N/A")
print(f"  Change CV (action differentiability): {ft09_stats['change_differentiability_cv']:.4f}")
print(f"  mean_change_per_action (first 8): {ft09_stats['mean_change_per_action'][:8]}")

print()
print("Interpretation:")
higher_diversity = "LS20" if ls20_stats['collision_rate'] < ft09_stats['collision_rate'] else "FT09"
print(f"  Higher diversity: {higher_diversity}")
more_differentiable = "LS20" if ls20_stats['change_differentiability_cv'] > ft09_stats['change_differentiability_cv'] else "FT09"
print(f"  More action-differentiable: {more_differentiable}")

print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 874 DONE")
