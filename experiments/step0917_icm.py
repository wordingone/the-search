"""
step0917_icm.py -- ICM (Pathak 2017) simplified for our framework.

Intrinsic Curiosity Module: prediction error in feature space as intrinsic reward.
Closest published method to our alpha mechanism.

Simplification (no backprop through policy, no inverse model training):
- Feature space: our avgpool16+centered encoding (256D) — skip learned features
- Forward model: linear W with delta rule (same as ours)
- Intrinsic reward = forward prediction error magnitude
- Action = softmax over cumulative intrinsic reward per action (EMA)

This isolates the action-selection policy from our alpha mechanism:
  895h: selects action with highest CHANGE in encoding (novelty of transitions)
  ICM:  selects action with highest PREDICTION ERROR (surprise at outcomes)

Run: LS20 + FT09, 25K, 10 seeds, cold, substrate_seed=seed.
Baseline comparison: 895h cold LS20=268.0/seed, FT09=0/10.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
ETA_W = 0.01
ALPHA_EMA = 0.05   # slow accumulation of intrinsic reward
INIT_REWARD = 1.0
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(rewards, temp, rng):
    x = rewards / temp
    x = x - np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(rewards), p=probs))


class ICM917:
    """ICM-simplified: forward model + intrinsic reward accumulation."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)

        # Forward model: predict enc_t from (enc_{t-1}, action)
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)

        # Cumulative intrinsic reward per action (EMA of prediction errors)
        self.reward_per_action = np.full(n_actions, INIT_REWARD, dtype=np.float32)

        # Running mean for centering
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        self._prev_enc = None
        self._prev_action = None

        # Diagnostics
        self._pred_errors = deque(maxlen=200)

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def process(self, obs):
        enc = self._encode(obs)

        if self._prev_enc is not None and self._prev_action is not None:
            # Forward prediction error
            inp = np.concatenate([self._prev_enc, one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp
            error = enc - pred

            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)

            # Intrinsic reward = prediction error magnitude
            intrinsic_r = float(np.linalg.norm(enc - (self.W @ inp)))
            self._pred_errors.append(intrinsic_r)

            a = self._prev_action
            self.reward_per_action[a] = (
                (1 - ALPHA_EMA) * self.reward_per_action[a] + ALPHA_EMA * intrinsic_r
            )

        # Action selection: softmax over accumulated intrinsic reward
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_action(self.reward_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None
        self._prev_action = None

    def mean_pred_error(self):
        if not self._pred_errors:
            return 0.0
        return float(np.mean(list(self._pred_errors)[-100:]))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    for seed in seeds:
        sub = ICM917(n_actions=n_actions, seed=seed)
        env = make_game(game_name)
        obs = env.reset(seed=seed * 1000)
        step = 0; completions = 0; current_level = 0
        while step < n_steps:
            if obs is None:
                obs = env.reset(seed=seed * 1000)
                sub.on_level_transition()
                continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
            obs, _, done, info = env.step(action)
            step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > current_level:
                completions += (cl - current_level)
                current_level = cl
                sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed * 1000)
                current_level = 0
                sub.on_level_transition()
        results.append(completions)
        print(f"    seed={seed}: L1={completions:4d}  mean_pred_error={sub.mean_pred_error():.3f}")
    return results


print("=" * 70)
print("STEP 917 — ICM (Pathak 2017) SIMPLIFIED ON LS20 + FT09")
print("=" * 70)
print("Forward model + intrinsic reward accumulation. No learned features, no inverse model.")
t0 = time.time()

print("\n--- FT09 (68 actions, 25K, 10 seeds) ---")
ft09_results = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS)
ft09_mean = np.mean(ft09_results)
ft09_zeros = sum(1 for x in ft09_results if x == 0)

print("\n--- LS20 (4 actions, 25K, 10 seeds) ---")
ls20_results = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS)
ls20_mean = np.mean(ls20_results)
ls20_zeros = sum(1 for x in ls20_results if x == 0)

print(f"\n{'='*70}")
print(f"STEP 917 RESULTS (ICM simplified):")
print(f"  FT09: L1={ft09_mean:.1f}/seed  std={np.std(ft09_results):.1f}  zero={ft09_zeros}/10")
print(f"  LS20: L1={ls20_mean:.1f}/seed  std={np.std(ls20_results):.1f}  zero={ls20_zeros}/10")
print(f"  {ft09_results}")
print(f"  {ls20_results}")
print(f"\nComparison:")
print(f"  895h cold LS20: 268.0/seed  FT09: 0/10")
print(f"  917 ICM  LS20: {ls20_mean:.1f}/seed  FT09: {ft09_mean:.1f}/seed")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 917 DONE")
