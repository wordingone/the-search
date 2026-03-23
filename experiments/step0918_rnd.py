"""
step0918_rnd.py -- RND (Burda 2018) simplified for our framework.

Random Network Distillation: fixed random target network + trained predictor.
Novelty = prediction error (high error = novel state the predictor hasn't seen).

Implementation:
- W_target: fixed random (256 → 64), NEVER updated
- W_pred: trainable (256 → 64), delta rule toward W_target outputs
- Novelty per observation = ||W_pred @ enc - W_target @ enc||
- Track novelty_per_action as EMA of novelty observed after each action
- Action: softmax over novelty_per_action (highest novelty = most novel next state)

Run: LS20 + FT09, 25K, 10 seeds, cold, substrate_seed=seed.
Baseline: 895h cold LS20=268.0/seed, FT09=0/10.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
RND_DIM = 64   # target/predictor output dim
ETA_W = 0.01
NOVELTY_EMA = 0.05
INIT_NOVELTY = 1.0
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def softmax_action(novelty, temp, rng):
    x = novelty / temp
    x = x - np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(novelty), p=probs))


class RND918:
    """RND-simplified: fixed target + trained predictor. EMA novelty per action."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 20000)  # separate seed for fixed weights

        # Fixed random target (never trained)
        self.W_target = rs.randn(RND_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Trainable predictor (learns to match target output)
        self.W_pred = np.zeros((RND_DIM, ENC_DIM), dtype=np.float32)

        # Novelty per action
        self.novelty_per_action = np.full(n_actions, INIT_NOVELTY, dtype=np.float32)

        # Running mean for centering
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        self._prev_action = None
        self._novelty_log = deque(maxlen=200)

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def process(self, obs):
        enc = self._encode(obs)

        # Compute novelty for current observation
        target_out = self.W_target @ enc
        pred_out = self.W_pred @ enc
        novelty = float(np.linalg.norm(pred_out - target_out))
        self._novelty_log.append(novelty)

        # Update predictor toward target
        error = pred_out - target_out
        err_norm = float(np.linalg.norm(error))
        if err_norm > 10.0:
            error = error * (10.0 / err_norm)
        if not np.any(np.isnan(error)):
            self.W_pred -= ETA_W * np.outer(error, enc)

        # Update novelty for previous action (reward after taking that action)
        if self._prev_action is not None:
            a = self._prev_action
            self.novelty_per_action[a] = (
                (1 - NOVELTY_EMA) * self.novelty_per_action[a] + NOVELTY_EMA * novelty
            )

        # Action selection: most novel expected next state
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_action(self.novelty_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_action = None

    def mean_novelty(self):
        if not self._novelty_log:
            return 0.0
        return float(np.mean(list(self._novelty_log)[-100:]))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    for seed in seeds:
        sub = RND918(n_actions=n_actions, seed=seed)
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
        print(f"    seed={seed}: L1={completions:4d}  mean_novelty={sub.mean_novelty():.3f}")
    return results


print("=" * 70)
print("STEP 918 — RND (Burda 2018) SIMPLIFIED ON LS20 + FT09")
print("=" * 70)
print("Fixed target W_target + trained predictor W_pred. Novelty = ||pred - target||.")
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
print(f"STEP 918 RESULTS (RND simplified):")
print(f"  FT09: L1={ft09_mean:.1f}/seed  std={np.std(ft09_results):.1f}  zero={ft09_zeros}/10")
print(f"  LS20: L1={ls20_mean:.1f}/seed  std={np.std(ls20_results):.1f}  zero={ls20_zeros}/10")
print(f"  {ft09_results}")
print(f"  {ls20_results}")
print(f"\nComparison:")
print(f"  895h cold LS20: 268.0/seed  FT09: 0/10")
print(f"  918 RND  LS20: {ls20_mean:.1f}/seed  FT09: {ft09_mean:.1f}/seed")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 918 DONE")
