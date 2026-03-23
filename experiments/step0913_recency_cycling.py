"""
step0913_recency_cycling.py -- Recency-weighted action cycling.

R3 hypothesis: global action recency forces the substrate to cycle through
actions systematically, preventing collapse onto one action. On FT09, this
guarantees all 68 click positions get tried periodically.

Architecture vs 895h:
- Alpha/W: identical (clamped 0.1-5.0, prediction error attention)
- Action selector: COMBINED. Score = alpha_weighted_delta + recency_bonus * age.
  age[a] = (step - last_used[a]) — steps since action a was last taken.
  NOT per-(state,action) — global recency only. NOT graph-banned.

Mechanism:
  last_used[a] = step when action a was last taken
  age[a] = current_step - last_used[a]
  score[a] = delta_per_action[a] + recency_weight * age[a]
  Softmax over scores → prefers both high-change AND unused actions.

Key parameter: recency_weight controls how strongly unused actions are preferred.
Too high → ignores change signal (uniform cycling). Too low → same as 895h.
Test: recency_weight = 0.01 (mild), 0.1 (strong).

Leo mail 2647. Protocol: LS20+FT09, 25K, 10 seeds cold, substrate_seed=seed.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ENC_DIM = 256
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
RECENCY_WEIGHT = 0.05   # balance between change signal and recency


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(scores, temp):
    x = scores / temp
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class RecencyCycling913(BaseSubstrate):
    """895h + global recency bonus. NOT per-state. NOT graph-banned."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON,
                 recency_weight=RECENCY_WEIGHT):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._recency_weight = recency_weight
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self.last_used = np.zeros(n_actions, dtype=np.int64)   # step when last used
        self._pred_errors = deque(maxlen=200)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._step = 0
        self._prev_enc = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY:
            return
        mean_errors = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(mean_errors)) or np.any(np.isinf(mean_errors)):
            return
        raw_alpha = np.sqrt(np.clip(mean_errors, 0, 1e6) + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8 or np.isnan(mean_raw):
            return
        self.alpha = raw_alpha / mean_raw
        self.alpha = np.clip(self.alpha, ALPHA_LO, ALPHA_HI)

    def process(self, observation):
        enc = self._encode(observation)
        self._step += 1

        if self._prev_enc is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_enc * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp
            error = (enc * self.alpha) - pred

            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            weighted_delta = (enc - self._prev_enc) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * change)
            self.last_used[a] = self._step

        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            # Combined score: change signal + recency bonus
            age = (self._step - self.last_used).astype(np.float32)
            scores = self.delta_per_action + self._recency_weight * age
            probs = softmax_action(scores, SOFTMAX_TEMP)
            action = int(self._rng.choice(self._n_actions, p=probs))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def alpha_concentration(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def recency_coverage(self):
        """Fraction of actions used in the last 200 steps."""
        threshold = max(0, self._step - 200)
        return float(np.sum(self.last_used >= threshold)) / self._n_actions

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + self._n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_per_action = np.full(self._n_actions, INIT_DELTA, dtype=np.float32)
        self.last_used = np.zeros(self._n_actions, dtype=np.int64)
        self._pred_errors = deque(maxlen=200)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._step = 0
        self._prev_enc = None; self._prev_action = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def run_game(game_name, n_actions, test_steps, recency_weight=RECENCY_WEIGHT):
    def make_game():
        try:
            import arcagi3; return arcagi3.make(game_name)
        except:
            import util_arcagi3; return util_arcagi3.make(game_name)

    comps = []; concs = []; coverages = []
    for ts in TEST_SEEDS:
        sub = RecencyCycling913(n_actions=n_actions, seed=ts, recency_weight=recency_weight)
        sub.reset(ts)
        env = make_game(); obs = env.reset(seed=ts * 1000)
        step = 0; completions = 0; current_level = 0
        while step < test_steps:
            if obs is None:
                obs = env.reset(seed=ts * 1000); sub.on_level_transition(); continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
            obs, _, done, info = env.step(action); step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > current_level:
                completions += (cl - current_level); current_level = cl
                sub.on_level_transition()
            if done:
                obs = env.reset(seed=ts * 1000); current_level = 0
                sub.on_level_transition()
        comps.append(completions)
        concs.append(sub.alpha_concentration())
        coverages.append(sub.recency_coverage())
        print(f"  {game_name} seed={ts}: L1={completions:4d}  alpha_conc={sub.alpha_concentration():.2f}  coverage={sub.recency_coverage():.3f}")
    return comps, concs, coverages


print("=" * 70)
print("STEP 913 — RECENCY-WEIGHTED ACTION CYCLING")
print("=" * 70)
print(f"Score = delta_per_action + {RECENCY_WEIGHT} * age. NOT per-state. NOT graph-banned.")
print(f"Forces cycling through all actions. FT09: 68 tiles get periodic visits.")
print(f"25K steps, 10 seeds cold, substrate_seed=seed.")

t0 = time.time()

print(f"\n--- LS20 (4 actions, recency_weight={RECENCY_WEIGHT}) ---")
ls20_comps, ls20_concs, ls20_covs = run_game("LS20", 4, TEST_STEPS)
ls20_mean = np.mean(ls20_comps); ls20_std = np.std(ls20_comps)
ls20_zero = sum(1 for x in ls20_comps if x == 0)
print(f"\nLS20 913 cold: L1={ls20_mean:.1f}/seed  std={ls20_std:.1f}  zero={ls20_zero}/10")
print(f"      {ls20_comps}")

print(f"\n--- FT09 (68 actions, recency_weight={RECENCY_WEIGHT}) ---")
ft09_comps, ft09_concs, ft09_covs = run_game("FT09", 68, TEST_STEPS)
ft09_mean = np.mean(ft09_comps); ft09_std = np.std(ft09_comps)
ft09_zero = sum(1 for x in ft09_comps if x == 0)
print(f"\nFT09 913 cold: L1={ft09_mean:.1f}/seed  std={ft09_std:.1f}  zero={ft09_zero}/10")
print(f"      {ft09_comps}")

print(f"\n{'='*70}")
print(f"COMPARISON:")
print(f"  895h cold (change-track, LS20):    268.0/seed  0/10 zeros  ← BEST")
print(f"  868d (raw L2, LS20):               203.9/seed  1/10 zeros  ← BASELINE")
print(f"  913 cold (recency, LS20):          {ls20_mean:.1f}/seed  std={ls20_std:.1f}  zero={ls20_zero}/10")
print(f"  895f cold (change-track, FT09):      0.0/seed  10/10 zeros ← FT09 baseline")
print(f"  913 cold (recency, FT09):          {ft09_mean:.1f}/seed  std={ft09_std:.1f}  zero={ft09_zero}/10")
print(f"  Recency coverage (LS20): {np.mean(ls20_covs):.3f}  (FT09): {np.mean(ft09_covs):.3f}")
if ft09_mean > 0:
    print(f"  BREAKTHROUGH: recency cycling achieves L1>0 on FT09!")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 913 DONE")
