"""
step0910a_ls20_compression.py -- Alpha-weighted compression progress on LS20.

R3 hypothesis: compression progress in alpha-weighted space provides a local
sequential progress signal. Actions where prediction error is DECREASING are
productive; softmax selects them preferentially.

Step 855 failed (action collapse, all 256 dims, static dims dominated signal).
Step 910 measures in alpha-weighted space → only informative dims contribute →
cleaner signal. Alpha concentrates on dynamic dims → compression progress
reflects actual game-relevant transitions.

Architecture vs 895h:
- Alpha/W: identical (prediction-error attention, clamped 0.1-5.0)
- Action selector: CHANGED. 895h used delta_per_action (L2 norm change).
  910a uses delta_E_per_action (EMA of prediction error improvement per action).
  delta_E > 0: prediction improving for this action (productive click/move).
  delta_E ≈ 0: stagnated (action leads nowhere learnable).

Kill criterion: if delta_E_per_action doesn't differentiate actions within 5K steps.

Leo mail 2643. Protocol: LS20, 25K, 10 seeds, substrate_seed=seed, cold only.
Compare: 868d=203.9 (L2 norm baseline), 895h cold=268.0 (change-tracking baseline).
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
N_ACTIONS = 4
ETA_W = 0.01
ALPHA_EMA = 0.10
ALPHA_UPDATE_DELAY = 50
ENC_DIM = 256
EPSILON = 0.20
SOFTMAX_TEMP = 0.01      # Sharp: prefer actions with positive compression progress
COMPRESS_EMA = 0.10      # EMA for compression progress
ALPHA_LO = 0.10
ALPHA_HI = 5.00


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(delta, temp):
    x = delta / temp
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class AlphaCompression910(BaseSubstrate):
    """Alpha-weighted compression progress action selector."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        # Compression progress per action
        self.delta_E_per_action = np.zeros(n_actions, dtype=np.float32)
        self.prev_error_per_action = np.zeros(n_actions, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
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

        if self._prev_enc is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_enc * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp
            error_vec = (enc * self.alpha) - pred      # per-dim error vector
            scalar_error = float(np.linalg.norm(error_vec))

            # W update with gradient clipping (stability)
            err_norm = scalar_error
            if err_norm > 10.0:
                error_clip = error_vec * (10.0 / err_norm)
            else:
                error_clip = error_vec
            if not np.any(np.isnan(error_clip)):
                self.W -= ETA_W * np.outer(error_clip, inp)
                self._pred_errors.append(np.abs(error_vec))
                self._update_alpha()

            # Compression progress for this action
            a = self._prev_action
            prev_err = self.prev_error_per_action[a]
            if prev_err > 0:
                progress = prev_err - scalar_error  # positive = error decreased
                self.delta_E_per_action[a] = (0.9 * self.delta_E_per_action[a]
                                               + COMPRESS_EMA * progress)
            self.prev_error_per_action[a] = scalar_error

        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            probs = softmax_action(self.delta_E_per_action, SOFTMAX_TEMP)
            action = int(self._rng.choice(self._n_actions, p=probs))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def alpha_concentration(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def delta_E_range(self):
        return float(np.max(self.delta_E_per_action) - np.min(self.delta_E_per_action))

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + self._n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_E_per_action = np.zeros(self._n_actions, dtype=np.float32)
        self.prev_error_per_action = np.zeros(self._n_actions, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


print("=" * 70)
print("STEP 910a — ALPHA-WEIGHTED COMPRESSION PROGRESS ON LS20")
print("=" * 70)
print(f"Action selector: EMA of prediction error improvement (delta_E).")
print(f"Positive delta_E = prediction improving = productive action.")
print(f"Compare: 868d=203.9/seed (L2 baseline), 895h cold=268.0/seed.")
print(f"Kill check: delta_E_range after 5K steps (should differentiate actions).")
print(f"25K steps, 10 seeds cold, substrate_seed=seed.")

t0 = time.time()
comps = []
concs = []
delta_E_ranges = []

for ts in TEST_SEEDS:
    sub = AlphaCompression910(n_actions=N_ACTIONS, seed=ts)
    sub.reset(ts)
    env = make_game(); obs = env.reset(seed=ts * 1000)
    step = 0; completions = 0; current_level = 0
    early_range = None

    while step < TEST_STEPS:
        if obs is None:
            obs = env.reset(seed=ts * 1000); sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=ts * 1000); current_level = 0
            sub.on_level_transition()
        if step == 5000 and early_range is None:
            early_range = sub.delta_E_range()

    comps.append(completions)
    concs.append(sub.alpha_concentration())
    delta_E_ranges.append(sub.delta_E_range())
    print(f"  seed={ts}: L1={completions:4d}  alpha_conc={sub.alpha_concentration():.2f}  "
          f"delta_E_range={sub.delta_E_range():.4f}  "
          f"delta_E={[f'{x:.3f}' for x in sub.delta_E_per_action.tolist()]}  "
          f"early_range@5K={early_range:.4f}")

mean_L1 = np.mean(comps); std_L1 = np.std(comps)
zero_seeds = sum(1 for x in comps if x == 0)
mean_range = np.mean(delta_E_ranges)
print(f"\nLS20 910a cold: L1={mean_L1:.1f}/seed  std={std_L1:.1f}  zero={zero_seeds}/{len(comps)}")
print(f"      {comps}")
print(f"      alpha_conc={np.mean(concs):.2f}  delta_E_range={mean_range:.4f}")
print(f"\nComparison:")
print(f"  868d (raw L2 baseline):     203.9/seed  std=105.8  zero=1/10")
print(f"  895h cold (change-track):   268.0/seed  std=75.2   zero=0/10  ← BEST")
print(f"  910a cold (compression):    {mean_L1:.1f}/seed  std={std_L1:.1f}  zero={zero_seeds}/10")
if mean_range < 0.001:
    print(f"  KILL SIGNAL: delta_E_range={mean_range:.6f} — no action differentiation")
else:
    print(f"  Actions differentiated (mean delta_E_range={mean_range:.4f})")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 910a DONE")
