"""
step0912_delta_direction.py -- Temporal difference action selection.

R3 hypothesis: tracking DIRECTION of change per action (not just magnitude)
allows the substrate to break out of loops by seeking directional novelty.

Architecture vs 895h:
- Alpha/W: identical (clamped 0.1-5.0, prediction error attention)
- Action selector: CHANGED. Track EMA of (enc - prev_enc) per action.
  Action selection: prefer actions whose PAST delta_direction differs most
  from the CURRENT delta. This forces directional exploration.

Mechanism:
  delta_direction[a] = EMA of (enc_after_a - enc_before_a) — per-dim direction
  novelty[a] = ||delta_direction[a] - current_delta||  (how different this action's
               typical trajectory is from what we just saw)
  Softmax over novelty → prefer direction-changing actions.

On FT09: clicking same tile repeatedly → same delta_direction. After N repeats,
current_delta matches delta_direction[that tile]. ALL OTHER tiles have different
directions → substrate naturally sequences through tiles.

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
DIR_EMA = 0.10         # EMA for delta_direction per action
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ENC_DIM = 256
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(scores, temp):
    x = scores / temp
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class DeltaDirection912(BaseSubstrate):
    """Temporal difference action selection: seek directional novelty."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        # Per-action: EMA of (enc - prev_enc) direction vectors
        self.delta_direction = np.zeros((n_actions, ENC_DIM), dtype=np.float32)
        # Also track magnitude for comparison (same as 895h)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None
        self._current_delta = np.zeros(ENC_DIM, dtype=np.float32)

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
            # W update (alpha-weighted prediction error)
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

            # Delta direction update (alpha-weighted)
            delta = (enc - self._prev_enc) * self.alpha
            a = self._prev_action
            self.delta_direction[a] = ((1 - DIR_EMA) * self.delta_direction[a]
                                        + DIR_EMA * delta)
            # Magnitude update (same as 895h)
            change = float(np.linalg.norm(delta))
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * change)
            self._current_delta = delta.copy()

        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            # Score = directional novelty: how different is this action's typical
            # direction from what we just observed?
            novelty = np.array([
                float(np.linalg.norm(self.delta_direction[a] - self._current_delta))
                for a in range(self._n_actions)
            ], dtype=np.float32)
            probs = softmax_action(novelty, SOFTMAX_TEMP)
            action = int(self._rng.choice(self._n_actions, p=probs))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def alpha_concentration(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def direction_diversity(self):
        """How different are the per-action direction vectors from each other?"""
        norms = [float(np.linalg.norm(self.delta_direction[a])) for a in range(self._n_actions)]
        return float(np.std(norms))

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + self._n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_direction = np.zeros((self._n_actions, ENC_DIM), dtype=np.float32)
        self.delta_per_action = np.full(self._n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None
        self._current_delta = np.zeros(ENC_DIM, dtype=np.float32)

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def run_game(game_name, n_actions, test_steps):
    def make_game():
        try:
            import arcagi3; return arcagi3.make(game_name)
        except:
            import util_arcagi3; return util_arcagi3.make(game_name)

    comps = []; concs = []; diversities = []
    for ts in TEST_SEEDS:
        sub = DeltaDirection912(n_actions=n_actions, seed=ts)
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
        diversities.append(sub.direction_diversity())
        print(f"  {game_name} seed={ts}: L1={completions:4d}  alpha_conc={sub.alpha_concentration():.2f}  dir_diversity={sub.direction_diversity():.4f}")
    return comps, concs, diversities


print("=" * 70)
print("STEP 912 — TEMPORAL DIFFERENCE ACTION SELECTION")
print("=" * 70)
print(f"Seek actions with delta_direction DIFFERENT from current delta.")
print(f"Directional novelty forces trajectory-breaking → sequence discovery.")
print(f"25K steps, 10 seeds cold, substrate_seed=seed.")

t0 = time.time()

print(f"\n--- LS20 (4 actions) ---")
ls20_comps, ls20_concs, ls20_divs = run_game("LS20", 4, TEST_STEPS)
ls20_mean = np.mean(ls20_comps); ls20_std = np.std(ls20_comps)
ls20_zero = sum(1 for x in ls20_comps if x == 0)
print(f"\nLS20 912 cold: L1={ls20_mean:.1f}/seed  std={ls20_std:.1f}  zero={ls20_zero}/10")
print(f"      {ls20_comps}")

print(f"\n--- FT09 (68 actions) ---")
ft09_comps, ft09_concs, ft09_divs = run_game("FT09", 68, TEST_STEPS)
ft09_mean = np.mean(ft09_comps); ft09_std = np.std(ft09_comps)
ft09_zero = sum(1 for x in ft09_comps if x == 0)
print(f"\nFT09 912 cold: L1={ft09_mean:.1f}/seed  std={ft09_std:.1f}  zero={ft09_zero}/10")
print(f"      {ft09_comps}")

print(f"\n{'='*70}")
print(f"COMPARISON:")
print(f"  895h cold (change-track, LS20):    268.0/seed  0/10 zeros  ← BEST")
print(f"  868d (raw L2, LS20):               203.9/seed  1/10 zeros  ← BASELINE")
print(f"  912 cold (delta-dir, LS20):        {ls20_mean:.1f}/seed  std={ls20_std:.1f}  zero={ls20_zero}/10")
print(f"  895f cold (change-track, FT09):      0.0/seed  10/10 zeros ← FT09 baseline")
print(f"  912 cold (delta-dir, FT09):        {ft09_mean:.1f}/seed  std={ft09_std:.1f}  zero={ft09_zero}/10")
if ft09_mean > 0:
    print(f"  BREAKTHROUGH: delta-direction achieves L1>0 on FT09!")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 912 DONE")
