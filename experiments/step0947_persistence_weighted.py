"""Step 947: Persistence-weighted prediction error for alpha weighting.

R3 hypothesis: Persistence-weighted error produces more discriminative alpha.
Meaningful change persists across steps; noise reverses. Weighting error by
running-mean drift rate concentrates alpha on progress-correlated dims.

Mechanism: track d_mean_ema = EMA of |delta running_mean| per dim.
persistence_weight = 1.0 + PERSISTENCE_SCALE * d_mean_ema.
Multiply error by persistence_weight before alpha update.

916 intact: alpha input unchanged, W_pred update unchanged, 800b unchanged.
Dimensionality: running_mean covers ENC_DIM (256D). H dims (256-319)
get persistence_weight=1.0 (no weighting — only ENC dims weighted).

Kill: LS20 < 72.7 (916 baseline at 10K).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
D_MEAN_EMA_DECAY = 0.95
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(delta, temp, rng):
    x = delta / temp
    x = x - np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class PersistenceWeighted947:
    """916 + persistence-weighted error. Alpha feedback loop intact."""

    def __init__(self, n_actions, seed, persistence_scale=1.0):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self._persistence_scale = persistence_scale

        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self.h = np.zeros(H_DIM, dtype=np.float32)

        # Running mean and its drift tracker (ENC_DIM only)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._prev_running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._d_mean_ema = np.zeros(ENC_DIM, dtype=np.float32)  # persistence signal
        self._n_obs = 0

        self._pred_errors = deque(maxlen=200)
        self._prev_ext = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._prev_running_mean = self._running_mean.copy()
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)

        # Update persistence signal: how fast is running_mean changing?
        d_mean = np.abs(self._running_mean - self._prev_running_mean)
        self._d_mean_ema = D_MEAN_EMA_DECAY * self._d_mean_ema + (1 - D_MEAN_EMA_DECAY) * d_mean

        return np.concatenate([enc, self.h]).astype(np.float32)

    def _persistence_weight(self):
        """EXT_DIM weight vector: ENC dims weighted by drift, H dims = 1.0."""
        enc_weight = 1.0 + self._persistence_scale * self._d_mean_ema
        h_weight = np.ones(H_DIM, dtype=np.float32)
        return np.concatenate([enc_weight, h_weight])

    def _update_alpha(self, weighted_error):
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

    def process(self, obs):
        ext_enc = self._encode(obs)

        if self._prev_ext is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_ext * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W_pred @ inp
            error = (ext_enc * self.alpha) - pred

            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)

            if not np.any(np.isnan(error)):
                # Apply persistence weighting before alpha update
                pw = self._persistence_weight()
                weighted_error = error * pw

                self.W_pred -= ETA_W * np.outer(error, inp)  # W_pred update unchanged
                self._pred_errors.append(np.abs(weighted_error))  # weighted for alpha
                self._update_alpha(weighted_error)

            # 800b unchanged
            weighted_delta = (ext_enc - self._prev_ext) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1 - ALPHA_EMA) * self.delta_per_action[a] + ALPHA_EMA * change

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_action(self.delta_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_ext = ext_enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def d_mean_top_dims(self, k=5):
        return list(np.argsort(self._d_mean_ema)[-k:][::-1])


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps, persistence_scale):
    results = []
    concs = []
    for seed in seeds:
        sub = PersistenceWeighted947(n_actions=n_actions, seed=seed,
                                     persistence_scale=persistence_scale)
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
        concs.append(sub.alpha_conc())
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}  persist_top={sub.d_mean_top_dims(3)}")
    return results, concs


if __name__ == "__main__":
    import os, time

    print("=" * 70)
    print("STEP 947 — PERSISTENCE-WEIGHTED PREDICTION ERROR")
    print("=" * 70)
    t0 = time.time()

    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), 'unknown')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), 'unknown')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")

    for SCALE in [1.0, 5.0]:
        print(f"\n{'='*70}\nPERSISTENCE_SCALE = {SCALE}\n{'='*70}")

        print(f"\n--- LS20 (4 actions, 10K steps, scale={SCALE}) ---")
        ls20_r, ls20_c = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS, SCALE)
        ls20_mean = np.mean(ls20_r)
        print(f"  LS20: L1={ls20_mean:.1f}/seed  zero={sum(1 for x in ls20_r if x==0)}/10  alpha_conc={np.mean(ls20_c):.2f}")
        print(f"  {ls20_r}")

        print(f"\n--- FT09 (68 actions, 10K steps, scale={SCALE}) ---")
        ft09_r, ft09_c = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS, SCALE)
        ft09_mean = np.mean(ft09_r)
        print(f"  FT09: L1={ft09_mean:.1f}/seed  zero={sum(1 for x in ft09_r if x==0)}/10  alpha_conc={np.mean(ft09_c):.2f}")
        print(f"  {ft09_r}")

        verdict = "KILL" if ls20_mean < 72.7 else "PASS"
        print(f"\n  VERDICT (scale={SCALE}): LS20={ls20_mean:.1f}  FT09={ft09_mean:.1f}  → {verdict}")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 947 DONE")
