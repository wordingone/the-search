"""Step 944: Ashby threshold reset — self-triggered alpha redistribution.

R3 hypothesis: Self-triggered discontinuous modification (state-dependent
restructuring) maintains discriminative capacity that frozen continuous
updates cannot. If threshold reset + continuous outperforms continuous-only,
self-triggered modification adds R3-relevant capability.

Mechanism: when alpha_conc (max/min) > THETA_CONC, reset alpha to uniform.
The trigger rule is frozen, but the modification is self-triggered — the
system's own dynamics determine WHEN restructuring occurs.

Concrete prediction: alpha_conc < 30 where 916-alone reaches ~50.
Kill: LS20 < 250 or chain kill (one game improves while other degrades).

Base: step0916_recurrent_trajectory.py (unchanged except reset addition).
Concentration metric: max/min (matches 916 — range 1..50 with clamp [0.1, 5.0]).
THETA_CONC values tested: 30, 20, 40.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM   # 320
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
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


class AshbyReset944:
    """916 + Ashby threshold reset: self-triggered alpha redistribution."""

    def __init__(self, n_actions, seed, theta_conc=30.0, reset_w_pred=False):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)  # fixed substrate seed
        self._theta_conc = theta_conc
        self._reset_w_pred = reset_w_pred
        self._reset_count = 0

        # Fixed random recurrent weights (never trained)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Trainable forward model on extended encoding
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), dtype=np.float32)

        # Alpha on extended encoding
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)

        # 800b change-tracking
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)

        # Recurrent state
        self.h = np.zeros(H_DIM, dtype=np.float32)

        # Running mean for centering
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        # History buffers
        self._pred_errors = deque(maxlen=200)
        self._prev_ext = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        return np.concatenate([enc, self.h]).astype(np.float32)

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

        # Ashby threshold reset: self-triggered discontinuous restructuring.
        # max/min matches 916's alpha_conc formula: range 1..50 with clamp [0.1, 5.0].
        # THETA=30 fires at 60% of max concentration (before full 50 degeneration).
        alpha_conc_maxmin = float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))
        if alpha_conc_maxmin > self._theta_conc:
            self.alpha = np.ones(EXT_DIM, dtype=np.float32)
            if self._reset_w_pred:
                self.W_pred *= 0.5
            self._reset_count += 1

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
                self.W_pred -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

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

    def alpha_conc_maxmin(self):
        """916-compatible concentration metric: max/min, range 1..50."""
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def alpha_conc_maxmean(self):
        return float(np.max(self.alpha) / (np.mean(self.alpha) + 1e-8))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps, theta_conc, reset_w_pred=False):
    results = []
    concs = []
    reset_counts = []
    for seed in seeds:
        sub = AshbyReset944(n_actions=n_actions, seed=seed,
                            theta_conc=theta_conc, reset_w_pred=reset_w_pred)
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
        concs.append(sub.alpha_conc_maxmin())
        reset_counts.append(sub._reset_count)
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc(max/min)={sub.alpha_conc_maxmin():.2f}  resets={sub._reset_count}")
    return results, concs, reset_counts


if __name__ == "__main__":
    import os
    import time

    print("=" * 70)
    print("STEP 944 — ASHBY THRESHOLD RESET")
    print("=" * 70)
    print("R3: self-triggered alpha reset when alpha_conc(max/min) > THETA_CONC.")
    print("Base: 916. 10K steps. LS20 + FT09 for chain kill check.")
    t0 = time.time()

    # Game version hashes
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), 'unknown')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), 'unknown')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")
    print()

    for THETA in [30, 20, 40]:
        print(f"\n{'='*70}")
        print(f"THETA_CONC = {THETA}")
        print(f"{'='*70}")

        print(f"\n--- LS20 (4 actions, 10K steps, THETA={THETA}) ---")
        ls20_results, ls20_concs, ls20_resets = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS, THETA)
        ls20_mean = np.mean(ls20_results)
        ls20_std = np.std(ls20_results)
        ls20_zeros = sum(1 for x in ls20_results if x == 0)
        ls20_mean_conc = np.mean(ls20_concs)
        ls20_mean_resets = np.mean(ls20_resets)
        print(f"  LS20: L1={ls20_mean:.1f}/seed  std={ls20_std:.1f}  zero={ls20_zeros}/10")
        print(f"  alpha_conc(max/min)={ls20_mean_conc:.2f}  resets/seed={ls20_mean_resets:.1f}")
        print(f"  {ls20_results}")

        print(f"\n--- FT09 (68 actions, 10K steps, THETA={THETA}) ---")
        ft09_results, ft09_concs, ft09_resets = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS, THETA)
        ft09_mean = np.mean(ft09_results)
        ft09_std = np.std(ft09_results)
        ft09_zeros = sum(1 for x in ft09_results if x == 0)
        ft09_mean_conc = np.mean(ft09_concs)
        ft09_mean_resets = np.mean(ft09_resets)
        print(f"  FT09: L1={ft09_mean:.1f}/seed  std={ft09_std:.1f}  zero={ft09_zeros}/10")
        print(f"  alpha_conc(max/min)={ft09_mean_conc:.2f}  resets/seed={ft09_mean_resets:.1f}")
        print(f"  {ft09_results}")

        # Chain kill verdict
        ls20_vs_baseline = "UNKNOWN"
        ft09_vs_baseline = "UNKNOWN"
        print(f"\n  CHAIN VERDICT (THETA={THETA}):")
        print(f"    LS20={ls20_mean:.1f}  FT09={ft09_mean:.1f}")
        if ls20_mean_resets == 0:
            print(f"    MECHANISM INERT — reset never fired (THETA too high for this game)")
        elif ls20_mean < 250:
            print(f"    KILL: LS20 below 250 threshold")
        elif ls20_mean_resets > ls20_mean_resets and ls20_mean_resets * 10 > TEST_STEPS:
            print(f"    NOTE: Reset fires >10%/step — THETA may be too low")
        else:
            print(f"    PASS threshold check")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 944 DONE")
