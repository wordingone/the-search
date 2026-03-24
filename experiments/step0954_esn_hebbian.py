"""Steps 954 + 954b: Echo State Network — new family.

Hebbian RNN family dead (948-953): W_a bootstrapping from near-zero W_h requires
lucky init. Structural gap: minimal recurrence → h ≈ input-only → W_a sees same
activations from similar states → winner-take-all lock.

R3 hypothesis: ESN's fixed W_h at spectral radius 0.9 provides strong recurrence
from step 1. h IS trajectory-dependent immediately. W_a @ h varies across positions
→ less winner-take-all lock → robust Hebbian bootstrapping.

Step 954: ESN + Hebbian W_a (same rule as 948, tanh h, sparse W_h at sr=0.9)
Step 954b: ESN + random fixed W_a (diagnostic — does reservoir alone produce signal?)

Kill: 954 AND 954b both 0/10 → reservoir doesn't help.
If 954 > 954b → W_a learning helps → iterate.
If 954b > 954 → fixed readout better → very interesting.
If 954 ≥ 3/10 → ESN fixes bootstrapping → iterate on reservoir params.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
SPECTRAL_RADIUS = 0.9
SPARSITY = 0.1     # keep 10% of connections
ETA_PRED = 0.01
ETA_ACTION = 0.001
EPSILON = 0.20
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000


def make_reservoir(h_dim, spectral_radius, sparsity, rs):
    """Build fixed ESN recurrent matrix: scaled to sr, then sparsified."""
    W = rs.randn(h_dim, h_dim).astype(np.float32)
    # Scale to spectral radius
    eigvals = np.abs(np.linalg.eigvals(W))
    W *= spectral_radius / eigvals.max()
    # Sparsify: keep 10% of connections
    mask = (rs.random(W.shape) < sparsity).astype(np.float32)
    W *= mask
    return W


class ESNHebbian954:
    """Echo State Network + Hebbian W_a. Fixed W_h/W_x, trained W_pred/W_a."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # FIXED reservoir
        self.W_h = make_reservoir(H_DIM, SPECTRAL_RADIUS, SPARSITY, rs)
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # TRAINED
        self.W_pred = np.zeros((ENC_DIM, H_DIM), dtype=np.float32)
        self.W_a = np.zeros((n_actions, H_DIM), dtype=np.float32)

        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_h = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)  # tanh, not sigmoid
        return enc

    def process(self, obs):
        enc = self._encode(obs)

        if self._prev_h is not None and self._prev_action is not None:
            pred = self.W_pred @ self._prev_h
            error = enc - pred
            delta = float(np.linalg.norm(error))

            if delta > 10.0:
                error = error * (10.0 / delta)
                delta = 10.0

            if not np.any(np.isnan(error)):
                self.W_pred += ETA_PRED * np.outer(error, self._prev_h)
                self.W_a[self._prev_action] += ETA_ACTION * delta * self.h

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            scores = self.W_a @ self.h
            action = int(np.argmax(scores))

        self._prev_h = self.h.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_h = None
        self._prev_action = None

    def w_a_norm(self):
        return float(np.linalg.norm(self.W_a))


class ESNFixedReadout954b:
    """Echo State Network + fixed random W_a (diagnostic). W_a never updated."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # FIXED reservoir
        self.W_h = make_reservoir(H_DIM, SPECTRAL_RADIUS, SPARSITY, rs)
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # TRAINED prediction
        self.W_pred = np.zeros((ENC_DIM, H_DIM), dtype=np.float32)

        # FIXED random readout — never updated
        ra = np.random.RandomState(seed + 20000)
        self.W_a = ra.randn(n_actions, H_DIM).astype(np.float32) * 0.1

        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_h = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        return enc

    def process(self, obs):
        enc = self._encode(obs)

        if self._prev_h is not None:
            pred = self.W_pred @ self._prev_h
            error = enc - pred
            if np.linalg.norm(error) > 10.0:
                error = error * (10.0 / np.linalg.norm(error))
            if not np.any(np.isnan(error)):
                self.W_pred += ETA_PRED * np.outer(error, self._prev_h)

        # Fixed readout: argmax + epsilon, W_a never changes
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            scores = self.W_a @ self.h
            action = int(np.argmax(scores))

        self._prev_h = self.h.copy()
        return action

    def on_level_transition(self):
        self._prev_h = None


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(SubstrateClass, game_name, n_actions, seeds, n_steps, **kwargs):
    results = []
    for seed in seeds:
        sub = SubstrateClass(n_actions=n_actions, seed=seed, **kwargs)
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
        w_norm = sub.w_a_norm() if hasattr(sub, 'w_a_norm') else 0.0
        print(f"    seed={seed}: L1={completions:4d}  W_a_norm={w_norm:.3f}")
    return results


if __name__ == "__main__":
    import os, time

    print("=" * 70)
    print("STEPS 954 + 954b — ECHO STATE NETWORK (new family)")
    print("=" * 70)
    t0 = time.time()

    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), 'unknown')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), 'unknown')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")
    print(f"H_DIM={H_DIM}  SPECTRAL_RADIUS={SPECTRAL_RADIUS}  SPARSITY={SPARSITY}")
    print(f"ETA_PRED={ETA_PRED}  ETA_ACTION={ETA_ACTION}  EPSILON={EPSILON}")

    # === Step 954: ESN + Hebbian W_a ===
    print()
    print("=" * 70)
    print("STEP 954: ESN + HEBBIAN W_a")
    print("=" * 70)

    print("\n--- LS20 (4 actions, 10K steps) ---")
    ls954_r = run_game(ESNHebbian954, "LS20", 4, TEST_SEEDS, TEST_STEPS)
    ls954_mean = np.mean(ls954_r)
    ls954_nz = sum(1 for x in ls954_r if x > 0)
    print(f"  LS20: L1={ls954_mean:.1f}/seed  nonzero={ls954_nz}/10  {ls954_r}")

    print("\n--- FT09 (68 actions, 10K steps) ---")
    ft954_r = run_game(ESNHebbian954, "FT09", 68, TEST_SEEDS, TEST_STEPS)
    ft954_mean = np.mean(ft954_r)
    ft954_nz = sum(1 for x in ft954_r if x > 0)
    print(f"  FT09: L1={ft954_mean:.1f}/seed  nonzero={ft954_nz}/10  {ft954_r}")

    # === Step 954b: ESN + fixed random W_a ===
    print()
    print("=" * 70)
    print("STEP 954b: ESN + FIXED RANDOM W_a (diagnostic)")
    print("=" * 70)

    print("\n--- LS20 (4 actions, 10K steps) ---")
    ls954b_r = run_game(ESNFixedReadout954b, "LS20", 4, TEST_SEEDS, TEST_STEPS)
    ls954b_mean = np.mean(ls954b_r)
    ls954b_nz = sum(1 for x in ls954b_r if x > 0)
    print(f"  LS20: L1={ls954b_mean:.1f}/seed  nonzero={ls954b_nz}/10  {ls954b_r}")

    print("\n--- FT09 (68 actions, 10K steps) ---")
    ft954b_r = run_game(ESNFixedReadout954b, "FT09", 68, TEST_SEEDS, TEST_STEPS)
    ft954b_mean = np.mean(ft954b_r)
    ft954b_nz = sum(1 for x in ft954b_r if x > 0)
    print(f"  FT09: L1={ft954b_mean:.1f}/seed  nonzero={ft954b_nz}/10  {ft954b_r}")

    # === Verdict ===
    print()
    print("=" * 70)
    print("COMBINED VERDICT:")
    print(f"  954 (Hebbian):  LS20={ls954_mean:.1f}  FT09={ft954_mean:.1f}  nonzero={ls954_nz}/10")
    print(f"  954b (Fixed):   LS20={ls954b_mean:.1f}  FT09={ft954b_mean:.1f}  nonzero={ls954b_nz}/10")

    if ls954_nz == 0 and ls954b_nz == 0:
        verdict = "KILL — reservoir doesn't help, structural limit of h→action pathway"
    elif ls954b_nz > ls954_nz:
        verdict = "INTERESTING — fixed readout beats Hebbian, W_a learning hurts"
    elif ls954_nz >= 3:
        verdict = f"SUCCESS — {ls954_nz}/10 seeds, ESN fixes bootstrapping"
    elif ls954_nz > ls954b_nz:
        verdict = f"SIGNAL — W_a learning helps ({ls954_nz} vs {ls954b_nz}), iterate"
    else:
        verdict = f"NEUTRAL — 954={ls954_nz}/10, 954b={ls954b_nz}/10"

    print(f"  VERDICT: {verdict}")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEPS 954+954b DONE")
