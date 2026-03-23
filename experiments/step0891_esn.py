"""
step0891_esn.py -- Echo State Network (ESN) forward model.

R3 hypothesis: fixed random reservoir W_res (spectral radius < 1) + trainable W_out
achieves higher prediction accuracy than linear W alone. Reservoir nonlinearity gives
ESN a capacity advantage with only O(n) trainable params (W_out).

Architecture:
- W_res: (RESERVOIR, RESERVOIR). Fixed random sparse matrix. Spectral radius 0.95.
- W_in: (RESERVOIR, 256 + n_actions). Fixed random input weights. Scaled.
- W_out: (256, RESERVOIR). Trainable. Delta rule, eta=0.01.
- state: running reservoir state h (RESERVOIR-dim). Updated: h = tanh(W_res @ h + W_in @ inp)
- Prediction: pred = W_out @ h (predict next encoding from reservoir state)

Action: prediction-contrast (argmax ||pred - enc||^2 over actions).

R3_cf protocol:
- Pretrain: seeds 1-5, 5K each. W_out learns dynamics. h reset on level transition.
- Cold test: fresh W_out + h=0, seeds 6-10, 10K each.
- Warm test: pretrained W_out (W-only transfer, fresh h + running_mean), seeds 6-10.
- Metric: pred accuracy + L1 completions.
- Baseline: linear W cold=10.37% (Step 835), MLP expected to improve.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

PRETRAIN_SEEDS = list(range(1, 6))
TEST_SEEDS = list(range(6, 11))
PRETRAIN_STEPS = 5_000
TEST_STEPS = 10_000
N_ACTIONS = 4
ETA = 0.01
ENC_DIM = 256
RESERVOIR = 256
SPECTRAL_RADIUS = 0.95
SPARSITY = 0.1  # fraction of non-zero connections in W_res


def make_reservoir(n, sparsity, spectral_radius, rng):
    """Create sparse random reservoir with given spectral radius."""
    W = rng.randn(n, n).astype(np.float32)
    mask = rng.random((n, n)) > sparsity
    W[mask] = 0.0
    # Scale to target spectral radius
    eigenvalues = np.linalg.eigvals(W)
    max_abs_eig = np.max(np.abs(eigenvalues))
    if max_abs_eig > 1e-8:
        W *= spectral_radius / max_abs_eig
    return W


class ESN891(BaseSubstrate):
    """Echo State Network forward model. Fixed reservoir + trainable W_out."""

    def __init__(self, n_actions=4, seed=0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + n_actions
        # Fixed components (frozen)
        self.W_res = make_reservoir(RESERVOIR, SPARSITY, SPECTRAL_RADIUS, rng)
        self.W_in = (rng.randn(RESERVOIR, inp_dim) * 0.1).astype(np.float32)
        # Trainable
        self.W_out = np.zeros((ENC_DIM, RESERVOIR), dtype=np.float32)
        # State
        self._h = np.zeros(RESERVOIR, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None
        self._prev_h = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self._running_mean = (1 - alpha) * self._running_mean + alpha * enc_raw
        return enc_raw - self._running_mean

    def _encode_for_pred(self, obs):
        return _enc_frame(np.asarray(obs, dtype=np.float32)) - self._running_mean

    def _update_reservoir(self, enc, action):
        a_oh = np.zeros(self._n_actions, np.float32)
        a_oh[action] = 1.0
        inp = np.concatenate([enc, a_oh])
        self._h = np.tanh(self.W_res @ self._h + self.W_in @ inp)
        return self._h.copy()

    def predict_from_h(self, h):
        return self.W_out @ h

    def _update_W_out(self, h, target_enc):
        pred = self.W_out @ h
        err = pred - target_enc
        self.W_out -= ETA * np.outer(err, h)
        return float(np.sum(err**2)), float(np.sum(target_enc**2)) + 1e-8

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc

        # Update W_out from previous step prediction
        if self._prev_h is not None:
            self._update_W_out(self._prev_h, enc)

        # Action selection: argmax predicted change
        best_a, best_score = 0, -1.0
        for a in range(self._n_actions):
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[a] = 1.0
            inp = np.concatenate([enc, a_oh])
            h_next = np.tanh(self.W_res @ self._h + self.W_in @ inp)
            pred = self.predict_from_h(h_next)
            score = float(np.sum((pred - enc)**2))
            if score > best_score:
                best_score = score; best_a = a

        # Update reservoir with chosen action
        prev_h_before = self._prev_h
        self._prev_h = self._update_reservoir(enc, best_a)
        self._prev_enc = enc.copy()
        self._prev_action = best_a
        return best_a

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + self._n_actions
        self.W_res = make_reservoir(RESERVOIR, SPARSITY, SPECTRAL_RADIUS, rng)
        self.W_in = (rng.randn(RESERVOIR, inp_dim) * 0.1).astype(np.float32)
        self.W_out = np.zeros((ENC_DIM, RESERVOIR), dtype=np.float32)
        self._h = np.zeros(RESERVOIR, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None
        self._last_enc = None; self._prev_h = None

    def on_level_transition(self):
        self._h = np.zeros(RESERVOIR, dtype=np.float32)
        self._prev_enc = None; self._prev_action = None; self._prev_h = None

    def get_state(self):
        return {"W_out": self.W_out.copy(), "W_res": self.W_res.copy(),
                "W_in": self.W_in.copy(), "running_mean": self._running_mean.copy(),
                "n_obs": self._n_obs}

    def set_state(self, s):
        self.W_out = s["W_out"].copy()
        self.W_res = s["W_res"].copy()
        self.W_in = s["W_in"].copy()
        self._running_mean = s["running_mean"].copy()
        self._n_obs = s["n_obs"]

    def frozen_elements(self): return ["W_res", "W_in"]


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_phase(substrate, env_seed, n_steps):
    env = make_game()
    obs = env.reset(seed=env_seed)
    step = 0; completions = 0; current_level = 0
    pred_errors = []; prev_enc = None; prev_h = None

    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed)
            substrate.on_level_transition()
            continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        prev_enc_before = substrate._prev_enc
        prev_h_before = substrate._prev_h

        action = substrate.process(obs_arr) % N_ACTIONS
        obs_next, _, done, info = env.step(action)
        step += 1

        # Pred accuracy
        if prev_h_before is not None and obs_next is not None:
            next_enc = substrate._encode_for_pred(np.asarray(obs_next, dtype=np.float32))
            pred = substrate.predict_from_h(prev_h_before)
            err = float(np.sum((pred - next_enc)**2))
            norm = float(np.sum(next_enc**2)) + 1e-8
            pred_errors.append((err, norm))

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs_next = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
        obs = obs_next

    pred_acc = None
    if pred_errors:
        te = sum(e for e, n in pred_errors)
        tn = sum(n for e, n in pred_errors)
        pred_acc = float(1.0 - te / tn) * 100.0
    return completions, pred_acc


print("=" * 70)
print("STEP 891 — ECHO STATE NETWORK (fixed reservoir + trainable W_out)")
print("=" * 70)
print(f"W_res: {RESERVOIR}x{RESERVOIR} sparse (sparsity={SPARSITY}), spectral_radius={SPECTRAL_RADIUS}. Fixed.")
print(f"W_out: {ENC_DIM}x{RESERVOIR}. Trainable, delta rule, eta={ETA}.")
print(f"Action: prediction-contrast argmax ||pred - enc||. W_out-only transfer.")

t0 = time.time()

# Pretrain
sub_p = ESN891(n_actions=N_ACTIONS, seed=0)
sub_p.reset(0)
for ps in PRETRAIN_SEEDS:
    sub_p.on_level_transition()
    env = make_game(); obs = env.reset(seed=ps * 1000); s = 0
    while s < PRETRAIN_STEPS:
        if obs is None:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition(); continue
        action = sub_p.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs, _, done, _ = env.step(action); s += 1
        if done:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition()
saved = sub_p.get_state()
print(f"Pretrain done ({time.time()-t0:.1f}s).")

cold_comps = []; cold_accs = []
warm_comps = []; warm_accs = []

for ts in TEST_SEEDS:
    # Cold: fresh W_out + fresh reservoir
    sub_c = ESN891(n_actions=N_ACTIONS, seed=0)
    sub_c.reset(0)
    c_comp, c_acc = run_phase(sub_c, ts * 1000, TEST_STEPS)
    cold_comps.append(c_comp); cold_accs.append(c_acc)

    # Warm: pretrained W_out + same W_res/W_in + fresh running_mean/h
    sub_w = ESN891(n_actions=N_ACTIONS, seed=0)
    sub_w.reset(0)
    sub_w.W_out = saved["W_out"].copy()
    sub_w.W_res = saved["W_res"].copy()
    sub_w.W_in = saved["W_in"].copy()
    # running_mean stays fresh (Step 824 finding)
    w_comp, w_acc = run_phase(sub_w, ts * 1000, TEST_STEPS)
    warm_comps.append(w_comp); warm_accs.append(w_acc)

mc = np.mean(cold_comps); mw = np.mean(warm_comps)
vc = [a for a in cold_accs if a is not None]
vw = [a for a in warm_accs if a is not None]
mc_acc = np.mean(vc) if vc else None
mw_acc = np.mean(vw) if vw else None

print()
print(f"RESULTS (ESN forward model):")
print(f"  cold: L1={mc:.0f}/seed  pred_acc={mc_acc:.2f}%" if mc_acc else f"  cold: L1={mc:.0f}/seed  pred_acc=N/A")
print(f"  warm: L1={mw:.0f}/seed  pred_acc={mw_acc:.2f}%" if mw_acc else f"  warm: L1={mw:.0f}/seed  pred_acc=N/A")
if mc_acc and mw_acc:
    print(f"  R3_cf: {'PASS' if mw_acc > mc_acc else 'FAIL'} ({mw_acc - mc_acc:+.2f}%)")
    print(f"\n  Baselines: linear W cold=10.37% warm=15.78% (+5.41%)")
    print(f"  ESN improvement: {mc_acc - 10.37:+.2f}% cold, {mw_acc - 15.78:+.2f}% warm")

print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 891 DONE")
