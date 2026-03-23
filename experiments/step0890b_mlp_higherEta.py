"""
step0890b_mlp_higherEta.py -- MLP Forward Model at eta=0.01 (10x higher).

Leo's mail 2615: step890 (eta=0.001) failed because non-convex loss with online SGD
doesn't converge in 5K steps. Test: eta=0.01. If MLP still underperforms linear W
(19.9% pred_acc) → MLP family KILLED at this scale. If better → eta was bottleneck.

Architecture: identical to step890 except ETA=0.01.
- W1: (128, 260). W2: (256, 128). 2-layer MLP.
- Delta rule backprop through both layers.
- Action: prediction-contrast (argmax ||pred_next - enc||).
- 20% epsilon.

Protocol: pretrain seeds 1-5 (5K), test cold vs warm (W1+W2 transfer) seeds 6-10 (10K).
Metric: pred accuracy (MSE-based), L1. Compare to linear W: pred_acc=19.9%, L1≈300/seed.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

PRETRAIN_SEEDS = list(range(1, 6))
TEST_SEEDS = list(range(6, 11))
PRETRAIN_STEPS = 5_000
TEST_STEPS = 10_000
N_ACTIONS = 4
ETA = 0.01            # 10x higher than step890
HIDDEN = 128
ENC_DIM = 256
EPSILON = 0.20


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


class MLP890b(BaseSubstrate):
    """2-layer MLP forward model, eta=0.01. Prediction-contrast action."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        rng = np.random.RandomState(seed)
        self._rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + n_actions
        self.W1 = rng.randn(HIDDEN, inp_dim).astype(np.float32) * np.sqrt(2.0 / inp_dim)
        self.W2 = rng.randn(ENC_DIM, HIDDEN).astype(np.float32) * np.sqrt(2.0 / HIDDEN)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def _forward(self, enc, action):
        inp = np.concatenate([enc, one_hot(action, self._n_actions)])
        h = np.maximum(0.0, self.W1 @ inp)  # ReLU
        return self.W2 @ h, h, inp

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc

        if self._prev_enc is not None and self._prev_action is not None:
            pred, h, inp = self._forward(self._prev_enc, self._prev_action)
            error = enc - pred
            # Gradient clip
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                # Backprop through W2
                grad_h = self.W2.T @ error  # (128,)
                grad_h_relu = grad_h * (h > 0).astype(np.float32)  # ReLU gate
                # Clip W2 gradient
                self.W2 -= ETA * np.outer(error, h)
                # Clip W1 gradient
                dW1 = np.outer(grad_h_relu, inp)
                dW1_norm = float(np.linalg.norm(dW1))
                if dW1_norm > 10.0:
                    dW1 = dW1 * (10.0 / dW1_norm)
                self.W1 -= ETA * dW1
                self._pred_errors.append(float(np.mean(error ** 2)))

        # Action: prediction-contrast (max predicted change from current enc)
        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            best_a = 0; best_score = -1.0
            for a in range(self._n_actions):
                pred, _, _ = self._forward(enc, a)
                score = float(np.sum((pred - enc) ** 2))
                if score > best_score:
                    best_score = score; best_a = a
            action = best_a

        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    def pred_accuracy(self):
        if len(self._pred_errors) < 10:
            return None
        return -float(np.mean(list(self._pred_errors)[-200:]))  # raw neg-MSE

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        rng = np.random.RandomState(seed)
        self._rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + self._n_actions
        self.W1 = rng.randn(HIDDEN, inp_dim).astype(np.float32) * np.sqrt(2.0 / inp_dim)
        self.W2 = rng.randn(ENC_DIM, HIDDEN).astype(np.float32) * np.sqrt(2.0 / HIDDEN)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def get_state(self):
        return {"W1": self.W1.copy(), "W2": self.W2.copy(),
                "running_mean": self._running_mean.copy(), "n_obs": self._n_obs}

    def set_state(self, s):
        self.W1 = s["W1"].copy(); self.W2 = s["W2"].copy()
        self._running_mean = s["running_mean"].copy(); self._n_obs = s["n_obs"]

    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_phase(substrate, env_fn, n_actions, env_seed, n_steps):
    env = env_fn(); obs = env.reset(seed=env_seed)
    step = 0; completions = 0; current_level = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); substrate.on_level_transition(); continue
        action = substrate.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
    return completions


print("=" * 70)
print("STEP 890b — MLP FORWARD MODEL eta=0.01 (10x higher than 890)")
print("=" * 70)
print(f"Testing if eta=0.001 was bottleneck. MLP W1(128,260)+W2(256,128).")
print(f"Kill: if pred_acc <= linear W (19.9%) → MLP family KILLED at this scale.")

t0 = time.time()

# Pretrain
sub_p = MLP890b(n_actions=N_ACTIONS, seed=0)
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
pa = sub_p.pred_accuracy()
print(f"  Pretrain done ({time.time()-t0:.1f}s). pred_acc(raw neg-MSE)={pa:.4f}" if pa else
      f"  Pretrain done ({time.time()-t0:.1f}s). pred_acc=N/A")

cold_comps = []; warm_comps = []
cold_pa = []; warm_pa = []

for ts in TEST_SEEDS:
    sub_c = MLP890b(n_actions=N_ACTIONS, seed=ts % 4)
    sub_c.reset(ts % 4)
    c_comp = run_phase(sub_c, make_game, N_ACTIONS, ts * 1000, TEST_STEPS)
    cold_comps.append(c_comp)
    cpa = sub_c.pred_accuracy()
    cold_pa.append(cpa if cpa else 0.0)

    sub_w = MLP890b(n_actions=N_ACTIONS, seed=ts % 4)
    sub_w.reset(ts % 4)
    sub_w.W1 = saved["W1"].copy(); sub_w.W2 = saved["W2"].copy()
    w_comp = run_phase(sub_w, make_game, N_ACTIONS, ts * 1000, TEST_STEPS)
    warm_comps.append(w_comp)
    wpa = sub_w.pred_accuracy()
    warm_pa.append(wpa if wpa else 0.0)
    print(f"  seed={ts}: cold L1={c_comp:4d}  warm L1={w_comp:4d}")

mc = np.mean(cold_comps); mw = np.mean(warm_comps)
mcp = np.mean(cold_pa); mwp = np.mean(warm_pa)
print(f"\ncold: L1={mc:.1f}/seed  pred_acc(neg-MSE)={mcp:.4f}  {cold_comps}")
print(f"warm: L1={mw:.1f}/seed  pred_acc(neg-MSE)={mwp:.4f}  {warm_comps}")
print(f"L1 delta: {mw-mc:+.1f}/seed")
print(f"\nLinear W pred_acc=19.9% (step835). If neg-MSE >> linear → MLP better.")
print(f"Kill check: {'ALIVE — MLP shows improvement over eta=0.001' if mcp > -0.1 else 'KILLED — MLP family dead at this scale'}")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 890b DONE")
