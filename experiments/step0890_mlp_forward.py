"""
step0890_mlp_forward.py -- 2-Layer MLP forward model for prediction contrast.

R3 hypothesis: does a non-linear (MLP) forward model achieve higher prediction
accuracy than linear W? Linear W gives 19.9% on LS20. If MLP >> 19.9% →
linear W was the bottleneck, not the architecture.

Architecture: enc(obs) → W1(128, ReLU) → W2 → predicted_next(256).
Update: delta rule backprop through both layers (MSE loss).
W1: (128, 256+n_actions). W2: (256, 128). eta=0.001.
Action: prediction-contrast (argmax ||pred - obs|| over all actions).

Protocol: pretrain seeds 1-5 (5K each), test cold vs warm seeds 6-10 (25K).
W-only transfer (reset running_mean on warm test — Step 824 finding).
Metric: pred accuracy AND L1 completions.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

PRETRAIN_SEEDS = [1, 2, 3, 4, 5]
TEST_SEEDS = [6, 7, 8, 9, 10]
PRETRAIN_STEPS = 5_000
TEST_STEPS = 25_000
N_ACTIONS = 4
ETA = 0.001
HIDDEN = 128
ENC_DIM = 256


class MLP890(BaseSubstrate):
    """2-layer MLP forward model with prediction-contrast action selection."""

    def __init__(self, n_actions=4, seed=0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + n_actions
        self.W1 = rng.randn(HIDDEN, inp_dim).astype(np.float32) * np.sqrt(2.0 / inp_dim)
        self.W2 = rng.randn(ENC_DIM, HIDDEN).astype(np.float32) * np.sqrt(2.0 / HIDDEN)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self._running_mean = (1 - alpha) * self._running_mean + alpha * enc_raw
        return enc_raw - self._running_mean

    def _encode_for_pred(self, obs):
        return _enc_frame(np.asarray(obs, dtype=np.float32)) - self._running_mean

    def _forward(self, enc, action):
        a_oh = np.zeros(self._n_actions, np.float32); a_oh[action] = 1.0
        inp = np.concatenate([enc, a_oh])
        hidden = np.maximum(0, self.W1 @ inp)  # ReLU
        pred = self.W2 @ hidden
        return pred, hidden, inp

    def _update(self, enc, prev_enc, prev_action):
        pred, hidden, inp = self._forward(prev_enc, prev_action)
        err = pred - enc
        grad_W2 = np.outer(err, hidden)
        grad_h = (self.W2.T @ err) * (hidden > 0)  # ReLU derivative
        grad_W1 = np.outer(grad_h, inp)
        self.W2 -= ETA * grad_W2
        self.W1 -= ETA * grad_W1
        return float(np.sum(err ** 2)), float(np.sum(enc ** 2)) + 1e-8

    def predict_next(self, enc, action):
        pred, _, _ = self._forward(enc, action)
        return pred

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        if self._prev_enc is not None and self._prev_action is not None:
            self._update(enc, self._prev_enc, self._prev_action)
        # Prediction contrast: argmax predicted change
        best_a, best_score = 0, -1.0
        for a in range(self._n_actions):
            pred, _, _ = self._forward(enc, a)
            score = float(np.sum((pred - enc) ** 2))
            if score > best_score:
                best_score = score; best_a = a
        self._prev_enc = enc.copy(); self._prev_action = best_a
        return best_a

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + self._n_actions
        self.W1 = rng.randn(HIDDEN, inp_dim).astype(np.float32) * np.sqrt(2.0 / inp_dim)
        self.W2 = rng.randn(ENC_DIM, HIDDEN).astype(np.float32) * np.sqrt(2.0 / HIDDEN)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def get_state(self):
        return {"W1": self.W1.copy(), "W2": self.W2.copy(),
                "running_mean": self._running_mean.copy(), "_n_obs": self._n_obs}

    def set_state(self, s):
        self.W1 = s["W1"].copy(); self.W2 = s["W2"].copy()
        self._running_mean = s["running_mean"].copy()
        self._n_obs = s["_n_obs"]

    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_phase(substrate, env_seed, n_steps):
    env = make_game()
    obs = env.reset(seed=env_seed)
    step = 0; pred_errors = []; prev_enc = None
    completions = 0; current_level = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); substrate.on_level_transition(); continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr)
        obs_next, _, done, info = env.step(action % N_ACTIONS); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs_next = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
        if prev_enc is not None and obs_next is not None and substrate._last_enc is not None:
            next_enc = substrate._encode_for_pred(np.asarray(obs_next, dtype=np.float32))
            pred = substrate.predict_next(prev_enc, action % N_ACTIONS)
            err = float(np.sum((pred - next_enc) ** 2))
            norm = float(np.sum(next_enc ** 2)) + 1e-8
            pred_errors.append((err, norm))
        if substrate._last_enc is not None:
            prev_enc = substrate._last_enc.copy()
        obs = obs_next
    pred_acc = None
    if pred_errors:
        te = sum(e for e, n in pred_errors)
        tn = sum(n for e, n in pred_errors)
        pred_acc = float(1.0 - te / tn) * 100.0
    return completions, pred_acc


print("=" * 70)
print("STEP 890 — 2-LAYER MLP FORWARD MODEL")
print("=" * 70)
print(f"Architecture: W1({HIDDEN},{ENC_DIM}+n_act) → ReLU → W2({ENC_DIM},{HIDDEN}). eta={ETA}")
print(f"Action: prediction-contrast (argmax ||pred - enc||)")
print(f"W-only transfer (reset running_mean on warm test).")

t0 = time.time()

# Pretrain
sub_p = MLP890(n_actions=N_ACTIONS, seed=0)
sub_p.reset(0)
for ps in PRETRAIN_SEEDS:
    sub_p.on_level_transition()
    env = make_game(); obs = env.reset(seed=ps * 1000); s = 0
    while s < PRETRAIN_STEPS:
        if obs is None:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition(); continue
        action = sub_p.process(np.asarray(obs, dtype=np.float32))
        obs, _, done, _ = env.step(action % N_ACTIONS); s += 1
        if done:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition()
saved = sub_p.get_state()
print(f"Pretrain done ({time.time()-t0:.1f}s).")

cold_comps = []; cold_accs = []
warm_comps = []; warm_accs = []

for ts in TEST_SEEDS:
    # Cold
    sub_c = MLP890(n_actions=N_ACTIONS, seed=0)
    sub_c.reset(0)
    c_comp, c_acc = run_phase(sub_c, ts * 1000, TEST_STEPS)
    cold_comps.append(c_comp); cold_accs.append(c_acc)

    # Warm W-only: transfer W1+W2, reset running_mean
    sub_w = MLP890(n_actions=N_ACTIONS, seed=0)
    sub_w.reset(0)
    sub_w.W1 = saved["W1"].copy()
    sub_w.W2 = saved["W2"].copy()
    # running_mean stays fresh (n_obs=0)
    w_comp, w_acc = run_phase(sub_w, ts * 1000, TEST_STEPS)
    warm_comps.append(w_comp); warm_accs.append(w_acc)

mc_comp = np.mean(cold_comps); mw_comp = np.mean(warm_comps)
vc = [a for a in cold_accs if a is not None]
vw = [a for a in warm_accs if a is not None]
mc_acc = np.mean(vc) if vc else None
mw_acc = np.mean(vw) if vw else None

print()
print(f"RESULTS (MLP forward model):")
print(f"  cold: L1={mc_comp:.0f}/seed  pred_acc={mc_acc:.2f}%" if mc_acc else f"  cold: L1={mc_comp:.0f}/seed  pred_acc=N/A")
print(f"  warm: L1={mw_comp:.0f}/seed  pred_acc={mw_acc:.2f}%" if mw_acc else f"  warm: L1={mw_comp:.0f}/seed  pred_acc=N/A")
if mc_acc and mw_acc:
    print(f"  R3_cf: {'PASS' if mw_acc > mc_acc else 'FAIL'} ({mw_acc-mc_acc:+.2f}%)")
    print(f"\n  Linear W baseline: cold=10.37%  warm=15.78%  (+5.41%)")
    print(f"  MLP improvement: {mc_acc - 10.37:+.2f}% cold, {mw_acc - 15.78:+.2f}% warm")

print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 890 DONE")
