"""
step0834_capacity_sweep.py -- Forward model capacity sweep: d=16,32,64,128,256.

R3 hypothesis: smaller W (lower-dim encoding) may transfer better (less overfitting).
Compares prediction accuracy R3_cf across encoding dimensions.

Protocol: use PredictionContrast780 architecture but with different avgpool sizes.
d=256: avgpool16 (current 674). d=128: avgpool22. d=64: avgpool32. d=32: avgpool45. d=16: avgpool64.
Approximate: use np.interp to resize to target dimension.

Simpler approach: project the 256-dim 674 encoding to lower d via random projection.
This tests if lower-dimensional representations transfer better.
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
ETA = 0.01
BASE_DIM = 256


class ProjectedForwardModel834(BaseSubstrate):
    """PredictionContrast with random projection to d dimensions."""

    def __init__(self, n_actions=4, seed=0, d=256):
        self._n_actions = n_actions
        self._seed = seed
        self._d = d
        rng = np.random.RandomState(seed)
        # Random projection matrix: 256 → d
        self._P = rng.randn(d, BASE_DIM).astype(np.float32) / np.sqrt(BASE_DIM)
        self.W = rng.randn(d, d + n_actions).astype(np.float32) * 0.01
        self.running_mean = np.zeros(d, np.float32)
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None

    def _encode(self, obs):
        x_raw = _enc_frame(np.asarray(obs, dtype=np.float32))  # 256-dim
        x = self._P @ x_raw  # project to d-dim
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def _encode_for_pred(self, obs):
        x_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        x = self._P @ x_raw
        return x - self.running_mean

    def predict_next(self, enc, action):
        a_oh = np.zeros(self._n_actions, np.float32); a_oh[action] = 1.0
        return self.W @ np.concatenate([enc, a_oh])

    def process(self, observation):
        x = self._encode(observation)
        self._last_enc = x
        if self._prev_enc is not None and self._prev_action is not None:
            a_oh = np.zeros(self._n_actions, np.float32); a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            pred_err = self.W @ inp - x
            self.W -= ETA * np.outer(pred_err, inp)
        # argmax predicted change
        best_a, best_score = 0, -1.0
        for a in range(self._n_actions):
            pred = self.predict_next(x, a)
            score = float(np.sum((pred - x) ** 2))
            if score > best_score:
                best_score = score; best_a = a
        self._prev_enc = x.copy(); self._prev_action = best_a
        return best_a

    @property
    def n_actions(self): return self._n_actions

    def get_state(self):
        return {"W": self.W.copy(), "running_mean": self.running_mean.copy(),
                "_n_obs": self._n_obs,
                "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
                "_prev_action": self._prev_action}

    def set_state(self, state):
        self.W = state["W"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self._prev_enc = state["_prev_enc"].copy() if state["_prev_enc"] is not None else None
        self._prev_action = state["_prev_action"]

    def reset(self, seed):
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def frozen_elements(self): return []


def run_phase(substrate, env_seed, n_steps):
    try:
        import arcagi3
        env = arcagi3.make("LS20")
    except:
        import util_arcagi3
        env = util_arcagi3.make("LS20")
    obs = env.reset(seed=env_seed)
    completions = 0; current_level = 0; step = 0
    pred_errors = []
    prev_enc = None
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); substrate.on_level_transition(); continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr)
        obs_next, _, done, info = env.step(action % N_ACTIONS)
        step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs_next = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
        if prev_enc is not None and obs_next is not None and hasattr(substrate, '_last_enc'):
            next_enc = substrate._encode_for_pred(np.asarray(obs_next, dtype=np.float32))
            pred = substrate.predict_next(prev_enc, action % N_ACTIONS)
            err = float(np.sum((pred - next_enc) ** 2))
            norm = float(np.sum(next_enc ** 2)) + 1e-8
            pred_errors.append((err, norm))
        if hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
            prev_enc = substrate._last_enc.copy()
        obs = obs_next
    pred_acc = None
    if pred_errors:
        te = sum(e for e, n in pred_errors)
        tn = sum(n for e, n in pred_errors)
        pred_acc = float(1.0 - te/tn) * 100.0
    return completions, pred_acc


print("=" * 70)
print("STEP 834 — FORWARD MODEL CAPACITY SWEEP (d=16,32,64,128,256)")
print("=" * 70)
print("Metric: R3_cf pred accuracy. Does lower d transfer better?")
print()

t0 = time.time()
DIMS = [16, 32, 64, 128, 256]

for d in DIMS:
    print(f"\n--- d={d} ---")
    # Pretrain
    sub_p = ProjectedForwardModel834(n_actions=N_ACTIONS, seed=0, d=d)
    sub_p.reset(0)
    for ps in PRETRAIN_SEEDS:
        sub_p.on_level_transition()
        try:
            import arcagi3; env = arcagi3.make("LS20")
        except:
            import util_arcagi3; env = util_arcagi3.make("LS20")
        obs = env.reset(seed=ps * 1000); s = 0
        while s < PRETRAIN_STEPS:
            if obs is None:
                obs = env.reset(seed=ps * 1000); sub_p.on_level_transition(); continue
            action = sub_p.process(np.asarray(obs, dtype=np.float32))
            obs, _, done, _ = env.step(action % N_ACTIONS); s += 1
            if done:
                obs = env.reset(seed=ps * 1000); sub_p.on_level_transition()
    saved = sub_p.get_state()

    cold_accs = []; warm_accs = []
    for ts in TEST_SEEDS:
        sub_c = ProjectedForwardModel834(n_actions=N_ACTIONS, seed=0, d=d)
        sub_c.reset(0)
        _, acc_c = run_phase(sub_c, ts * 1000, TEST_STEPS)
        cold_accs.append(acc_c)

        sub_w = ProjectedForwardModel834(n_actions=N_ACTIONS, seed=0, d=d)
        sub_w.reset(0); sub_w.set_state(saved)
        _, acc_w = run_phase(sub_w, ts * 1000, TEST_STEPS)
        warm_accs.append(acc_w)

    vc = [a for a in cold_accs if a is not None]
    vw = [a for a in warm_accs if a is not None]
    mc = np.mean(vc) if vc else None
    mw = np.mean(vw) if vw else None
    r3_pass = mc is not None and mw is not None and mw > mc
    print(f"  d={d}: cold={mc:.2f}%  warm={mw:.2f}%  {'PASS' if r3_pass else 'FAIL'} (+{mw-mc:.2f}%)" if mc and mw else f"  d={d}: N/A")

print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 834 DONE")
