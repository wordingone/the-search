"""
step0832_encoding_sweep.py -- Encoding function sweep for forward model transfer.

R3 hypothesis: does encoding choice affect D(s) transfer quality?
Tests 4 encoding variants:
  1. _enc_frame (step674, avgpool16, 256-dim) — baseline
  2. Raw flatten (first 256 pixels) — no aggregation
  3. Row-mean (mean per row of observation) — low-dim global
  4. Random projection from raw (same dim, different basis)

Metric: R3_cf pred accuracy warm vs cold. Higher warm-cold gap → encoding captures more transferable dynamics.
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
DIM = 256
ETA = 0.01


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


class ForwardModelBase832(BaseSubstrate):
    """Forward model substrate with configurable encoding function."""

    def __init__(self, n_actions=4, seed=0, enc_fn=None, dim=256):
        self._n_actions = n_actions
        self._seed = seed
        self._enc_fn = enc_fn
        self._dim = dim
        rng = np.random.RandomState(seed)
        self.W = rng.randn(dim, dim + n_actions).astype(np.float32) * 0.01
        self.running_mean = np.zeros(dim, np.float32)
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None

    def _encode_raw(self, obs):
        return self._enc_fn(np.asarray(obs, dtype=np.float32))

    def _encode(self, obs):
        x = self._encode_raw(obs)
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def _encode_for_pred(self, obs):
        return self._encode_raw(obs) - self.running_mean

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


# Define encoding functions
_RAND_PROJ = np.random.RandomState(99).randn(DIM, 256).astype(np.float32) / np.sqrt(256)

def enc_avgpool(obs):
    """Step674 avgpool16 encoding → 256-dim."""
    return _enc_frame(obs)

def enc_raw_flatten(obs):
    """First 256 pixels flattened (clipped to obs size)."""
    flat = obs.flatten().astype(np.float32)
    if len(flat) >= DIM:
        return flat[:DIM]
    out = np.zeros(DIM, np.float32)
    out[:len(flat)] = flat
    return out

def enc_row_mean(obs):
    """Mean of each row → pad/clip to 256 dims."""
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        return enc_avgpool(obs)  # fallback
    # Mean over last axis
    means = arr.mean(axis=-1).flatten()
    if len(means) >= DIM:
        return means[:DIM]
    out = np.zeros(DIM, np.float32)
    out[:len(means)] = means
    return out

def enc_rand_proj(obs):
    """Random projection from first 256 pixels."""
    flat = obs.flatten().astype(np.float32)
    if len(flat) >= 256:
        x = flat[:256]
    else:
        x = np.zeros(256, np.float32)
        x[:len(flat)] = flat
    return _RAND_PROJ @ x


ENCODINGS = [
    ("avgpool674", enc_avgpool),
    ("raw_flatten", enc_raw_flatten),
    ("row_mean", enc_row_mean),
    ("rand_proj", enc_rand_proj),
]


def run_phase(substrate, env_seed, n_steps):
    env = make_game()
    obs = env.reset(seed=env_seed)
    step = 0; pred_errors = []; prev_enc = None
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); substrate.on_level_transition(); continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr)
        obs_next, _, done, _ = env.step(action % N_ACTIONS); step += 1
        if done:
            obs_next = env.reset(seed=env_seed); substrate.on_level_transition()
        if prev_enc is not None and obs_next is not None and substrate._last_enc is not None:
            next_enc = substrate._encode_for_pred(np.asarray(obs_next, dtype=np.float32))
            pred = substrate.predict_next(prev_enc, action % N_ACTIONS)
            err = float(np.sum((pred - next_enc) ** 2))
            norm = float(np.sum(next_enc ** 2)) + 1e-8
            pred_errors.append((err, norm))
        if substrate._last_enc is not None:
            prev_enc = substrate._last_enc.copy()
        obs = obs_next
    if not pred_errors: return None
    return float(1.0 - sum(e for e,n in pred_errors) / sum(n for e,n in pred_errors)) * 100.0


print("=" * 70)
print("STEP 832 — ENCODING FUNCTION SWEEP (4 variants)")
print("=" * 70)

t0 = time.time()

for enc_name, enc_fn in ENCODINGS:
    print(f"\n--- encoding={enc_name} ---")

    sub_p = ForwardModelBase832(n_actions=N_ACTIONS, seed=0, enc_fn=enc_fn, dim=DIM)
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

    cold_accs = []; warm_accs = []
    for ts in TEST_SEEDS:
        sub_c = ForwardModelBase832(n_actions=N_ACTIONS, seed=0, enc_fn=enc_fn, dim=DIM)
        sub_c.reset(0)
        acc_c = run_phase(sub_c, ts * 1000, TEST_STEPS)
        cold_accs.append(acc_c)

        sub_w = ForwardModelBase832(n_actions=N_ACTIONS, seed=0, enc_fn=enc_fn, dim=DIM)
        sub_w.reset(0); sub_w.set_state(saved)
        acc_w = run_phase(sub_w, ts * 1000, TEST_STEPS)
        warm_accs.append(acc_w)

    vc = [a for a in cold_accs if a is not None]
    vw = [a for a in warm_accs if a is not None]
    mc = np.mean(vc) if vc else None
    mw = np.mean(vw) if vw else None
    r3_pass = mc is not None and mw is not None and mw > mc
    if mc is not None and mw is not None:
        print(f"  {enc_name}: cold={mc:.2f}%  warm={mw:.2f}%  {'PASS' if r3_pass else 'FAIL'}  (+{mw-mc:.2f}%)")
    else:
        print(f"  {enc_name}: N/A")

print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 832 DONE")
