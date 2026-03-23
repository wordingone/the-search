"""
step0892_mlp_predictive_novelty.py -- MLP forward model + visited_set novelty detection.

R3 hypothesis: MLP (non-linear) W achieves higher pred accuracy than linear W (19.9%),
enabling visited_set novelty detection to select more correct novel-state actions.
This is step889 but with MLP instead of linear W.

Architecture:
- W1: (128, 256+n_actions). W2: (256, 128). 2-layer MLP forward model (step890 architecture).
- visited_set: set of enc hashes. Per-observation. Resets on episode end.
- running_mean: centering.

Action selection:
- For each action a: h = ReLU(W1 @ concat(enc, onehot(a))); pred = W2 @ h
- h_pred = enc_hash(pred)
- novelty(a) = 1 if h_pred not in visited_set else 0
- Action = uniform random from novelty=1 set. If all =0: random.

on_level_transition(): RESET visited_set (new episode). KEEP W1, W2 (dynamics transfer).

R3_cf protocol:
- Pretrain: random actions, 5K steps/seed, seeds 1-5. W1/W2 learn dynamics.
- Cold test: fresh W1/W2 + empty visited_set, seeds 6-10, 10K steps.
- Warm test: W1/W2-only transfer (fresh running_mean), empty visited_set, seeds 6-10.
- Metric: pred accuracy (hash match rate), L1 completions.
- Baseline: linear W pred_acc=10.37% cold (Step 835). MLP step890 to be compared.

Leo mail 2605-2609: run 890 first to confirm MLP improvement, then 892.
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
ETA = 0.001
ENC_DIM = 256
HIDDEN = 128
EPSILON = 0.05


def enc_hash(enc):
    """Coarse hash of encoding vector (quantize to 4 bits)."""
    coarse = (enc[:32] * 4).astype(np.int8)
    return hash(coarse.tobytes())


class MLPPredictiveNovelty892(BaseSubstrate):
    """MLP forward model + visited_set novelty detection."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + n_actions
        rng2 = np.random.RandomState(seed)
        self.W1 = rng2.randn(HIDDEN, inp_dim).astype(np.float32) * np.sqrt(2.0 / inp_dim)
        self.W2 = rng2.randn(ENC_DIM, HIDDEN).astype(np.float32) * np.sqrt(2.0 / HIDDEN)
        self.visited_set = set()
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
        a_oh = np.zeros(self._n_actions, np.float32)
        a_oh[action] = 1.0
        inp = np.concatenate([enc, a_oh])
        hidden = np.maximum(0, self.W1 @ inp)
        pred = self.W2 @ hidden
        return pred, hidden, inp

    def _update(self, enc, prev_enc, prev_action):
        pred, hidden, inp = self._forward(prev_enc, prev_action)
        err = pred - enc
        grad_W2 = np.outer(err, hidden)
        grad_h = (self.W2.T @ err) * (hidden > 0)
        grad_W1 = np.outer(grad_h, inp)
        self.W2 -= ETA * grad_W2
        self.W1 -= ETA * grad_W1
        return float(np.sum(err**2)), float(np.sum(enc**2)) + 1e-8

    def predict_next(self, enc, action):
        pred, _, _ = self._forward(enc, action)
        return pred

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        h = enc_hash(enc)
        self.visited_set.add(h)

        if self._prev_enc is not None and self._prev_action is not None:
            self._update(enc, self._prev_enc, self._prev_action)

        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            novel_actions = []
            for a in range(self._n_actions):
                pred, _, _ = self._forward(enc, a)
                h_pred = enc_hash(pred)
                if h_pred not in self.visited_set:
                    novel_actions.append(a)
            action = int(self._rng.choice(novel_actions)) if novel_actions else int(self._rng.randint(0, self._n_actions))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def process_random(self, observation):
        """Update W with random action (pretrain phase)."""
        enc = self._encode(observation)
        self._last_enc = enc
        h = enc_hash(enc)
        self.visited_set.add(h)

        if self._prev_enc is not None and self._prev_action is not None:
            self._update(enc, self._prev_enc, self._prev_action)

        action = int(self._rng.randint(0, self._n_actions))
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed)
        inp_dim = ENC_DIM + self._n_actions
        self.W1 = rng2.randn(HIDDEN, inp_dim).astype(np.float32) * np.sqrt(2.0 / inp_dim)
        self.W2 = rng2.randn(ENC_DIM, HIDDEN).astype(np.float32) * np.sqrt(2.0 / HIDDEN)
        self.visited_set = set()
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        """Reset visited_set on episode end. Keep W1, W2 (dynamics transfer)."""
        self.visited_set = set()
        self._prev_enc = None; self._prev_action = None

    def get_state(self):
        return {"W1": self.W1.copy(), "W2": self.W2.copy(),
                "running_mean": self._running_mean.copy(), "n_obs": self._n_obs}

    def set_state(self, s):
        self.W1 = s["W1"].copy(); self.W2 = s["W2"].copy()
        self._running_mean = s["running_mean"].copy()
        self._n_obs = s["n_obs"]

    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def pretrain_phase(substrate, seeds, n_steps_per_seed):
    """Pretrain MLP with random actions across multiple seeds."""
    for ps in seeds:
        env = make_game()
        obs = env.reset(seed=ps * 1000)
        step = 0
        substrate.on_level_transition()
        while step < n_steps_per_seed:
            if obs is None:
                obs = env.reset(seed=ps * 1000)
                substrate.on_level_transition()
                continue
            action = substrate.process_random(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
            obs, _, done, info = env.step(action)
            step += 1
            if done:
                obs = env.reset(seed=ps * 1000)
                substrate.on_level_transition()


def test_phase(substrate, env_seed, n_steps):
    """Test with novelty-based action selection."""
    env = make_game()
    obs = env.reset(seed=env_seed)
    step = 0; completions = 0; current_level = 0
    correct_preds = 0; total_preds = 0

    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed)
            substrate.on_level_transition()
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        prev_enc = substrate._prev_enc
        prev_action = substrate._prev_action

        action = substrate.process(obs_arr) % N_ACTIONS
        obs_next, _, done, info = env.step(action)
        step += 1

        # Pred accuracy: hash of predicted next vs actual next
        if prev_enc is not None and prev_action is not None and obs_next is not None:
            pred = substrate.predict_next(prev_enc, prev_action)
            h_pred = enc_hash(pred)
            actual_enc = substrate._encode_for_pred(np.asarray(obs_next, dtype=np.float32))
            h_actual = enc_hash(actual_enc)
            if h_pred == h_actual:
                correct_preds += 1
            total_preds += 1

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs_next = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
        obs = obs_next

    pred_acc = correct_preds / max(total_preds, 1) * 100.0
    return completions, pred_acc


print("=" * 70)
print("STEP 892 — MLP PREDICTIVE NOVELTY (W1/W2 + visited_set)")
print("=" * 70)
print(f"W1({HIDDEN},{ENC_DIM}+n_act) → ReLU → W2({ENC_DIM},{HIDDEN}). visited_set per-obs.")
print(f"Action: uniform random from novelty=1 MLP-predicted states. Fallback: random.")
print(f"Pretrain: random actions {PRETRAIN_STEPS} steps/seed, seeds 1-5.")
print(f"Test: cold (fresh W) vs warm (W1/W2-only transfer), {TEST_STEPS} steps, seeds 6-10.")

t0 = time.time()

# Pretrain with random actions
sub_pretrain = MLPPredictiveNovelty892(n_actions=N_ACTIONS, seed=0)
sub_pretrain.reset(0)
pretrain_phase(sub_pretrain, PRETRAIN_SEEDS, PRETRAIN_STEPS)
saved_W1 = sub_pretrain.W1.copy()
saved_W2 = sub_pretrain.W2.copy()
print(f"Pretrain done ({time.time()-t0:.1f}s). visited_set size: {len(sub_pretrain.visited_set)}")

cold_comps = []; cold_accs = []
warm_comps = []; warm_accs = []

for ts in TEST_SEEDS:
    # Cold: fresh W1/W2 + empty visited_set
    sub_c = MLPPredictiveNovelty892(n_actions=N_ACTIONS, seed=0)
    sub_c.reset(0)
    c_comp, c_acc = test_phase(sub_c, ts * 1000, TEST_STEPS)
    cold_comps.append(c_comp); cold_accs.append(c_acc)

    # Warm: trained W1/W2 + empty visited_set (fresh running_mean)
    sub_w = MLPPredictiveNovelty892(n_actions=N_ACTIONS, seed=0)
    sub_w.reset(0)
    sub_w.W1 = saved_W1.copy()
    sub_w.W2 = saved_W2.copy()
    w_comp, w_acc = test_phase(sub_w, ts * 1000, TEST_STEPS)
    warm_comps.append(w_comp); warm_accs.append(w_acc)

mc = np.mean(cold_comps); mw = np.mean(warm_comps)
mc_acc = np.mean(cold_accs); mw_acc = np.mean(warm_accs)

print()
print(f"RESULTS (MLP predictive novelty):")
print(f"  cold: L1={mc:.1f}/seed  pred_acc={mc_acc:.2f}%  ({cold_comps})")
print(f"  warm: L1={mw:.1f}/seed  pred_acc={mw_acc:.2f}%  ({warm_comps})")
print(f"  R3_cf: {'PASS' if mw_acc > mc_acc else 'FAIL'} (pred_acc: {mc_acc:.2f}% → {mw_acc:.2f}%, delta={mw_acc-mc_acc:+.2f}%)")
print(f"  L1 delta: {mw-mc:+.1f}/seed")
print(f"\n  Baselines: linear W cold=10.37% warm=15.78% (+5.41%)")
print(f"  MLP improvement: {mc_acc - 10.37:+.2f}% cold, {mw_acc - 15.78:+.2f}% warm")

print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 892 DONE")
