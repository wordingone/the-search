"""
step0889_predictive_novelty.py -- PredictiveNovelty: visited_set + W forward model.

R3 hypothesis: W predicts next obs. Actions with predicted novel next states are selected.
visited_set is per-observation hash (NOT per-(state,action)) — passes graph ban.
W is global linear forward model — passes graph ban.

Architecture:
- W: (256, 256+n_actions). Linear forward model. Delta rule, eta=0.01.
- visited_set: set of enc hashes. Per-obs. Resets on episode end.
- running_mean: centering.

Action selection:
- For each action a: pred = W @ concat(enc, onehot(a))
- h_pred = enc_hash(pred)
- novelty(a) = 1 if h_pred not in visited_set else 0
- Action = uniform random from novelty=1 set. If all =0: random.

on_level_transition(): RESET visited_set (new episode). KEEP W (dynamics transfer).

R3_cf protocol:
- Pretrain: random actions, 5K steps/seed, seeds 1-5. W learns dynamics.
- Cold test: fresh W + empty visited_set, seeds 6-10, 10K steps each.
- Warm test: trained W (W-only transfer, fresh running_mean), seeds 6-10, 10K steps each.
- Metric: pred accuracy (hash match rate), L1 completions.

Games: LS20 (4 actions), FT09 (10 action sample to avoid 68-action scan bottleneck).

Leo mail 2603 spec: "visited_set + W prediction. This bridges Prop 21."
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

PRETRAIN_SEEDS = list(range(1, 6))
TEST_SEEDS = list(range(6, 11))
PRETRAIN_STEPS = 5_000   # per seed; cap for runtime
TEST_STEPS = 10_000      # per seed; cap for runtime
N_ACTIONS_LS20 = 4
N_ACTIONS_FT09 = 68
ETA = 0.01
ENC_DIM = 256
EPSILON = 0.05


def enc_hash(enc):
    """Coarse hash of encoding vector (quantize to 4 bits)."""
    coarse = (enc[:32] * 4).astype(np.int8)
    return hash(coarse.tobytes())


class PredictiveNovelty889(BaseSubstrate):
    """W forward model + visited_set novelty detection. visited_set per-observation."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + n_actions
        self.W = np.zeros((ENC_DIM, inp_dim), dtype=np.float32)
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

    def _predict_next(self, enc, action):
        a_oh = np.zeros(self._n_actions, np.float32)
        a_oh[action] = 1.0
        inp = np.concatenate([enc, a_oh])
        return self.W @ inp

    def _update_W(self, enc, prev_enc, prev_action):
        a_oh = np.zeros(self._n_actions, np.float32)
        a_oh[prev_action] = 1.0
        inp = np.concatenate([prev_enc, a_oh])
        pred = self.W @ inp
        err = pred - enc
        self.W -= ETA * np.outer(err, inp)
        return float(np.sum(err**2)), float(np.sum(enc**2)) + 1e-8

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        h = enc_hash(enc)
        self.visited_set.add(h)

        if self._prev_enc is not None and self._prev_action is not None:
            self._update_W(enc, self._prev_enc, self._prev_action)

        if self._rng.random() < self._epsilon:
            action = self._rng.randint(0, self._n_actions)
        else:
            # Score actions: 1 if predicted next obs is novel, else 0
            novel_actions = []
            for a in range(self._n_actions):
                pred = self._predict_next(enc, a)
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
            self._update_W(enc, self._prev_enc, self._prev_action)

        action = int(self._rng.randint(0, self._n_actions))
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + self._n_actions
        self.W = np.zeros((ENC_DIM, inp_dim), dtype=np.float32)
        self.visited_set = set()
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        """Reset visited_set on episode end. Keep W (dynamics transfer)."""
        self.visited_set = set()
        self._prev_enc = None; self._prev_action = None

    def get_state(self):
        return {"W": self.W.copy(), "running_mean": self._running_mean.copy(),
                "n_obs": self._n_obs}

    def set_state(self, s):
        self.W = s["W"].copy()
        self._running_mean = s["running_mean"].copy()
        self._n_obs = s["n_obs"]

    def frozen_elements(self): return []


def make_ls20():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def make_ft09():
    try:
        import arcagi3; return arcagi3.make("FT09")
    except:
        import util_arcagi3; return util_arcagi3.make("FT09")


def pretrain_phase(substrate, env_fn, n_actions, seeds, n_steps_per_seed):
    """Pretrain W with random actions. visited_set reset between episodes."""
    for ps in seeds:
        env = env_fn()
        obs = env.reset(seed=ps * 1000)
        step = 0
        substrate.on_level_transition()
        while step < n_steps_per_seed:
            if obs is None:
                obs = env.reset(seed=ps * 1000)
                substrate.on_level_transition()
                continue
            action = substrate.process_random(np.asarray(obs, dtype=np.float32)) % n_actions
            obs, _, done, info = env.step(action)
            step += 1
            if done:
                obs = env.reset(seed=ps * 1000)
                substrate.on_level_transition()


def test_phase(substrate, env_fn, n_actions, env_seed, n_steps):
    """Test with novelty-based action selection. Track pred accuracy and L1."""
    env = env_fn()
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

        action = substrate.process(obs_arr) % n_actions
        obs_next, _, done, info = env.step(action)
        step += 1

        # Pred accuracy: predicted hash vs actual hash of next obs
        if prev_enc is not None and prev_action is not None and obs_next is not None:
            pred = substrate._predict_next(prev_enc, prev_action)
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
print("STEP 889 — PREDICTIVE NOVELTY (visited_set + W forward model)")
print("=" * 70)
print(f"W linear forward model. visited_set per-observation. Novelty = unvisited predicted next.")
print(f"Pretrain: random actions {PRETRAIN_STEPS} steps/seed, seeds 1-5.")
print(f"Test: cold (fresh W) vs warm (W-only transfer), {TEST_STEPS} steps, seeds 6-10.")

t0 = time.time()

for game_name, env_fn, n_actions in [("LS20", make_ls20, N_ACTIONS_LS20),
                                       ("FT09", make_ft09, N_ACTIONS_FT09)]:
    print(f"\n--- {game_name} (n_actions={n_actions}) ---")
    cold_comps = []; cold_accs = []
    warm_comps = []; warm_accs = []

    # Pretrain with random actions
    sub_pretrain = PredictiveNovelty889(n_actions=n_actions, seed=0)
    sub_pretrain.reset(0)
    pretrain_phase(sub_pretrain, env_fn, n_actions, PRETRAIN_SEEDS, PRETRAIN_STEPS)
    saved_W = sub_pretrain.W.copy()
    print(f"  Pretrain done ({time.time()-t0:.1f}s). visited_set size: {len(sub_pretrain.visited_set)}")

    for ts in TEST_SEEDS:
        # Cold: fresh W + empty visited_set
        sub_c = PredictiveNovelty889(n_actions=n_actions, seed=0)
        sub_c.reset(0)
        c_comp, c_acc = test_phase(sub_c, env_fn, n_actions, ts * 1000, TEST_STEPS)
        cold_comps.append(c_comp); cold_accs.append(c_acc)

        # Warm: trained W + empty visited_set (fresh running_mean, n_obs=0)
        sub_w = PredictiveNovelty889(n_actions=n_actions, seed=0)
        sub_w.reset(0)
        sub_w.W = saved_W.copy()
        w_comp, w_acc = test_phase(sub_w, env_fn, n_actions, ts * 1000, TEST_STEPS)
        warm_comps.append(w_comp); warm_accs.append(w_acc)

    mc = np.mean(cold_comps); mw = np.mean(warm_comps)
    mc_acc = np.mean(cold_accs); mw_acc = np.mean(warm_accs)
    print(f"  cold: L1={mc:.1f}/seed  pred_acc={mc_acc:.2f}%  ({cold_comps})")
    print(f"  warm: L1={mw:.1f}/seed  pred_acc={mw_acc:.2f}%  ({warm_comps})")
    r3_pass = mw_acc > mc_acc
    print(f"  R3_cf: {'PASS' if r3_pass else 'FAIL'} (pred_acc: {mc_acc:.2f}% → {mw_acc:.2f}%, delta={mw_acc-mc_acc:+.2f}%)")
    print(f"  L1 delta: {mw-mc:+.1f}/seed")

print(f"\nBaseline: linear W cold pred_acc=10.37% (Step 835). Random L1=36.4/seed.")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 889 DONE")
