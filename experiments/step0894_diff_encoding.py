"""
step0894_diff_encoding.py -- Diff-encoded forward model on FT09 and LS20.

R3 hypothesis: encoding frame DIFFERENCES (not frames) exposes productive clicks on FT09.
FT09: 98.7% collision rate → static background → trivially predictable.
But DIFF encoding: enc(obs_t) = avgpool16(|obs_t - obs_{t-1}|). Encodes change.
Productive clicks produce visible diffs. Non-productive clicks → zero diff.
Step 558 (pre-ban): frame-diff bimodal with gap at 0.082. Signal exists.

Architecture: diff-encoded forward model W.
W predicts diff(t+1) from (diff(t), action). Delta rule.
Action: argmax_a ||W(diff, a)|| = pick action predicted to produce LARGEST diff.

Protocol: 25K steps, varied substrate seeds (seed % 4), 10 seeds each.
Games: FT09 (main test), LS20 (comparison).
Metric: L1, pred accuracy, which actions have highest predicted diff.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate

TEST_SEEDS = list(range(1, 11))  # 10 seeds
TEST_STEPS = 25_000
N_ACTIONS_FT09 = 68
N_ACTIONS_LS20 = 4
ETA = 0.01
EPSILON = 0.20
DIM = 256


def _avgpool16(obs):
    """avgpool16 encoding: pool 16x16 blocks, return 256-dim."""
    try:
        from substrates.step0674 import _enc_frame
        return _enc_frame(np.asarray(obs, dtype=np.float32))
    except:
        arr = np.asarray(obs, dtype=np.float32)
        flat = arr.flatten()
        if len(flat) >= DIM:
            return flat[:DIM]
        out = np.zeros(DIM, np.float32)
        out[:len(flat)] = flat
        return out


class DiffEncodedForwardModel(BaseSubstrate):
    """Forward model on frame diffs. Selects action with max predicted diff."""

    def __init__(self, n_actions=4, seed=0, epsilon=0.20):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        rng = np.random.RandomState(seed)
        self.W = rng.randn(DIM, DIM + n_actions).astype(np.float32) * 0.01
        self._prev_obs = None
        self._prev_diff = None
        self._prev_action = None
        self._last_enc = None
        self._running_diff_mean = np.zeros(DIM, np.float32)
        self._n_diffs = 0

    def _encode_diff(self, obs):
        obs_arr = np.asarray(obs, dtype=np.float32)
        if self._prev_obs is None:
            diff = np.zeros(DIM, np.float32)
        else:
            raw_diff = np.abs(obs_arr - self._prev_obs)
            diff = _avgpool16(raw_diff)
        return diff

    def _normalize_diff(self, diff):
        self._n_diffs += 1
        alpha = 1.0 / self._n_diffs
        self._running_diff_mean = (1 - alpha) * self._running_diff_mean + alpha * diff
        return diff - self._running_diff_mean

    def predict_next_diff(self, diff_enc, action):
        a_oh = np.zeros(self._n_actions, np.float32); a_oh[action] = 1.0
        return self.W @ np.concatenate([diff_enc, a_oh])

    def process(self, observation):
        obs_arr = np.asarray(observation, dtype=np.float32)
        diff = self._encode_diff(obs_arr)
        diff_norm = self._normalize_diff(diff)
        self._last_enc = diff_norm

        # Update W from previous step
        if self._prev_diff is not None and self._prev_action is not None:
            a_oh = np.zeros(self._n_actions, np.float32); a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_diff, a_oh])
            pred_err = self.W @ inp - diff_norm
            self.W -= ETA * np.outer(pred_err, inp)

        # Action selection: argmax predicted diff magnitude
        if self._rng.random() < self._epsilon:
            action = self._rng.randint(0, self._n_actions)
        else:
            best_a, best_score = 0, -1.0
            for a in range(self._n_actions):
                pred = self.predict_next_diff(diff_norm, a)
                score = float(np.sum(pred ** 2))
                if score > best_score:
                    best_score = score; best_a = a
            action = best_a

        self._prev_obs = obs_arr.copy()
        self._prev_diff = diff_norm.copy()
        self._prev_action = action
        return action

    def _encode_for_pred(self, obs):
        obs_arr = np.asarray(obs, dtype=np.float32)
        if self._prev_obs is not None:
            raw_diff = np.abs(obs_arr - self._prev_obs)
            diff = _avgpool16(raw_diff)
        else:
            diff = np.zeros(DIM, np.float32)
        return diff - self._running_diff_mean

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(DIM, DIM + self._n_actions).astype(np.float32) * 0.01
        self._prev_obs = None; self._prev_diff = None; self._prev_action = None
        self._last_enc = None
        self._running_diff_mean = np.zeros(DIM, np.float32)
        self._n_diffs = 0

    def on_level_transition(self):
        self._prev_obs = None; self._prev_diff = None; self._prev_action = None

    def get_state(self):
        return {"W": self.W.copy(), "rdm": self._running_diff_mean.copy(), "nd": self._n_diffs}

    def set_state(self, s):
        self.W = s["W"].copy()
        self._running_diff_mean = s["rdm"].copy()
        self._n_diffs = s["nd"]

    def frozen_elements(self): return []


def make_ft09():
    try:
        import arcagi3; return arcagi3.make("FT09")
    except:
        import util_arcagi3; return util_arcagi3.make("FT09")


def make_ls20():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_phase(substrate, env_fn, n_actions, env_seed, n_steps):
    env = env_fn()
    obs = env.reset(seed=env_seed)
    step = 0; pred_errors = []; completions = 0; current_level = 0
    prev_diff = None

    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); substrate.on_level_transition(); continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr) % n_actions
        obs_next, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs_next = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
        # Pred accuracy tracking
        if prev_diff is not None and obs_next is not None and substrate._last_enc is not None:
            next_diff = substrate._encode_for_pred(np.asarray(obs_next, dtype=np.float32))
            pred = substrate.predict_next_diff(prev_diff, action)
            err = float(np.sum((pred - next_diff) ** 2))
            norm = float(np.sum(next_diff ** 2)) + 1e-8
            pred_errors.append((err, norm))
        if substrate._last_enc is not None:
            prev_diff = substrate._last_enc.copy()
        obs = obs_next

    pred_acc = None
    if pred_errors:
        te = sum(e for e, n in pred_errors)
        tn = sum(n for e, n in pred_errors)
        pred_acc = float(1.0 - te / tn) * 100.0
    return completions, pred_acc


print("=" * 70)
print("STEP 894 — DIFF-ENCODED FORWARD MODEL (FT09 + LS20)")
print("=" * 70)
print(f"Encoding: avgpool16(|obs_t - obs_{{t-1}}|). Forward model predicts next diff.")

t0 = time.time()

for game_name, env_fn, n_actions in [("FT09", make_ft09, N_ACTIONS_FT09), ("LS20", make_ls20, N_ACTIONS_LS20)]:
    print(f"\n--- {game_name} (n_actions={n_actions}) ---")
    l1_list = []; acc_list = []

    for s in TEST_SEEDS:
        substrate_seed = s % 4
        sub = DiffEncodedForwardModel(n_actions=n_actions, seed=substrate_seed, epsilon=EPSILON)
        sub.reset(substrate_seed)
        c, a = run_phase(sub, env_fn, n_actions, s * 1000, TEST_STEPS)
        l1_list.append(c); acc_list.append(a)

    valid_acc = [a for a in acc_list if a is not None]
    mean_l1 = np.mean(l1_list)
    mean_acc = np.mean(valid_acc) if valid_acc else None
    print(f"  L1: {mean_l1:.1f}/seed  (per seed: {l1_list})")
    print(f"  Pred acc: {mean_acc:.2f}%" if mean_acc else "  Pred acc: N/A")

    # Report top 3 actions by diff magnitude (from last substrate)
    if hasattr(sub, 'W'):
        # Compute predicted diff magnitude per action with zero diff input
        zero_diff = np.zeros(DIM, np.float32)
        scores = []
        for a in range(min(n_actions, 8)):
            pred = sub.predict_next_diff(zero_diff, a)
            scores.append((float(np.sum(pred ** 2)), a))
        scores.sort(reverse=True)
        print(f"  Top actions by predicted diff (action, score): {scores[:3]}")

print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 894 DONE")
