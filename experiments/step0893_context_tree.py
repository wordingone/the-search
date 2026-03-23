"""
step0893_context_tree.py -- Compression-based prediction (Context Tree approach).

R3 hypothesis: a dictionary-based prediction (no W matrix, no gradients) can predict
next observation transitions. If game is deterministic (same seed, same state → same
next state), a context tree converges to exact prediction quickly.

Architecture:
- Transition dictionary: T[(hash(enc), action)] → most_common_next_hash
- For each action, predict next_hash. Novel if predicted hash not in visited set.
- Action: argmax_a [novelty(T.get(hash(enc), a))]
- Falls back to least-visited action if no prediction available.

No W, no gradients. R1/R2 compliant: no objective function, no optimizer.
The "dynamics model" IS the dictionary.

Protocol: 25K steps, varied seeds (1-10), LS20 (deterministic).
Metric: L1, prediction accuracy (exact hash match rate), growth of novel T entries.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import defaultdict
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000
N_ACTIONS = 4
EPSILON = 0.10  # small epsilon for exploitation of predictions


def enc_hash(enc):
    """Coarse hash of encoding vector (8-bit per element, 32 elements)."""
    # Use first 32 elements, quantize to 4 bits
    coarse = (enc[:32] * 4).astype(np.int8)
    return hash(coarse.tobytes())


class ContextTreeSubstrate(BaseSubstrate):
    """Dictionary-based prediction substrate. No W, no gradients."""

    def __init__(self, n_actions=4, seed=0, epsilon=0.10):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self._epsilon = epsilon
        # Transition counts: T[(state_hash, action)] → {next_hash: count}
        self._transitions = defaultdict(lambda: defaultdict(int))
        # Visit counts per state_hash
        self._visit_counts = defaultdict(int)
        self._running_mean = np.zeros(256, np.float32)
        self._n_obs = 0
        self._prev_hash = None
        self._prev_action = None
        self._last_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self._running_mean = (1 - alpha) * self._running_mean + alpha * enc_raw
        return enc_raw - self._running_mean

    def _predict_next_hash(self, state_hash, action):
        """Predict next state hash given (state_hash, action). Returns (pred_hash, confidence)."""
        key = (state_hash, action)
        if key not in self._transitions or not self._transitions[key]:
            return None, 0.0
        next_counts = self._transitions[key]
        best_next = max(next_counts, key=next_counts.get)
        total = sum(next_counts.values())
        return best_next, next_counts[best_next] / total

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        h = enc_hash(enc)

        # Update transition dictionary
        if self._prev_hash is not None and self._prev_action is not None:
            self._transitions[(self._prev_hash, self._prev_action)][h] += 1
        self._visit_counts[h] += 1

        if self._rng.random() < self._epsilon:
            action = self._rng.randint(0, self._n_actions)
        else:
            # Score each action by novelty of predicted next state
            scores = []
            for a in range(self._n_actions):
                pred_hash, conf = self._predict_next_hash(h, a)
                if pred_hash is None:
                    # Unknown transition → maximally novel
                    novel_score = 1.0
                else:
                    # Novel if predicted state is infrequently visited
                    visit = self._visit_counts.get(pred_hash, 0)
                    novel_score = 1.0 / (visit + 1)  # inverse visit count = novelty
                scores.append(novel_score)

            # argmax novelty
            action = int(np.argmax(scores))

        self._prev_hash = h
        self._prev_action = action
        return action

    def get_prediction_accuracy(self, prev_hash, action, actual_next_hash):
        """Check if our prediction was correct."""
        pred_hash, _ = self._predict_next_hash(prev_hash, action)
        return pred_hash == actual_next_hash

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._transitions = defaultdict(lambda: defaultdict(int))
        self._visit_counts = defaultdict(int)
        self._running_mean = np.zeros(256, np.float32)
        self._n_obs = 0
        self._prev_hash = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_hash = None; self._prev_action = None

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


print("=" * 70)
print("STEP 893 — CONTEXT TREE PREDICTION (no W, no gradients)")
print("=" * 70)
print(f"Dictionary-based transitions. Action: argmax novelty(predicted_next).")
print(f"eps={EPSILON}, seeds 1-10, 25K steps each.")

t0 = time.time()
l1_list = []; dict_sizes = []; acc_list = []

for s in TEST_SEEDS:
    substrate_seed = s % 4
    sub = ContextTreeSubstrate(n_actions=N_ACTIONS, seed=substrate_seed, epsilon=EPSILON)
    sub.reset(substrate_seed)
    env = make_game(); obs = env.reset(seed=s * 1000)
    completions = 0; current_level = 0; step = 0
    correct_preds = 0; total_preds = 0

    while step < TEST_STEPS:
        if obs is None:
            obs = env.reset(seed=s * 1000); current_level = 0
            sub.on_level_transition(); continue
        prev_hash = sub._prev_hash
        prev_action = sub._prev_action

        action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs_next, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs_next = env.reset(seed=s * 1000); current_level = 0
            sub.on_level_transition()

        # Check prediction accuracy
        if prev_hash is not None and prev_action is not None and obs_next is not None:
            next_enc = sub._encode(np.asarray(obs_next, dtype=np.float32))
            next_h = enc_hash(next_enc)
            if sub.get_prediction_accuracy(prev_hash, prev_action, next_h):
                correct_preds += 1
            total_preds += 1

        obs = obs_next

    dict_size = len(sub._transitions)
    pred_acc = correct_preds / max(total_preds, 1) * 100.0
    l1_list.append(completions); dict_sizes.append(dict_size); acc_list.append(pred_acc)
    print(f"  seed={s:3d}: L1={completions:4d}  dict_size={dict_size:5d}  pred_acc={pred_acc:.2f}%")

print()
print(f"Mean L1: {np.mean(l1_list):.1f}/seed  (random=36.4)")
print(f"Mean dict size: {np.mean(dict_sizes):.0f} (state,action) entries")
print(f"Mean pred accuracy: {np.mean(acc_list):.2f}% (exact hash match)")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 893 DONE")
