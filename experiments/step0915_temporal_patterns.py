"""
step0915_temporal_patterns.py -- Temporal action pattern memory.

R3 hypothesis: tracking SEQUENCES of actions and their outcomes allows the
substrate to learn which action CHAINS produce progress, without tracking
per-state data (not graph-banned).

Architecture vs 895h:
- Alpha/W: identical (clamped 0.1-5.0, prediction error attention)
- Action history: deque of last K actions
- sequence_outcomes: dict mapping tuple(last K actions) → EMA of alpha-weighted change
- Action selection: score[a] = sequence_outcome[(*history[-K+1:], a)]
                   Fallback to delta_per_action if sequence unseen.

Graph ban compliance:
- Key is a SEQUENCE OF ACTIONS, not (state, action).
- No state representation in any data structure.
- Cannot reproduce per-state visit counts.
- ALLOWED per Leo mail 2648.

Why this helps FT09:
- FT09's 7-click solution is a specific sequence: A, B, C, ...
- Successful sub-sequences produce alpha-weighted change
- sequence_outcomes learns: (prev_2_actions, next_action) → expected change
- Over time, productive sub-sequences chain into full solution

Parameters:
  K = 3 (sequence length)
  68 actions → max 68³ = 314K sequences, but sparse (only visited ones stored)
  At 25K steps, ~25K unique sequences. Tractable.

Leo mail 2648. Protocol: FT09 first (25K, 10 seeds cold, substrate_seed=seed).
If FT09 L1 > 0, run on LS20.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000
N_ACTIONS_FT09 = 68
N_ACTIONS_LS20 = 4
K = 3                  # sequence length
ETA_W = 0.01
ALPHA_EMA = 0.10
SEQ_EMA = 0.10         # EMA for sequence outcomes
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ENC_DIM = 256
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(scores, temp):
    x = scores / temp
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class TemporalPatterns915(BaseSubstrate):
    """Temporal action sequence memory. K-step history → EMA outcome.
    Action selection: prefer actions that complete high-outcome sequences."""

    def __init__(self, n_actions=N_ACTIONS_FT09, seed=0, epsilon=EPSILON, k=K):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._k = k
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        # 800b fallback: magnitude change per action
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        # Sequence memory: tuple(k actions) → EMA of alpha-weighted change
        self.sequence_outcomes = {}
        # Rolling action history
        self.action_history = deque(maxlen=k)
        self._pred_errors = deque(maxlen=200)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY:
            return
        mean_errors = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(mean_errors)) or np.any(np.isinf(mean_errors)):
            return
        raw_alpha = np.sqrt(np.clip(mean_errors, 0, 1e6) + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8 or np.isnan(mean_raw):
            return
        self.alpha = raw_alpha / mean_raw
        self.alpha = np.clip(self.alpha, ALPHA_LO, ALPHA_HI)

    def process(self, observation):
        enc = self._encode(observation)

        if self._prev_enc is not None and self._prev_action is not None:
            # W update (alpha-weighted prediction error)
            inp = np.concatenate([self._prev_enc * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp
            error = (enc * self.alpha) - pred

            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            # Alpha-weighted change magnitude (800b)
            weighted_delta = (enc - self._prev_enc) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * change)

            # Sequence outcome update: record outcome of completed sequence
            if len(self.action_history) >= self._k:
                seq = tuple(list(self.action_history)[-self._k:])
                if seq in self.sequence_outcomes:
                    self.sequence_outcomes[seq] = (
                        (1 - SEQ_EMA) * self.sequence_outcomes[seq] + SEQ_EMA * change
                    )
                else:
                    self.sequence_outcomes[seq] = change

        self.action_history.append(self._prev_action if self._prev_action is not None else 0)

        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            # Build prefix: last K-1 actions
            history_list = list(self.action_history)
            prefix = history_list[-(self._k - 1):] if len(history_list) >= self._k - 1 else history_list

            scores = np.array([
                self.sequence_outcomes.get(tuple(prefix + [a]), self.delta_per_action[a])
                for a in range(self._n_actions)
            ], dtype=np.float32)
            probs = softmax_action(scores, SOFTMAX_TEMP)
            action = int(self._rng.choice(self._n_actions, p=probs))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def alpha_concentration(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def sequence_count(self):
        return len(self.sequence_outcomes)

    def top_sequences(self, k=5):
        if not self.sequence_outcomes:
            return []
        return sorted(self.sequence_outcomes.items(), key=lambda x: -x[1])[:k]

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + self._n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_per_action = np.full(self._n_actions, INIT_DELTA, dtype=np.float32)
        self.sequence_outcomes = {}
        self.action_history = deque(maxlen=self._k)
        self._pred_errors = deque(maxlen=200)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None
        self.action_history.clear()

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def run_game(game_name, n_actions, test_steps):
    def make_game():
        try:
            import arcagi3; return arcagi3.make(game_name)
        except:
            import util_arcagi3; return util_arcagi3.make(game_name)

    comps = []; concs = []; seq_counts = []
    for ts in TEST_SEEDS:
        sub = TemporalPatterns915(n_actions=n_actions, seed=ts, k=K)
        sub.reset(ts)
        env = make_game(); obs = env.reset(seed=ts * 1000)
        step = 0; completions = 0; current_level = 0
        while step < test_steps:
            if obs is None:
                obs = env.reset(seed=ts * 1000); sub.on_level_transition(); continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
            obs, _, done, info = env.step(action); step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > current_level:
                completions += (cl - current_level); current_level = cl
                sub.on_level_transition()
            if done:
                obs = env.reset(seed=ts * 1000); current_level = 0
                sub.on_level_transition()
        comps.append(completions)
        concs.append(sub.alpha_concentration())
        seq_counts.append(sub.sequence_count())
        top = sub.top_sequences(3)
        print(f"  {game_name} seed={ts}: L1={completions:4d}  "
              f"alpha_conc={sub.alpha_concentration():.2f}  "
              f"seq_count={sub.sequence_count()}  "
              f"top_seq={top[:2]}")
    return comps, concs, seq_counts


print("=" * 70)
print("STEP 915 — TEMPORAL ACTION PATTERN MEMORY (K=3)")
print("=" * 70)
print(f"K={K}: track last {K} actions → EMA of alpha-weighted change.")
print(f"Graph ban compliance: key=action sequence (not state). ALLOWED.")
print(f"FT09 first: 7-click solution should produce high-scoring sub-sequences.")
print(f"25K steps, 10 seeds cold, substrate_seed=seed.")

t0 = time.time()

print(f"\n--- FT09 ({N_ACTIONS_FT09} actions) ---")
ft09_comps, ft09_concs, ft09_seqs = run_game("FT09", N_ACTIONS_FT09, TEST_STEPS)
ft09_mean = np.mean(ft09_comps); ft09_std = np.std(ft09_comps)
ft09_zero = sum(1 for x in ft09_comps if x == 0)
print(f"\nFT09 915 cold: L1={ft09_mean:.1f}/seed  std={ft09_std:.1f}  zero={ft09_zero}/10")
print(f"      {ft09_comps}")
print(f"      avg sequences learned: {np.mean(ft09_seqs):.0f}")

print(f"\nComparison:")
print(f"  895f cold (change-track, FT09):  0.0/seed  10/10 zeros  ← BASELINE")
print(f"  910b cold (compression, FT09):   0.0/seed  10/10 zeros  ← FAILED")
print(f"  915 cold (temporal K=3, FT09):   {ft09_mean:.1f}/seed  std={ft09_std:.1f}  zero={ft09_zero}/10")

if ft09_mean > 0:
    print(f"  BREAKTHROUGH: temporal sequence memory achieves L1>0 on FT09!")
    print(f"\n--- LS20 ({N_ACTIONS_LS20} actions) ---")
    ls20_comps, ls20_concs, ls20_seqs = run_game("LS20", N_ACTIONS_LS20, TEST_STEPS)
    ls20_mean = np.mean(ls20_comps); ls20_std = np.std(ls20_comps)
    ls20_zero = sum(1 for x in ls20_comps if x == 0)
    print(f"\nLS20 915 cold: L1={ls20_mean:.1f}/seed  std={ls20_std:.1f}  zero={ls20_zero}/10")
    print(f"      {ls20_comps}")
    print(f"  895h cold (change-track, LS20):  268.0/seed  0/10 zeros  ← BEST")
    print(f"  915 cold (temporal K=3, LS20):   {ls20_mean:.1f}/seed  std={ls20_std:.1f}  zero={ls20_zero}/10")
else:
    print(f"  FT09 BOTTLENECK PERSISTS. Sequence memory insufficient.")

print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 915 DONE")
