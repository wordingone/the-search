"""
step0922_long_phase1_filter.py -- Long phase 1 (15K) + filtered sequence memory.

Step 921 failed: 5K phase 1 → alpha_conc=3-13 → wrong top_K filter (9/10 seeds miss {60,51,52}).
Fix: 15K phase 1 → alpha_conc reliably high → delta_per_action reliable → correct top_K.

R3 hypothesis: Alpha after 15K steps reliably concentrates on puzzle tile dims [60,51,52].
Delta_per_action for tile-60/51/52 actions will be high (they change alpha-attended dims).
Filtering to top-6 after 15K captures the correct tiles. Sequence memory on 6^3=216 seqs.

Key diagnostic: at step 15K, log top_K and overlap with baseline_actions [17,19,15,21,65,26].

Protocol: FT09, 25K total (15K phase 1 + 10K phase 2), 10 seeds cold.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque, defaultdict
from substrates.step0674 import _enc_frame

ENC_DIM = 256
N_ACTIONS = 68
BASELINE_ACTIONS = [17, 19, 15, 21, 65, 26]
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON_P1 = 0.20
EPSILON_P2 = 0.30   # higher for more exploration among filtered set
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
PHASE1_STEPS = 15_000
FILTER_K = 6
SEQ_LEN = 3
SEQ_EMA = 0.10
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_select(scores, temp, rng):
    x = np.array(scores) / temp
    x -= np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(scores), p=probs))


class LongPhase1Filter922:

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + N_ACTIONS), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_per_action = np.full(N_ACTIONS, INIT_DELTA, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self._prev_enc = None; self._prev_action = None
        self._step = 0
        self._top_K = None
        self._seq_outcomes = defaultdict(lambda: INIT_DELTA)
        self._action_history = deque(maxlen=SEQ_LEN - 1)
        self._filter_log = None  # logged once at phase transition

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1-a)*self._running_mean + a*enc_raw
        return enc_raw - self._running_mean

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY: return
        me = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(me)) or np.any(np.isinf(me)): return
        ra = np.sqrt(np.clip(me, 0, 1e6) + 1e-8)
        mr = np.mean(ra)
        if mr < 1e-8 or np.isnan(mr): return
        self.alpha = np.clip(ra / mr, ALPHA_LO, ALPHA_HI)

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def process(self, obs):
        enc = self._encode(obs)
        self._step += 1

        if self._prev_enc is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_enc * self.alpha,
                                   one_hot(self._prev_action, N_ACTIONS)])
            pred = self.W @ inp
            error = (enc * self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0: error *= 10.0/en
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()
            weighted_delta = (enc - self._prev_enc) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1-ALPHA_EMA)*self.delta_per_action[a] + ALPHA_EMA*change
            if self._step > PHASE1_STEPS and len(self._action_history) >= SEQ_LEN-1:
                seq = tuple(self._action_history) + (a,)
                self._seq_outcomes[seq] = (1-SEQ_EMA)*self._seq_outcomes[seq] + SEQ_EMA*change

        # Phase transition
        if self._step == PHASE1_STEPS and self._top_K is None:
            self._top_K = list(np.argsort(self.delta_per_action)[-FILTER_K:])
            overlap = [a for a in self._top_K if a in BASELINE_ACTIONS]
            self._filter_log = (self._top_K, overlap, self.alpha_conc())

        # Action selection
        if self._step <= PHASE1_STEPS or self._top_K is None:
            if self._rng.random() < EPSILON_P1:
                action = int(self._rng.randint(0, N_ACTIONS))
            else:
                x = self.delta_per_action / SOFTMAX_TEMP
                x -= np.max(x); e = np.exp(x)
                probs = e / (e.sum() + 1e-12)
                action = int(self._rng.choice(N_ACTIONS, p=probs))
        else:
            if self._rng.random() < EPSILON_P2:
                action = int(self._rng.choice(self._top_K))
            else:
                history = tuple(self._action_history)
                scores = [self._seq_outcomes[history+(a,)] for a in self._top_K]
                idx = softmax_select(scores, SOFTMAX_TEMP, self._rng)
                action = self._top_K[idx]

        self._prev_enc = enc.copy()
        self._prev_action = action
        self._action_history.append(action)
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None
        self._action_history.clear()


def make_game():
    try:
        import arcagi3; return arcagi3.make("FT09")
    except:
        import util_arcagi3; return util_arcagi3.make("FT09")


results = []
t0 = time.time()
print("=" * 70)
print("STEP 922 — LONG PHASE 1 (15K) + FILTERED SEQUENCE MEMORY (FT09)")
print("=" * 70)
print(f"Phase 1: 0-15K steps, 895h all 68 actions. Phase 2: top-{FILTER_K} + seq memory.")
print(f"Target tiles: {BASELINE_ACTIONS}")

for seed in TEST_SEEDS:
    sub = LongPhase1Filter922(seed=seed)
    env = make_game()
    obs = env.reset(seed=seed * 1000)
    step = 0; completions = 0; current_level = 0
    while step < TEST_STEPS:
        if obs is None:
            obs = env.reset(seed=seed * 1000); sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32))
        obs, _, done, info = env.step(action)
        step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=seed * 1000); current_level = 0
            sub.on_level_transition()
    results.append(completions)
    fl = sub._filter_log
    if fl:
        top_k, overlap, conc = fl
        print(f"  seed={seed}: L1={completions:4d}  alpha_conc@15K={conc:.2f}  "
              f"top_K={top_k}  overlap_with_baseline={overlap}({len(overlap)}/6)")
    else:
        print(f"  seed={seed}: L1={completions:4d}  (phase 2 not reached?)")

mean = np.mean(results)
zeros = sum(1 for x in results if x == 0)
print(f"\n{'='*70}")
print(f"STEP 922 RESULTS (long phase 1 filter):")
print(f"  FT09: L1={mean:.1f}/seed  std={np.std(results):.1f}  zero={zeros}/10")
print(f"  {results}")
print(f"\nComparison:")
print(f"  921 K=5 (5K phase 1):  0.0/seed  10/10 zeros (wrong tiles)")
print(f"  922 K=6 (15K phase 1): {mean:.1f}/seed  {zeros}/10 zeros")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 922 DONE")
