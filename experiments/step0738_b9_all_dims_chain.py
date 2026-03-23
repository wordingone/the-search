"""
Step 738 (B9): All D1-D5 combined on chain.

R3 hypothesis: full self-directed attention = maximum R3 for encoding.
674 + D1 (channel weights) + D3 (continuous K) + D5 (adaptive centering).
Note: D2 (adaptive pool) and D4 (frame stack) were KILLs on LS20 — excluded.
Using D1+D3+D5 as the best combination.

Chain: CIFAR → LS20 → FT09 → VC33 → CIFAR.
Measure: R3 dynamics. Compare to B8 (D1+D3 only).
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import DIM, K_NAV, REFINE_EVERY, MIN_VISITS_ALIAS
from substrates.chain import SplitCIFAR100Wrapper
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 738 (B9) — ALL DIMS D1+D3+D5 CHAIN")
print("=" * 65)

SEED = 0
N_STEPS_GAME = 10_000
N_STEPS_CIFAR = 500
CHECKPOINT_EVERY = 1000
K_FINE = 20
MIN_OBS = 10
H_SPLIT = 0.05

K_MIN = 8
K_MAX = 24
I_STEP = 1
ALPHA_BASE = 1.0
EXPECTED_RATE = 10.0
RATE_WINDOW = 1000


def _enc_b9(frame, ch_weights, alpha_center):
    """D1+D5: channel-weighted + adaptive centering."""
    frame = np.array(frame, dtype=np.float32)
    if frame.ndim == 3:
        if frame.shape[0] < frame.shape[1] and frame.shape[0] < frame.shape[2]:
            frame = frame.transpose(1, 2, 0)
        h, w = frame.shape[:2]
        n_ch = frame.shape[2]
        a = np.zeros((h, w), np.float32)
        w_total = ch_weights[:n_ch].sum()
        for c in range(min(n_ch, 3)):
            ch = frame[:, :, c].astype(np.float32)
            if ch.max() > 1:
                ch = ch / 255.0
            a += ch_weights[c] * ch
        if w_total > 1e-8:
            a /= w_total
        ph, pw = max(h // 16, 1), max(w // 16, 1)
        pad_h, pad_w = ph * 16, pw * 16
        if h < pad_h or w < pad_w:
            buf = np.zeros((pad_h, pad_w), np.float32)
            buf[:min(h, pad_h), :min(w, pad_w)] = a[:min(h, pad_h), :min(w, pad_w)]
            a = buf
        pooled = a[:ph*16, :pw*16].reshape(16, ph, 16, pw).mean(axis=(1, 3))
        x = pooled.flatten()[:DIM]
        if len(x) < DIM:
            x = np.pad(x, (0, DIM - len(x)))
    else:
        x = frame.flatten()[:DIM].astype(np.float32)
        if len(x) < DIM:
            x = np.pad(x, (0, DIM - len(x)))
    return x - alpha_center * x.mean()


def _hash_k(x, H, k):
    bits = (H[:k] @ x > 0).astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


class B9_AllDims(BaseSubstrate):
    """D1+D3+D5 combined: channel weights + adaptive K + adaptive centering."""

    def __init__(self, n_actions=7, seed=0):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(K_MAX, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self._n_actions = n_actions
        self.G = {}; self.live = set()
        self.inconsistency = {}
        self.G_fine = {}; self.aliased = set()
        self._pn = self._pbase = self._pa = None
        self._pfn = None; self.t = 0; self._cn = self._fn = None
        # D1
        self.ch_weights = np.ones(3, np.float32) / 3.0
        self.ch_var_ema = np.ones(3, np.float32) * 0.01
        self.last_frame = None; self.n_frames = 0
        # D5
        self.alpha_center = ALPHA_BASE
        self._live_snapshot = 0; self._window_start = 0

    def _update_ch_weights(self, frame):
        frame = np.array(frame, dtype=np.float32)
        if frame.ndim != 3:
            return
        if frame.shape[0] < frame.shape[1] and frame.shape[0] < frame.shape[2]:
            frame = frame.transpose(1, 2, 0)
        n_ch = min(frame.shape[2], 3)
        if self.last_frame is not None and self.last_frame.ndim == 3:
            self.n_frames += 1
            alpha = min(0.05, 1.0 / max(self.n_frames, 1))
            n_ch_compare = min(n_ch, self.last_frame.shape[2])
            for c in range(n_ch_compare):
                diff_var = float(np.var(frame[:, :, c] - self.last_frame[:, :, c]))
                self.ch_var_ema[c] = (1 - alpha) * self.ch_var_ema[c] + alpha * diff_var
            total = self.ch_var_ema[:n_ch].sum()
            if total > 1e-8 and self.n_frames >= 50:
                self.ch_weights[:n_ch] = self.ch_var_ema[:n_ch] / total
        self.last_frame = frame

    def _update_alpha(self):
        if self.t - self._window_start >= RATE_WINDOW:
            new_cells = len(self.live) - self._live_snapshot
            ratio = min(2.0, (new_cells / max(RATE_WINDOW, 1) * RATE_WINDOW) / max(EXPECTED_RATE, 1))
            self.alpha_center = float(np.clip(ALPHA_BASE * ratio, 0.1, 2.0))
            self._live_snapshot = len(self.live)
            self._window_start = self.t

    def _get_k(self, base_cell):
        inc = self.inconsistency.get(base_cell, 0)
        return min(K_MIN + 2 * (inc // max(I_STEP, 1)), K_MAX)

    def _hash_fine(self, x):
        return int(np.packbits((self.H_fine @ x > 0).astype(np.uint8),
                               bitorder='big').tobytes().hex(), 16)

    def _select(self):
        if self._cn in self.aliased and self._fn is not None:
            best_a, best_s = 0, float('inf')
            for a in range(self._n_actions):
                s = sum(self.G_fine.get((self._fn, a), {}).values())
                if s < best_s:
                    best_s, best_a = s, a
            return best_a
        best_a, best_s = 0, float('inf')
        for a in range(self._n_actions):
            s = sum(self.G.get((self._cn, a), {}).values())
            if s < best_s:
                best_s, best_a = s, a
        return best_a

    def process(self, observation):
        obs_arr = np.array(observation, dtype=np.float32)
        self._update_ch_weights(obs_arr)
        self._update_alpha()
        x = _enc_b9(obs_arr, self.ch_weights, self.alpha_center)
        base = _hash_k(x, self.H, K_MIN)
        k = self._get_k(base)
        cell = _hash_k(x, self.H, k)
        fn = self._hash_fine(x)
        self.live.add(cell)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[cell] = d.get(cell, 0) + 1
            if self._pbase is not None:
                distinct = max(len(self.G.get((self._pn, a), {})) for a in range(self._n_actions))
                self.inconsistency[self._pbase] = max(self.inconsistency.get(self._pbase, 0), distinct)
            succ = self.G.get((self._pn, self._pa), {})
            if sum(succ.values()) >= MIN_VISITS_ALIAS and len(succ) >= 2:
                self.aliased.add(self._pn)
            if self._pn in self.aliased and self._pfn is not None:
                df = self.G_fine.setdefault((self._pfn, self._pa), {})
                df[fn] = df.get(fn, 0) + 1
        self._cn = cell; self._fn = fn
        action = self._select()
        self._pn = cell; self._pbase = base; self._pfn = fn; self._pa = action
        return action

    def get_state(self):
        return {"G_size": len(self.G), "live_count": len(self.live),
                "aliased_count": len(self.aliased), "t": self.t,
                "alpha_center": self.alpha_center,
                "ch_weights": self.ch_weights.copy()}

    def frozen_elements(self):
        return [
            {"name": "channel_weights", "class": "M", "justification": "D1: EMA channel variance. System-driven."},
            {"name": "inconsistency_map", "class": "M", "justification": "D3: K per cell grows. System-driven."},
            {"name": "centering_rate_tracker", "class": "M", "justification": "D5: alpha adapts to exploration. System-driven."},
            {"name": "edge_count_update", "class": "M", "justification": "G grows. System-driven."},
            {"name": "aliased_set", "class": "M", "justification": "Aliased cells grow. System-driven."},
            {"name": "H_24planes", "class": "U", "justification": "24 planes. System doesn't choose."},
            {"name": "K_MIN", "class": "U", "justification": "K_MIN=8. System doesn't choose."},
            {"name": "I_STEP", "class": "U", "justification": "I_STEP=1. System doesn't choose."},
            {"name": "binary_hash", "class": "I", "justification": "Sign projection. Irreducible."},
            {"name": "argmin_edge_count", "class": "I", "justification": "Argmin. Irreducible."},
            {"name": "fine_graph_priority", "class": "I", "justification": "Fine graph at aliased. Irreducible."},
        ]

    def reset(self, seed):
        rng = np.random.RandomState(seed * 1000)
        self.H = rng.randn(K_MAX, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.G = {}; self.live = set(); self.inconsistency = {}
        self.G_fine = {}; self.aliased = set()
        self._pn = self._pbase = self._pa = None
        self._pfn = None; self.t = 0; self._cn = self._fn = None
        self.ch_weights = np.ones(3, np.float32) / 3.0
        self.ch_var_ema = np.ones(3, np.float32) * 0.01
        self.last_frame = None; self.n_frames = 0
        self.alpha_center = ALPHA_BASE
        self._live_snapshot = 0; self._window_start = 0

    def on_level_transition(self):
        self._pn = self._pbase = self._pa = self._pfn = None

    @property
    def n_actions(self):
        return self._n_actions


def _make_env(game):
    try:
        import arcagi3
        return arcagi3.make(game)
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game)


judge = ConstitutionalJudge()

# ---- Static R3 audit ----
print("\n-- Static R3 audit --")
elems = B9_AllDims(n_actions=7, seed=0).frozen_elements()
m_names = [e["name"] for e in elems if e["class"] == "M"]
i_names = [e["name"] for e in elems if e["class"] == "I"]
u_names = [e["name"] for e in elems if e["class"] == "U"]
print(f"  B9 D1+D3+D5: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"  M: {m_names}")

# ---- CIFAR before ----
cifar = SplitCIFAR100Wrapper(n_images_per_task=N_STEPS_CIFAR)
cifar_sub = B9_AllDims(n_actions=5, seed=SEED)
cifar_sub.reset(SEED)
t0 = time.time()
print("\n-- Split-CIFAR-100 (before) --")
if not cifar._load():
    avg_acc_before = None
    print("  CIFAR not available")
else:
    rng = np.random.RandomState(SEED)
    cifar_acc_before = []
    for task_idx in range(20):
        task_images, task_labels = cifar._data[task_idx]
        idx = rng.choice(len(task_images), min(N_STEPS_CIFAR, len(task_images)), replace=False)
        correct = sum(int(cifar_sub.process(task_images[i].astype(np.float32)/255.0) % 5 == task_labels[i]) for i in idx)
        cifar_acc_before.append(correct / len(idx))
    avg_acc_before = float(np.mean(cifar_acc_before))
    cifar_sub.on_level_transition()
print(f"  avg_accuracy={avg_acc_before:.4f}")

# ---- Game chain ----
sub = B9_AllDims(n_actions=7, seed=SEED)
sub.reset(SEED)
all_phase_data = {}

for game in ["LS20", "FT09", "VC33"]:
    print(f"\n-- Phase: {game} --")
    env = _make_env(game)
    n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
    sub._n_actions = n_valid
    obs = env.reset(seed=SEED)
    level = 0; l1_step = None; steps = 0; fresh = True
    obs_seq = []; t_start = time.time()

    while steps < N_STEPS_GAME and (time.time() - t_start) < 280:
        if obs is None:
            obs = env.reset(seed=SEED); sub.on_level_transition(); fresh = True; continue
        obs_arr = np.array(obs, dtype=np.float32)
        obs_seq.append(obs_arr)
        action = sub.process(obs_arr)
        obs, reward, done, info = env.step(action % n_valid)
        steps += 1
        if fresh:
            fresh = False; continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1_step is None:
                l1_step = steps
            level = cl; sub.on_level_transition()
        if done:
            obs = env.reset(seed=SEED); sub.on_level_transition(); fresh = True

    state = sub.get_state()
    ch_w_str = [f"{v:.2f}" for v in state['ch_weights']]
    print(f"  steps={steps} l1={l1_step} live={state['live_count']} G={state['G_size']} "
          f"alpha={state['alpha_center']:.2f} chW={ch_w_str}")
    all_phase_data[game] = {"l1": l1_step}

    if obs_seq:
        class _B9(B9_AllDims):
            def __init__(self): super().__init__(n_actions=7, seed=0)
        r3 = judge.measure_r3_dynamics(_B9, obs_sequence=obs_seq[:2000], n_steps=2000, n_checkpoints=10)
        all_phase_data[game]["r3_score"] = r3.get("r3_dynamic_score")
        print(f"  R3: {r3.get('r3_dynamic_score')}")

# ---- CIFAR after ----
print("\n-- Split-CIFAR-100 (after) --")
if avg_acc_before is not None:
    rng2 = np.random.RandomState(SEED + 1)
    cifar_acc_after = []
    for task_idx in range(20):
        task_images, task_labels = cifar._data[task_idx]
        idx = rng2.choice(len(task_images), min(N_STEPS_CIFAR, len(task_images)), replace=False)
        correct = sum(int(cifar_sub.process(task_images[i].astype(np.float32)/255.0) % 5 == task_labels[i]) for i in idx)
        cifar_acc_after.append(correct / len(idx))
    avg_acc_after = float(np.mean(cifar_acc_after))
    bwt = avg_acc_after - avg_acc_before
else:
    avg_acc_after = bwt = None
print(f"  avg_accuracy={avg_acc_after} BWT={bwt}")

print("\n" + "=" * 65)
print("B9 CHAIN SUMMARY (ALL DIMS D1+D3+D5)")
print("=" * 65)
print(f"Static R3: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"CIFAR before: {avg_acc_before}  after: {avg_acc_after}  BWT: {bwt}")
for game in ["LS20", "FT09", "VC33"]:
    d = all_phase_data.get(game, {})
    print(f"{game:<8}: l1={d.get('l1')} R3_dyn={d.get('r3_score')}")
print("Compare B8 (D1+D3): LS20 l1=1105, R3=1.0")
print("=" * 65)
print("STEP 738 DONE")
print("=" * 65)
