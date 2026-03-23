"""
Step 762 - D1+D3 self-directed attention on Split-CIFAR-100.

R3 hypothesis: D1 channel weights (M element) adapt from uniform [1/3,1/3,1/3]
based on per-channel transition variance in natural images. D3 adaptive K
sharpens representation for ambiguous cells. If ch_weights ≠ uniform after
20 tasks: D1 discovered CIFAR channel structure autonomously.

Control comparison: Step 760 (674): 20.2%. Step 761 (PlainLSH): 20.0%.
Both at chance. Does D1+D3 break 20%? Or confirm chance is the R1 floor?

Key metric: final ch_weights (shows D1 autonomy), avg_accuracy vs chance.
5 seeds, 20-task protocol. Same SplitCIFAR100Wrapper as steps 760-761.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.chain import SplitCIFAR100Wrapper

print("=" * 65)
print("STEP 762 - D1+D3 SELF-DIRECTED ATTENTION ON SPLIT-CIFAR-100")
print("=" * 65)

# D3 params (from step737)
K_MIN = 8
K_MAX = 24
I_STEP = 1       # +2 bits per distinct successor
DIM = 256
REFINE_EVERY = 5000


def _enc_d1(frame, ch_weights):
    """D1: channel-weighted avgpool16 encoding."""
    frame = np.array(frame, dtype=np.float32)
    if frame.ndim == 3:
        if frame.shape[0] < frame.shape[1] and frame.shape[0] < frame.shape[2]:
            frame = frame.transpose(1, 2, 0)
        h, w = frame.shape[:2]
        n_ch = frame.shape[2]
        if n_ch == 1:
            a = frame[:, :, 0]
            if a.max() > 1:
                a = a / 255.0
        else:
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
    x = x - x.mean()
    nm = np.linalg.norm(x)
    return (x / nm).astype(np.float32) if nm > 1e-8 else x


def _hash_k(x, H, k):
    bits = (H[:k] @ x > 0).astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


class D1D3Substrate:
    """D1 (channel weights) + D3 (adaptive K) substrate for CIFAR."""

    def __init__(self, n_actions=7, seed=0):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(K_MAX, DIM).astype(np.float32)
        self._n_actions = n_actions
        self.G = {}
        self.live = set()
        self.inconsistency = {}
        self._pn = None
        self._pbase = None
        self._pa = None
        self.t = 0
        # D1: channel weights
        self.ch_weights = np.ones(3, np.float32) / 3.0
        self.ch_var_ema = np.ones(3, np.float32) * 0.01
        self.last_frame = None
        self.n_frames = 0

    def _update_ch_weights(self, frame):
        frame = np.array(frame, dtype=np.float32)
        if frame.ndim != 3:
            return
        if frame.shape[0] < frame.shape[1] and frame.shape[0] < frame.shape[2]:
            frame = frame.transpose(1, 2, 0)
        n_ch = min(frame.shape[2], 3)
        if self.last_frame is not None:
            self.n_frames += 1
            alpha = min(0.05, 1.0 / max(self.n_frames, 1))
            lf = self.last_frame
            n_ch_compare = min(n_ch, lf.shape[2]) if lf.ndim == 3 else 0
            for c in range(n_ch_compare):
                f_c = frame[:, :, c].astype(np.float32)
                l_c = lf[:, :, c].astype(np.float32)
                diff_var = float(np.var(f_c - l_c))
                self.ch_var_ema[c] = (1 - alpha) * self.ch_var_ema[c] + alpha * diff_var
            total = self.ch_var_ema[:n_ch].sum()
            if total > 1e-8 and self.n_frames >= 50:
                self.ch_weights[:n_ch] = self.ch_var_ema[:n_ch] / total
        self.last_frame = frame

    def _get_k(self, base_cell):
        inc = self.inconsistency.get(base_cell, 0)
        return min(K_MIN + 2 * (inc // max(I_STEP, 1)), K_MAX)

    def process(self, observation):
        obs_arr = np.array(observation, dtype=np.float32)
        self._update_ch_weights(obs_arr)
        x = _enc_d1(obs_arr, self.ch_weights)
        base = _hash_k(x, self.H, K_MIN)
        k = self._get_k(base)
        cell = _hash_k(x, self.H, k)
        self.live.add(cell)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[cell] = d.get(cell, 0) + 1
            if self._pbase is not None:
                distinct = max(
                    len(self.G.get((self._pn, a), {}))
                    for a in range(self._n_actions)
                )
                self.inconsistency[self._pbase] = max(
                    self.inconsistency.get(self._pbase, 0), distinct
                )
        best_a, best_s = 0, float('inf')
        for a in range(self._n_actions):
            s = sum(self.G.get((cell, a), {}).values())
            if s < best_s:
                best_s = s
                best_a = a
        self._pn = cell
        self._pbase = base
        self._pa = best_a
        return best_a

    def reset(self, seed):
        rng = np.random.RandomState(seed * 1000)
        self.H = rng.randn(K_MAX, DIM).astype(np.float32)
        self.G = {}
        self.live = set()
        self.inconsistency = {}
        self._pn = None
        self._pbase = None
        self._pa = None
        self.t = 0
        self.ch_weights = np.ones(3, np.float32) / 3.0
        self.ch_var_ema = np.ones(3, np.float32) * 0.01
        self.last_frame = None
        self.n_frames = 0

    def on_level_transition(self):
        self._pn = None
        self._pbase = None

    @property
    def n_actions(self):
        return self._n_actions

    def get_state(self):
        return {
            "ch_weights": self.ch_weights.copy(),
            "n_frames": self.n_frames,
            "live_count": len(self.live),
            "G_size": len(self.G),
            "avg_k": float(np.mean([
                self._get_k(b) for b in list(self.inconsistency.keys())[:100]
            ])) if self.inconsistency else float(K_MIN),
        }


N_SEEDS = 5
N_IMAGES_PER_TASK = 500
PER_SEED_TIME = 60

wrapper = SplitCIFAR100Wrapper(n_images_per_task=N_IMAGES_PER_TASK,
                                per_seed_time=PER_SEED_TIME)
if not wrapper._load():
    print("ERROR: CIFAR-100 not available.")
    sys.exit(1)

print(f"CIFAR-100 loaded: {wrapper.N_TASKS} tasks × {wrapper.CLASSES_PER_TASK} classes/task")

all_results = []

for seed in range(N_SEEDS):
    print(f"\n-- Seed {seed} --")
    sub = D1D3Substrate(n_actions=7, seed=seed)
    result = wrapper.run_seed(sub, seed=seed)

    if result.get("error"):
        print(f"  ERROR: {result['error']}")
    else:
        accs = result.get("task_accuracies", [])
        avg = result.get("avg_accuracy")
        bwt = result.get("backward_transfer")
        state = sub.get_state()
        ch_w = [f"{v:.3f}" for v in state["ch_weights"]]
        print(f"  tasks_completed={result.get('tasks_completed')}/20")
        if accs:
            print(f"  per-task acc: {[f'{a:.2f}' for a in accs]}")
        print(f"  avg_accuracy={avg}  BWT={bwt}  elapsed={result.get('elapsed')}s")
        print(f"  ch_weights={ch_w}  avg_k={state['avg_k']:.1f}  n_frames={state['n_frames']}")
    all_results.append(result)

valid = [r for r in all_results if r.get("avg_accuracy") is not None]
avg_accs = [r["avg_accuracy"] for r in valid]
bwts = [r["backward_transfer"] for r in valid if r.get("backward_transfer") is not None]

print("\n" + "=" * 65)
print("STEP 762 SUMMARY - D1+D3 ON SPLIT-CIFAR-100")
print("=" * 65)
if avg_accs:
    print(f"Avg accuracy (mean over seeds): {float(np.mean(avg_accs)):.4f}")
    print(f"Individual seeds: {[f'{a:.4f}' for a in avg_accs]}")
    if bwts:
        print(f"BWT (mean): {float(np.mean(bwts)):.4f}")
    print(f"")
    print(f"Chance:                  20.0%")
    print(f"Step 761 PlainLSH:       20.04%")
    print(f"Step 760 674:            20.21%")
    print(f"If D1+D3 > 20.21%: channel adaptation helps CIFAR accuracy.")
    print(f"If D1+D3 ≈ 20%:    chance is the R1 floor regardless of R3 mechanism.")
    print(f"Watch ch_weights — if non-uniform: D1 discovered color channel structure.")
else:
    print("No valid results.")
print("=" * 65)
print("STEP 762 DONE")
print("=" * 65)
