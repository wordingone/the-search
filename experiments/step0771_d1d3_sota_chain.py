"""
Step 771 - SOTA chain with D1+D3: LS20 pretraining → Split-CIFAR-100.

Control variant of Step 770 using D1+D3 substrate (channel adaptation + adaptive K).
Same protocol: 10K LS20 steps → CIFAR-100 (no substrate reset).

R3 hypothesis: D1 ch_weights adapt during LS20 (where gray luminance dominates),
then when transitioning to CIFAR (full RGB), D1 must re-adapt. Does this re-adaptation
show R3 dynamics at domain boundary? Does prior LS20 channel structure help or hurt CIFAR?

Compare: Step 770 (674 cold LS20→CIFAR). Step 762 (D1+D3 cold CIFAR): 19.65%.

Key: ch_weights before and after domain transition. If ch_weights shift at CIFAR
boundary: D1 detects the domain change. If accuracy > step762: LS20 pretraining helps.

3 seeds, same protocol as step 770.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.chain import SplitCIFAR100Wrapper

print("=" * 65)
print("STEP 771 - D1+D3 SOTA CHAIN: LS20 → CIFAR-100")
print("=" * 65)

K_MIN = 8; K_MAX = 24; I_STEP = 1; DIM = 256


def _enc_d1(frame, ch_weights):
    frame = np.array(frame, dtype=np.float32)
    if frame.ndim == 3:
        if frame.shape[0] < frame.shape[1] and frame.shape[0] < frame.shape[2]:
            frame = frame.transpose(1, 2, 0)
        h, w = frame.shape[:2]
        n_ch = frame.shape[2]
        if n_ch == 1:
            a = frame[:, :, 0]
            if a.max() > 1: a = a / 255.0
        else:
            a = np.zeros((h, w), np.float32)
            w_total = ch_weights[:n_ch].sum()
            for c in range(min(n_ch, 3)):
                ch = frame[:, :, c].astype(np.float32)
                if ch.max() > 1: ch = ch / 255.0
                a += ch_weights[c] * ch
            if w_total > 1e-8: a /= w_total
        ph, pw = max(h // 16, 1), max(w // 16, 1)
        buf = np.zeros((ph * 16, pw * 16), np.float32)
        buf[:min(h, ph*16), :min(w, pw*16)] = a[:min(h, ph*16), :min(w, pw*16)]
        pooled = buf.reshape(16, ph, 16, pw).mean(axis=(1, 3))
        x = pooled.flatten()[:DIM]
        if len(x) < DIM: x = np.pad(x, (0, DIM - len(x)))
    else:
        x = frame.flatten()[:DIM].astype(np.float32)
        if len(x) < DIM: x = np.pad(x, (0, DIM - len(x)))
    x = x - x.mean()
    nm = np.linalg.norm(x)
    return (x / nm).astype(np.float32) if nm > 1e-8 else x


def _hash_k(x, H, k):
    bits = (H[:k] @ x > 0).astype(np.uint8)
    val = 0
    for b in bits: val = (val << 1) | int(b)
    return val


class D1D3Substrate:
    def __init__(self, n_actions=7, seed=0):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(K_MAX, DIM).astype(np.float32)
        self._n_actions = n_actions
        self.G = {}; self.live = set(); self.inconsistency = {}
        self._pn = self._pbase = self._pa = None; self.t = 0
        self.ch_weights = np.ones(3, np.float32) / 3.0
        self.ch_var_ema = np.ones(3, np.float32) * 0.01
        self.last_frame = None; self.n_frames = 0

    def _update_ch(self, frame):
        frame = np.array(frame, dtype=np.float32)
        if frame.ndim != 3: return
        if frame.shape[0] < frame.shape[1] and frame.shape[0] < frame.shape[2]:
            frame = frame.transpose(1, 2, 0)
        n_ch = min(frame.shape[2], 3)
        if self.last_frame is not None:
            self.n_frames += 1
            alpha = min(0.05, 1.0 / max(self.n_frames, 1))
            lf = self.last_frame
            n_cc = min(n_ch, lf.shape[2]) if lf.ndim == 3 else 0
            for c in range(n_cc):
                diff_var = float(np.var(frame[:,:,c].astype(np.float32) - lf[:,:,c].astype(np.float32)))
                self.ch_var_ema[c] = (1 - alpha) * self.ch_var_ema[c] + alpha * diff_var
            total = self.ch_var_ema[:n_ch].sum()
            if total > 1e-8 and self.n_frames >= 50:
                self.ch_weights[:n_ch] = self.ch_var_ema[:n_ch] / total
        self.last_frame = frame

    def _get_k(self, base): return min(K_MIN + 2 * (self.inconsistency.get(base, 0) // max(I_STEP, 1)), K_MAX)

    def process(self, observation):
        obs = np.array(observation, dtype=np.float32)
        self._update_ch(obs)
        x = _enc_d1(obs, self.ch_weights)
        base = _hash_k(x, self.H, K_MIN); k = self._get_k(base); cell = _hash_k(x, self.H, k)
        self.live.add(cell); self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {}); d[cell] = d.get(cell, 0) + 1
            if self._pbase is not None:
                distinct = max(len(self.G.get((self._pn, a), {})) for a in range(self._n_actions))
                self.inconsistency[self._pbase] = max(self.inconsistency.get(self._pbase, 0), distinct)
        best_a, best_s = 0, float('inf')
        for a in range(self._n_actions):
            s = sum(self.G.get((cell, a), {}).values())
            if s < best_s: best_s = s; best_a = a
        self._pn = cell; self._pbase = base; self._pa = best_a
        return best_a

    def reset(self, seed):
        rng = np.random.RandomState(seed * 1000)
        self.H = rng.randn(K_MAX, DIM).astype(np.float32)
        self.G = {}; self.live = set(); self.inconsistency = {}
        self._pn = self._pbase = self._pa = None; self.t = 0
        self.ch_weights = np.ones(3, np.float32) / 3.0
        self.ch_var_ema = np.ones(3, np.float32) * 0.01
        self.last_frame = None; self.n_frames = 0

    def on_level_transition(self):
        self._pn = self._pbase = None

    @property
    def n_actions(self): return self._n_actions

    def get_state(self):
        return {"ch_weights": self.ch_weights.copy(), "n_frames": self.n_frames,
                "live_count": len(self.live), "G_size": len(self.G)}


def _make_ls20():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3; return util_arcagi3.make("LS20")


N_SEEDS = 3; LS20_STEPS = 10_000; N_IMAGES_PER_TASK = 500; PER_SEED_TIME = 60

wrapper = SplitCIFAR100Wrapper(n_images_per_task=N_IMAGES_PER_TASK, per_seed_time=PER_SEED_TIME)
if not wrapper._load():
    print("ERROR: CIFAR-100 not available."); sys.exit(1)
print(f"CIFAR-100 loaded: {wrapper.N_TASKS} tasks × {wrapper.CLASSES_PER_TASK} classes/task")

all_results = []

for seed in range(N_SEEDS):
    print(f"\n-- Seed {seed} --")
    sub = D1D3Substrate(n_actions=7, seed=seed)

    # Phase 1: LS20
    print(f"  Phase 1: LS20 pretraining ({LS20_STEPS} steps)...")
    try:
        env = _make_ls20()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        obs = env.reset(seed=seed * 100); steps = 0
        while steps < LS20_STEPS:
            if obs is None:
                obs = env.reset(seed=seed * 100); sub.on_level_transition(); continue
            action = sub.process(np.array(obs, dtype=np.float32))
            obs, _, done, _ = env.step(action % n_valid); steps += 1
            if done:
                obs = env.reset(seed=seed * 100); sub.on_level_transition()
        state_ls20 = sub.get_state()
        ch_ls20 = [f"{v:.3f}" for v in state_ls20["ch_weights"]]
        print(f"  LS20: live={state_ls20['live_count']} ch_weights={ch_ls20}")
    except Exception as e:
        print(f"  LS20 error: {e}")
        all_results.append({"seed": seed, "error": str(e)}); continue

    # Phase 2: CIFAR (no reset)
    print(f"  Phase 2: CIFAR (no reset)...")
    result = wrapper.run_seed(sub, seed=seed)
    if result.get("error"):
        print(f"  CIFAR ERROR: {result['error']}")
    else:
        accs = result.get("task_accuracies", [])
        avg = result.get("avg_accuracy"); bwt = result.get("backward_transfer")
        state_cifar = sub.get_state()
        ch_cifar = [f"{v:.3f}" for v in state_cifar["ch_weights"]]
        print(f"  tasks={result.get('tasks_completed')}/20 avg={avg} BWT={bwt}")
        print(f"  ch_weights after CIFAR={ch_cifar}")
    result["seed"] = seed; result["ch_ls20"] = ch_ls20
    all_results.append(result)

valid = [r for r in all_results if r.get("avg_accuracy") is not None]
avg_accs = [r["avg_accuracy"] for r in valid]
bwts = [r["backward_transfer"] for r in valid if r.get("backward_transfer") is not None]

print("\n" + "=" * 65)
print("STEP 771 SUMMARY - D1+D3 SOTA CHAIN: LS20 → CIFAR")
print("=" * 65)
if avg_accs:
    print(f"Avg accuracy: {float(np.mean(avg_accs)):.4f} (seeds: {[f'{a:.4f}' for a in avg_accs]})")
    if bwts: print(f"BWT: {float(np.mean(bwts)):.4f}")
    print(f"\nBaseline (Step 762 D1+D3 cold CIFAR):       19.65%")
    print(f"Baseline (Step 770 674 LS20→CIFAR):         see step770 result")
    print(f"Baseline (Step 760 674 cold CIFAR):          20.21%")
    print(f"If > 19.65%: LS20 pretraining helps D1+D3 on CIFAR.")
else:
    print("No valid results.")
print("=" * 65)
print("STEP 771 DONE")
print("=" * 65)
