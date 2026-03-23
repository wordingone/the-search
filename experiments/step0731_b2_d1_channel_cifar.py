"""
Step 731 (B2): D1 Channel-weighted encoding on Split-CIFAR-100.

R3 hypothesis: RGB channels carry class-relevant signal. Channel weighting via
per-channel transition variance selects class-relevant channels → improved NMI.
Compare to A2 baseline (greyscale, NMI=0.013).

Measure: NMI per task. Compare channel weights before/after each task.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import DIM, K_NAV, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT
from substrates.chain import SplitCIFAR100Wrapper
from substrates.judge import ConstitutionalJudge
from sklearn.metrics import normalized_mutual_info_score

print("=" * 65)
print("STEP 731 (B2) — D1 CHANNEL-WEIGHTED ENCODING ON CIFAR-100")
print("=" * 65)

SEED = 0
N_PER_TASK = 500

# ---- D1 CIFAR encoding ----

def _enc_d1_cifar(frame, ch_weights):
    """D1: channel-weighted encoding for RGB images."""
    frame = np.array(frame, dtype=np.float32)
    if frame.ndim == 3:
        if frame.shape[0] < frame.shape[1] and frame.shape[0] < frame.shape[2]:
            frame = frame.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
        h, w = frame.shape[:2]
        n_ch = frame.shape[2]
        # Weighted sum across channels
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
    return x - x.mean()


class D1_CIFAR(BaseSubstrate):
    """D1 on CIFAR: adaptive channel weights from per-channel transition variance."""

    def __init__(self, n_actions=5, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self._n_actions = n_actions
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self.G_fine = {}
        self.aliased = set()
        self._pn = self._pa = self._px = None
        self._pfn = None
        self.t = 0
        self._cn = self._fn = None
        # D1
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
        if self.last_frame is not None and self.last_frame.ndim == 3:
            self.n_frames += 1
            alpha = min(0.05, 1.0 / max(self.n_frames, 1))
            n_ch_compare = min(n_ch, self.last_frame.shape[2])
            for c in range(n_ch_compare):
                f_c = frame[:, :, c].astype(np.float32)
                l_c = self.last_frame[:, :, c].astype(np.float32)
                diff_var = float(np.var(f_c - l_c))
                self.ch_var_ema[c] = (1 - alpha) * self.ch_var_ema[c] + alpha * diff_var
            total = self.ch_var_ema[:n_ch].sum()
            if total > 1e-8 and self.n_frames >= 50:
                self.ch_weights[:n_ch] = self.ch_var_ema[:n_ch] / total
        self.last_frame = frame

    def _hash_nav(self, x):
        return int(np.packbits((self.H_nav @ x > 0).astype(np.uint8),
                               bitorder='big').tobytes().hex(), 16)

    def _hash_fine(self, x):
        return int(np.packbits((self.H_fine @ x > 0).astype(np.uint8),
                               bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_nav(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

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

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref:
                continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS:
                continue
            v = np.array(list(d.values()), np.float64)
            p = v / v.sum()
            h = float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))
            if h < H_SPLIT:
                continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0]))
            r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3:
                continue
            diff = r0[0]/r0[1] - r1[0]/r1[1]
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)
            did += 1
            if did >= 3:
                break

    def process(self, observation):
        obs_arr = np.array(observation, dtype=np.float32)
        self._update_ch_weights(obs_arr)
        x = _enc_d1_cifar(obs_arr, self.ch_weights)
        n = self._node(x)
        fn = self._hash_fine(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(DIM, np.float64), 0))
            self.C[k] = (s + x.astype(np.float64), c + 1)
            succ = self.G.get((self._pn, self._pa), {})
            if sum(succ.values()) >= MIN_VISITS_ALIAS and len(succ) >= 2:
                self.aliased.add(self._pn)
            if self._pn in self.aliased and self._pfn is not None:
                df = self.G_fine.setdefault((self._pfn, self._pa), {})
                df[fn] = df.get(fn, 0) + 1
        self._px = x
        self._cn = n
        self._fn = fn
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        action = self._select()
        self._pn = n
        self._pfn = fn
        self._pa = action
        return action

    def get_state(self):
        return {
            "G_size": len(self.G),
            "live_count": len(self.live),
            "aliased_count": len(self.aliased),
            "ref_count": len(self.ref),
            "t": self.t,
            "ch_weights": self.ch_weights.copy(),
        }

    def frozen_elements(self):
        return [
            {"name": "channel_weights", "class": "M",
             "justification": "ch_weights updated via EMA of per-channel transition variance. System-driven."},
            {"name": "edge_count_update", "class": "M",
             "justification": "G updated by transitions. System-driven."},
            {"name": "aliased_set", "class": "M",
             "justification": "Aliased cells grow. System-driven."},
            {"name": "ref_hyperplanes", "class": "M",
             "justification": "Refinement planes from frame diffs. System-derived."},
            {"name": "avgpool16", "class": "U",
             "justification": "16x16 pooling. System doesn't choose."},
            {"name": "mean_centering", "class": "U",
             "justification": "Subtract mean. System doesn't choose."},
            {"name": "H_nav_planes", "class": "U",
             "justification": "k=12 random planes. System doesn't choose."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection. Irreducible."},
            {"name": "argmin_edge_count", "class": "I",
             "justification": "Argmin of outgoing edges. Irreducible."},
            {"name": "fine_graph_priority", "class": "I",
             "justification": "Fine graph at aliased cells. Irreducible."},
        ]

    def reset(self, seed):
        rng = np.random.RandomState(seed * 1000)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self.G_fine = {}
        self.aliased = set()
        self._pn = self._pa = self._px = None
        self._pfn = None
        self.t = 0
        self._cn = self._fn = None
        self.ch_weights = np.ones(3, np.float32) / 3.0
        self.ch_var_ema = np.ones(3, np.float32) * 0.01
        self.last_frame = None
        self.n_frames = 0

    def on_level_transition(self):
        self._pn = None
        self._pfn = None
        self._px = None

    @property
    def n_actions(self):
        return self._n_actions


judge = ConstitutionalJudge()

# ---- Static R3 audit ----
print("\n-- Static R3 audit --")
elems = D1_CIFAR(n_actions=5, seed=0).frozen_elements()
m_names = [e["name"] for e in elems if e["class"] == "M"]
i_names = [e["name"] for e in elems if e["class"] == "I"]
u_names = [e["name"] for e in elems if e["class"] == "U"]
print(f"  D1_CIFAR: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"  M elements: {m_names}")
print(f"  U elements: {u_names}")

# ---- CIFAR run ----
print("\n-- Split-CIFAR-100 (20 tasks x 500 imgs) --")
cifar = SplitCIFAR100Wrapper(n_images_per_task=N_PER_TASK)
sub = D1_CIFAR(n_actions=5, seed=SEED)
sub.reset(SEED)

if not cifar._load():
    print("  CIFAR not available")
else:
    rng = np.random.RandomState(SEED)
    task_nmi = []
    task_ch_weights = []

    for task_idx in range(20):
        task_images, task_labels = cifar._data[task_idx]
        n_imgs = min(N_PER_TASK, len(task_images))
        idx = rng.choice(len(task_images), n_imgs, replace=False)

        cells = []
        for i in idx:
            obs = task_images[i].astype(np.float32) / 255.0
            action = sub.process(obs)
            cells.append(sub._cn)

        labels_used = [task_labels[i] for i in idx]
        try:
            nmi = normalized_mutual_info_score(labels_used, cells)
        except Exception:
            nmi = 0.0

        state = sub.get_state()
        task_nmi.append(nmi)
        task_ch_weights.append(state["ch_weights"].copy())
        w = [f"{v:.3f}" for v in state["ch_weights"]]
        print(f"  task={task_idx:>2} NMI={nmi:.4f} ch_weights={w} G={state['G_size']}")
        sub.on_level_transition()

    avg_nmi = float(np.mean(task_nmi))
    print(f"\n  avg NMI: {avg_nmi:.4f} (A2 baseline: ~0.013)")
    final_weights = sub.ch_weights
    print(f"  Final ch_weights: {[f'{v:.3f}' for v in final_weights]}")
    print(f"  Weight spread (max-min): {float(final_weights.max() - final_weights.min()):.4f}")

    verdict = "PASS" if avg_nmi > 0.013 else "KILL"
    print(f"\n  Verdict: {verdict} (avg NMI {'>' if avg_nmi > 0.013 else '<='} A2 0.013)")

print("\n" + "=" * 65)
print("B2 SUMMARY")
print("=" * 65)
print(f"D1_CIFAR R3 static: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
if cifar._load():
    print(f"avg NMI: {avg_nmi:.4f} vs A2 baseline 0.013 → {verdict}")
print("=" * 65)
print("STEP 731 DONE")
print("=" * 65)
