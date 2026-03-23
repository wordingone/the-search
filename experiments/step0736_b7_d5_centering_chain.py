"""
Step 736 (B7): D5 Adaptive centering rate on chain.

R3 hypothesis: different exploration phases need different centering speed.
α(t) = α_base * (new_cells_last_1000 / expected_rate).
High discovery → fast centering (adapt to new domain quickly).
Low discovery → stable centering (preserve learned structure).
centering_rate_tracker is the M element.

Chain: CIFAR → LS20 → FT09 → VC33 → CIFAR.
Measure: forgetting delta, per-phase performance, centering rate over time.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import DIM, K_NAV, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT
from substrates.chain import SplitCIFAR100Wrapper
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 736 (B7) — D5 ADAPTIVE CENTERING RATE CHAIN")
print("=" * 65)

SEED = 0
N_STEPS_GAME = 10_000
N_STEPS_CIFAR = 500
CHECKPOINT_EVERY = 1000

ALPHA_BASE = 1.0        # baseline centering strength
EXPECTED_RATE = 10.0    # expected new cells per 1000 steps (stable phase)
RATE_WINDOW = 1000      # window for tracking new cell rate


def _enc_d5(frame, alpha_center):
    """D5: adaptive-centering encoding. alpha_center in [0,1] controls subtraction."""
    frame = np.array(frame, dtype=np.float32)
    if frame.ndim == 3:
        if frame.shape[0] < frame.shape[1] and frame.shape[0] < frame.shape[2]:
            frame = frame.transpose(1, 2, 0)
        h, w = frame.shape[:2]
        a = frame[:, :, 0].astype(np.float32)
        if a.max() > 1:
            a = a / 15.0
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
    # D5: adaptive centering
    return x - alpha_center * x.mean()


class B7_D5(BaseSubstrate):
    """674 + D5: adaptive centering rate based on exploration velocity.

    M element: centering_rate_tracker — tracks new cells per RATE_WINDOW steps.
    alpha(t) adapts based on exploration activity.
    """

    def __init__(self, n_actions=7, seed=0):
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
        # D5: adaptive centering
        self.alpha_center = ALPHA_BASE
        self._live_snapshot = 0  # live count at start of current window
        self._window_start = 0

    def _update_alpha(self):
        """Update centering rate based on recent exploration velocity."""
        elapsed = self.t - self._window_start
        if elapsed >= RATE_WINDOW:
            new_cells = len(self.live) - self._live_snapshot
            rate = new_cells / max(elapsed, 1) * RATE_WINDOW
            # High rate → faster centering (domain actively changing)
            # Low rate → slower centering (stable domain)
            ratio = min(2.0, rate / max(EXPECTED_RATE, 1.0))
            self.alpha_center = float(np.clip(ALPHA_BASE * ratio, 0.1, 2.0))
            self._live_snapshot = len(self.live)
            self._window_start = self.t

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
        self._update_alpha()
        x = _enc_d5(observation, self.alpha_center)
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
            "alpha_center": self.alpha_center,
        }

    def frozen_elements(self):
        return [
            {"name": "centering_rate_tracker", "class": "M",
             "justification": "alpha_center updated by new_cells/RATE_WINDOW ratio. System-driven."},
            {"name": "edge_count_update", "class": "M",
             "justification": "G updated by transitions. System-driven."},
            {"name": "aliased_set", "class": "M",
             "justification": "Aliased cells grow. System-driven."},
            {"name": "ref_hyperplanes", "class": "M",
             "justification": "Refinement planes. System-derived."},
            {"name": "avgpool16", "class": "U",
             "justification": "16x16 pooling. System doesn't choose."},
            {"name": "H_nav_planes", "class": "U",
             "justification": "k=12 random planes. System doesn't choose."},
            {"name": "rate_window", "class": "U",
             "justification": "RATE_WINDOW=1000. System doesn't choose."},
            {"name": "expected_rate", "class": "U",
             "justification": "EXPECTED_RATE=10. System doesn't choose."},
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
        self.alpha_center = ALPHA_BASE
        self._live_snapshot = 0
        self._window_start = 0

    def on_level_transition(self):
        self._pn = None
        self._pfn = None
        self._px = None

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
elems = B7_D5(n_actions=7, seed=0).frozen_elements()
m_names = [e["name"] for e in elems if e["class"] == "M"]
i_names = [e["name"] for e in elems if e["class"] == "I"]
u_names = [e["name"] for e in elems if e["class"] == "U"]
print(f"  B7_D5: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"  M elements: {m_names}")

# ---- CIFAR before ----
print("\n-- Split-CIFAR-100 (before) --")
cifar = SplitCIFAR100Wrapper(n_images_per_task=N_STEPS_CIFAR)
cifar_sub = B7_D5(n_actions=5, seed=SEED)
cifar_sub.reset(SEED)
t0 = time.time()
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
print(f"  avg_accuracy={avg_acc_before:.4f} elapsed={time.time()-t0:.2f}s")

# ---- Game phases ----
sub = B7_D5(n_actions=7, seed=SEED)
sub.reset(SEED)
all_phase_data = {}

for game in ["LS20", "FT09", "VC33"]:
    print(f"\n-- Phase: {game} --")
    env = _make_env(game)
    n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
    sub._n_actions = n_valid
    obs = env.reset(seed=SEED)
    level = 0
    l1_step = None
    steps = 0
    fresh = True
    checkpoints = []
    obs_seq = []
    t_start = time.time()

    while steps < N_STEPS_GAME and (time.time() - t_start) < 280:
        if obs is None:
            obs = env.reset(seed=SEED)
            sub.on_level_transition()
            fresh = True
            continue
        obs_arr = np.array(obs, dtype=np.float32)
        obs_seq.append(obs_arr)
        action = sub.process(obs_arr)
        obs, reward, done, info = env.step(action % n_valid)
        steps += 1
        if fresh:
            fresh = False
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1_step is None:
                l1_step = steps
            level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=SEED)
            sub.on_level_transition()
            fresh = True
        if steps % CHECKPOINT_EVERY == 0:
            state = sub.get_state()
            checkpoints.append({"step": steps, "G": state["G_size"],
                                 "live": state["live_count"], "alpha": state["alpha_center"]})

    elapsed = time.time() - t_start
    print(f"  steps={steps} elapsed={elapsed:.1f}s l1={l1_step}")
    alphas = [f"{c['alpha']:.2f}" for c in checkpoints]
    print(f"  alpha trajectory: {alphas}")
    all_phase_data[game] = {"l1": l1_step, "checkpoints": checkpoints, "n_obs": len(obs_seq)}

    if obs_seq:
        class _B7(B7_D5):
            def __init__(self): super().__init__(n_actions=7, seed=0)
        r3 = judge.measure_r3_dynamics(_B7, obs_sequence=obs_seq[:2000], n_steps=2000, n_checkpoints=10)
        all_phase_data[game]["r3_score"] = r3.get("r3_dynamic_score")
        print(f"  R3 dynamic: {r3.get('r3_dynamic_score')} profile: {r3.get('dynamics_profile')}")

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

# ---- Summary ----
print("\n" + "=" * 65)
print("B7 CHAIN SUMMARY")
print("=" * 65)
print(f"Static R3: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"CIFAR before: {avg_acc_before}  after: {avg_acc_after}  BWT: {bwt}")
for game in ["LS20", "FT09", "VC33"]:
    d = all_phase_data.get(game, {})
    print(f"{game:<8}: l1={d.get('l1')} R3_dyn={d.get('r3_score')}")
print("=" * 65)
print("STEP 736 DONE")
print("=" * 65)
