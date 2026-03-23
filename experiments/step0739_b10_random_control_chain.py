"""
Step 739 (B10): Random attention control — ABLATION.

R3 hypothesis: if random encoding modifications produce same R3 as transition-driven
self-direction, then self-direction doesn't add value.
Control: 674 + random channel weights (uniform [0,1]) + random K per cell (uniform [8,24]).
Compare performance and R3 dynamics against B8 (transition-driven D1+D3).
Success: perf < B8 (self-direction matters). Kill: perf >= B8.

Chain: CIFAR → LS20 → FT09 → VC33 → CIFAR.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM
from substrates.chain import SplitCIFAR100Wrapper
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 739 (B10) — RANDOM ATTENTION CONTROL CHAIN")
print("=" * 65)

SEED = 0
N_STEPS_GAME = 10_000
N_STEPS_CIFAR = 500
CHECKPOINT_EVERY = 1000

K_MIN = 8
K_MAX = 24


def _hash_k(x, H, k):
    bits = (H[:k] @ x > 0).astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


class B10_Random(BaseSubstrate):
    """Control: random channel weights + random K per cell (no adaptation).

    No M elements beyond edge_count_update — ch_weights and K are fixed at init,
    not modified by dynamics. This tests whether self-direction adds value over random.
    """

    def __init__(self, n_actions=7, seed=0):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(K_MAX, DIM).astype(np.float32)
        self._n_actions = n_actions
        self.G = {}
        self.live = set()
        self._pn = None
        self._pa = None
        self.t = 0
        # Random (fixed) channel weights and K distribution
        self.ch_weights = rng.uniform(0, 1, 3).astype(np.float32)
        self.ch_weights /= self.ch_weights.sum()
        # Random K mapping: for each base cell, K is drawn randomly [K_MIN, K_MAX]
        # and fixed at init (not adapted). Use a fixed mapping: hash → K
        self._k_rng = np.random.RandomState(seed + 1)

    def _get_k(self, base_cell):
        # Random K per base cell — NOT adapted, just fixed random
        rng = np.random.RandomState(hash(base_cell) % (2**31))
        return int(rng.choice(range(K_MIN, K_MAX + 1, 2)))

    def _enc_random(self, observation):
        """Random channel-weighted encoding (fixed weights, not transition-driven)."""
        frame = np.array(observation, dtype=np.float32)
        if frame.ndim == 3:
            if frame.shape[0] < frame.shape[1] and frame.shape[0] < frame.shape[2]:
                frame = frame.transpose(1, 2, 0)
            h, w = frame.shape[:2]
            n_ch = frame.shape[2]
            a = np.zeros((h, w), np.float32)
            for c in range(min(n_ch, 3)):
                ch = frame[:, :, c].astype(np.float32)
                if ch.max() > 1:
                    ch = ch / 255.0
                a += self.ch_weights[c] * ch
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

    def process(self, observation):
        x = self._enc_random(observation)
        base = _hash_k(x, self.H, K_MIN)
        k = self._get_k(base)
        cell = _hash_k(x, self.H, k)
        self.live.add(cell)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[cell] = d.get(cell, 0) + 1
        best_a, best_s = 0, float('inf')
        for a in range(self._n_actions):
            s = sum(self.G.get((cell, a), {}).values())
            if s < best_s:
                best_s = s
                best_a = a
        self._pn = cell
        self._pa = best_a
        return best_a

    def get_state(self):
        k_sample = [self._get_k(i) for i in range(10)]
        return {
            "G_size": len(self.G),
            "live_count": len(self.live),
            "t": self.t,
            "ch_weights": self.ch_weights.copy(),
            "k_sample_mean": float(np.mean(k_sample)),
        }

    def frozen_elements(self):
        return [
            {"name": "ch_weights_fixed_random", "class": "U",
             "justification": "Random uniform weights fixed at init. NOT adapted by dynamics."},
            {"name": "k_fixed_random", "class": "U",
             "justification": "Random K per cell fixed at init. NOT adapted by dynamics."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection for cell identity. Irreducible."},
            {"name": "argmin_edge_count", "class": "I",
             "justification": "Argmin of outgoing edges. Irreducible."},
            {"name": "edge_count_update", "class": "M",
             "justification": "G updated by transitions. System-driven."},
        ]

    def reset(self, seed):
        rng = np.random.RandomState(seed * 1000)
        self.H = rng.randn(K_MAX, DIM).astype(np.float32)
        self.ch_weights = rng.uniform(0, 1, 3).astype(np.float32)
        self.ch_weights /= self.ch_weights.sum()
        self.G = {}
        self.live = set()
        self._pn = None
        self._pa = None
        self.t = 0

    def on_level_transition(self):
        self._pn = None
        self._pa = None

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
elems = B10_Random(n_actions=7, seed=0).frozen_elements()
m_names = [e["name"] for e in elems if e["class"] == "M"]
i_names = [e["name"] for e in elems if e["class"] == "I"]
u_names = [e["name"] for e in elems if e["class"] == "U"]
print(f"  B10_Random: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"  M elements: {m_names}")
print(f"  U elements (unjustified): {u_names}")
print(f"  Control: U elements are INTENTIONALLY random — not self-directed")

# ---- CIFAR before ----
print("\n-- Split-CIFAR-100 (before) --")
cifar = SplitCIFAR100Wrapper(n_images_per_task=N_STEPS_CIFAR)
sub = B10_Random(n_actions=5, seed=SEED)
sub.reset(SEED)
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
        correct = sum(int(sub.process(task_images[i].astype(np.float32)/255.0) % 5 == task_labels[i]) for i in idx)
        cifar_acc_before.append(correct / len(idx))
    avg_acc_before = float(np.mean(cifar_acc_before))
    sub.on_level_transition()
print(f"  avg_accuracy={avg_acc_before} elapsed={time.time()-t0:.2f}s")

# ---- Game chain ----
sub_game = B10_Random(n_actions=7, seed=SEED)
sub_game.reset(SEED)
all_phase_data = {}

for game in ["LS20", "FT09", "VC33"]:
    print(f"\n-- Phase: {game} --")
    env = _make_env(game)
    n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
    sub_game._n_actions = n_valid
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
            sub_game.on_level_transition()
            fresh = True
            continue
        obs_arr = np.array(obs, dtype=np.float32)
        obs_seq.append(obs_arr)
        action = sub_game.process(obs_arr)
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
            sub_game.on_level_transition()
        if done:
            obs = env.reset(seed=SEED)
            sub_game.on_level_transition()
            fresh = True
        if steps % CHECKPOINT_EVERY == 0:
            state = sub_game.get_state()
            checkpoints.append({"step": steps, "G": state["G_size"], "live": state["live_count"]})

    elapsed = time.time() - t_start
    print(f"  steps={steps} elapsed={elapsed:.1f}s l1={l1_step}")
    all_phase_data[game] = {"l1": l1_step, "n_obs": len(obs_seq)}

    if obs_seq:
        class _B10(B10_Random):
            def __init__(self): super().__init__(n_actions=7, seed=0)
        r3 = judge.measure_r3_dynamics(_B10, obs_sequence=obs_seq[:2000], n_steps=2000, n_checkpoints=10)
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
        correct = sum(int(sub.process(task_images[i].astype(np.float32)/255.0) % 5 == task_labels[i]) for i in idx)
        cifar_acc_after.append(correct / len(idx))
    avg_acc_after = float(np.mean(cifar_acc_after))
    bwt = avg_acc_after - avg_acc_before
else:
    avg_acc_after = bwt = None
print(f"  avg_accuracy={avg_acc_after} BWT={bwt}")

# ---- Summary ----
print("\n" + "=" * 65)
print("B10 CHAIN SUMMARY (RANDOM CONTROL)")
print("=" * 65)
print(f"Static R3: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"Control note: U elements are random (intentional — ablation)")
print(f"CIFAR before: {avg_acc_before}  after: {avg_acc_after}  BWT: {bwt}")
for game in ["LS20", "FT09", "VC33"]:
    d = all_phase_data.get(game, {})
    print(f"{game:<8}: l1={d.get('l1')} R3_dyn={d.get('r3_score')}")
print("Compare to B8: D1+D3 self-directed LS20 l1=1105, R3=1.0")
print("If B10 LS20 l1 >> B8 LS20 l1: self-direction doesn't matter")
print("If B10 LS20 l1 >> 1105 or R3 same: KILL self-direction")
print("=" * 65)
print("STEP 739 DONE")
print("=" * 65)
