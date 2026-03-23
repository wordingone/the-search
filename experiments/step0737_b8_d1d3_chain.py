"""
Step 737 (B8): D1+D3 combined substrate — KEY Proposition 18 test.

R3 hypothesis (Prop 18): channel weights (D1) + continuous K per cell (D3)
together produce strongest self-directed attention. R3 dynamics non-zero at
phase transitions — the encoding structure changes as game phases change.

D1: per-channel transition variance → adaptive channel weights (M element).
D3: K(n) grows with cell inconsistency (distinct successors) → finer hashing
for ambiguous cells. inconsistency_map is an M element.

Chain: CIFAR → LS20 → FT09 → VC33 → CIFAR.
Measure: R3 dynamics at each phase transition. CIFAR NMI. R3 audit.
Success: R3 > 0 at transitions (Prop 18 confirmed).
Kill: R3 = 0 throughout (Prop 18 falsified).
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM, K_NAV, REFINE_EVERY, MIN_VISITS_ALIAS
from substrates.judge import ConstitutionalJudge
from substrates.chain import SplitCIFAR100Wrapper

print("=" * 65)
print("STEP 737 (B8) — D1+D3 COMBINED CHAIN (PROP 18 KEY TEST)")
print("=" * 65)

SEED = 0
N_STEPS_GAME = 10_000
N_STEPS_CIFAR = 500
CHECKPOINT_EVERY = 1000

# ---- D3 params ----
K_MIN = 8
K_MAX = 24
I_STEP = 1   # 1 distinct successor → +2 bits K


def _enc_d1(frame, ch_weights):
    """D1: channel-weighted encoding. For 1-ch frames: same as _enc_frame.
    For 3-ch frames: weighted sum of channels instead of channel_0_only."""
    frame = np.array(frame, dtype=np.float32)
    if frame.ndim == 3:
        if frame.shape[0] < frame.shape[1] and frame.shape[0] < frame.shape[2]:
            frame = frame.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
        h, w = frame.shape[:2]
        n_ch = frame.shape[2]
        if n_ch == 1:
            a = frame[:, :, 0]
            if a.max() > 1:
                a = a / 15.0
        else:
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
        # Avgpool to 16x16
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


def _hash_k(x, H, k):
    """First k bits of hash from H planes."""
    bits = (H[:k] @ x > 0).astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


class B8_D1D3(BaseSubstrate):
    """D1 (channel weights) + D3 (adaptive K) combined substrate.

    D1 M element: channel_weights — updated via EMA of per-channel frame variance.
    D3 M element: inconsistency_map — K per cell grows with # distinct successors.
    """

    def __init__(self, n_actions=4, seed=0):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(K_MAX, DIM).astype(np.float32)  # shared K_MAX planes
        self._n_actions = n_actions

        # D3: adaptive K state
        self.G = {}              # (cell, action) -> {successor: count}
        self.live = set()
        self.inconsistency = {}  # base_cell -> max distinct successors
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
        """Update D1 channel weights via EMA of per-channel transition variance."""
        frame = np.array(frame, dtype=np.float32)
        if frame.ndim != 3:
            return
        # Convert (C,H,W) -> (H,W,C): check if first dim < spatial dims
        if frame.shape[0] < frame.shape[1] and frame.shape[0] < frame.shape[2]:
            frame = frame.transpose(1, 2, 0)
        n_ch = min(frame.shape[2], 3)

        if self.last_frame is not None:
            self.n_frames += 1
            alpha = min(0.05, 1.0 / max(self.n_frames, 1))
            # Cap n_ch by last_frame's channels too (games may vary frame shape)
            lf = self.last_frame
            if lf.ndim == 3:
                n_ch_compare = min(n_ch, lf.shape[2])
            else:
                n_ch_compare = 0
            for c in range(n_ch_compare):
                f_c = frame[:, :, c].astype(np.float32)
                l_c = lf[:, :, c].astype(np.float32)
                diff_var = float(np.var(f_c - l_c))
                self.ch_var_ema[c] = (1 - alpha) * self.ch_var_ema[c] + alpha * diff_var
            total = self.ch_var_ema[:n_ch].sum()
            if total > 1e-8 and self.n_frames >= 50:
                self.ch_weights[:n_ch] = self.ch_var_ema[:n_ch] / total

        # Store frame for next step (keep as-is, not transposed)
        self.last_frame = frame

    def _get_k(self, base_cell):
        inc = self.inconsistency.get(base_cell, 0)
        return min(K_MIN + 2 * (inc // max(I_STEP, 1)), K_MAX)

    def process(self, observation):
        obs_arr = np.array(observation, dtype=np.float32)
        self._update_ch_weights(obs_arr)
        x = _enc_d1(obs_arr, self.ch_weights)

        base = _hash_k(x, self.H, K_MIN)  # stable 8-bit base cell
        k = self._get_k(base)
        cell = _hash_k(x, self.H, k)      # adaptive K-bit cell

        self.live.add(cell)
        self.t += 1

        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[cell] = d.get(cell, 0) + 1
            # Update inconsistency for prev base cell
            if self._pbase is not None:
                distinct = max(
                    len(self.G.get((self._pn, a), {}))
                    for a in range(self._n_actions)
                )
                self.inconsistency[self._pbase] = max(
                    self.inconsistency.get(self._pbase, 0), distinct
                )

        # Argmin
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

    def get_state(self):
        k_vals = [self._get_k(b) for b in self.inconsistency] or [K_MIN]
        return {
            "G_size": len(self.G),
            "live_count": len(self.live),
            "t": self.t,
            "n_base_cells": len(self.inconsistency),
            "avg_k": float(np.mean(k_vals)),
            "max_k": float(np.max(k_vals)),
            "ch_weights": self.ch_weights.copy(),
            "ch_var_ema": self.ch_var_ema.copy(),
            "inconsistency_map_size": len(self.inconsistency),
        }

    def frozen_elements(self):
        return [
            {"name": "H_24planes", "class": "U",
             "justification": "24 random LSH planes. System doesn't choose count or direction."},
            {"name": "K_MIN", "class": "U",
             "justification": "K_MIN=8. System doesn't choose minimum resolution."},
            {"name": "I_STEP", "class": "U",
             "justification": "I_STEP=1. System doesn't choose K increment threshold."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection for cell identity. Irreducible — removing destroys graph."},
            {"name": "argmin_edge_count", "class": "I",
             "justification": "Argmin of outgoing edges. Irreducible — removing causes random walk."},
            {"name": "edge_count_update", "class": "M",
             "justification": "G updated by each (cell, action, successor) transition. System-driven."},
            {"name": "inconsistency_map", "class": "M",
             "justification": "I(n) grows with distinct successors. Determines K per cell. System-driven."},
            {"name": "channel_weights", "class": "M",
             "justification": "ch_weights updated via EMA of per-channel transition variance. System-driven."},
        ]

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
        self._pa = None

    @property
    def n_actions(self):
        return self._n_actions


judge = ConstitutionalJudge()


def _make_env(game):
    try:
        import arcagi3
        return arcagi3.make(game)
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game)


def run_game_phase(game, sub, n_steps, checkpoint_every=CHECKPOINT_EVERY):
    """Run one game phase, return (l1_step, checkpoints, obs_seq)."""
    env = _make_env(game)
    # FIX 2026-03-23b: use len(env._action_space) = 7 (ACTION1..ACTION7, RESET excluded)
    n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
    sub._n_actions = n_valid

    obs = env.reset(seed=SEED)
    level = 0
    l1_step = l2_step = None
    steps = 0
    fresh = True
    checkpoints = []
    obs_seq = []
    t_start = time.time()

    while steps < n_steps and (time.time() - t_start) < 280:
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
            if cl == 2 and l2_step is None:
                l2_step = steps
            level = cl
            sub.on_level_transition()

        if done:
            obs = env.reset(seed=SEED)
            sub.on_level_transition()
            fresh = True

        if steps % checkpoint_every == 0:
            state = sub.get_state()
            checkpoints.append({
                "step": steps,
                "G_size": state["G_size"],
                "live": state["live_count"],
                "n_base_cells": state["n_base_cells"],
                "avg_k": round(state["avg_k"], 2),
                "max_k": round(state["max_k"], 2),
                "ch_weights": state["ch_weights"].tolist(),
            })

    elapsed = time.time() - t_start
    print(f"  steps={steps} elapsed={elapsed:.1f}s l1={l1_step} l2={l2_step}")
    print(f"    Step |  Live |   G  | BaseC | AvgK | MaxK | ChWeights")
    for c in checkpoints:
        w = [f"{v:.2f}" for v in c["ch_weights"]]
        print(f"    {c['step']:>4} | {c['live']:>5} | {c['G_size']:>4} | "
              f"{c['n_base_cells']:>5} | {c['avg_k']:>4.1f} | "
              f"{c['max_k']:>4.0f} | {w}")
    return l1_step, checkpoints, obs_seq


# ---- Phase 1: CIFAR (before) ----
print("\n-- Phase 1: Split-CIFAR-100 (before) --")
cifar = SplitCIFAR100Wrapper(n_images_per_task=N_STEPS_CIFAR)
cifar_sub = B8_D1D3(n_actions=4, seed=SEED)
cifar_sub.reset(SEED)
t0 = time.time()
if not cifar._load():
    print("  CIFAR not available")
    avg_acc_before = None
else:
    cifar_acc_before = []
    rng = np.random.RandomState(SEED)
    for task_idx in range(20):
        task_images, task_labels = cifar._data[task_idx]
        idx = rng.choice(len(task_images), min(N_STEPS_CIFAR, len(task_images)), replace=False)
        correct = 0
        for i in idx:
            obs = task_images[i].astype(np.float32) / 255.0
            action = cifar_sub.process(obs) % 5
            correct += int(action == task_labels[i])
        acc = correct / len(idx)
        cifar_acc_before.append(acc)
    cifar_sub.on_level_transition()
    avg_acc_before = float(np.mean(cifar_acc_before))
print(f"  avg_accuracy={avg_acc_before:.4f} tasks=20 elapsed={time.time()-t0:.2f}s")

# ---- Phase 2: Game chain ----
sub = B8_D1D3(n_actions=4, seed=SEED)
sub.reset(SEED)
all_phase_data = {}

for game in ["LS20", "FT09", "VC33"]:
    print(f"\n-- Phase: {game} --")
    l1, ckpts, obs_seq = run_game_phase(game, sub, N_STEPS_GAME)
    all_phase_data[game] = {"l1": l1, "checkpoints": ckpts, "n_obs": len(obs_seq)}

    # R3 dynamics on collected obs
    if obs_seq:
        class _B8(B8_D1D3):
            def __init__(self): super().__init__(n_actions=4, seed=0)
        r3_dyn = judge.measure_r3_dynamics(
            _B8, obs_sequence=obs_seq[:2000], n_steps=2000, n_checkpoints=10)
        all_phase_data[game]["r3_score"] = r3_dyn.get("r3_dynamic_score")
        all_phase_data[game]["r3_profile"] = r3_dyn.get("dynamics_profile")
        all_phase_data[game]["r3_verified_M"] = r3_dyn.get("verified_m_elements", [])
        print(f"  R3 dynamic score: {r3_dyn.get('r3_dynamic_score')} "
              f"profile: {r3_dyn.get('dynamics_profile')}")
        print(f"  Verified M: {r3_dyn.get('verified_m_elements', [])}")

# ---- Phase 3: CIFAR (after) ----
print("\n-- Phase 4: Split-CIFAR-100 (after) --")
if avg_acc_before is None:
    avg_acc_after = None
    bwt = None
else:
    cifar_acc_after = []
    rng2 = np.random.RandomState(SEED + 1)
    for task_idx in range(20):
        task_images, task_labels = cifar._data[task_idx]
        idx = rng2.choice(len(task_images), min(N_STEPS_CIFAR, len(task_images)), replace=False)
        correct = 0
        for i in idx:
            obs = task_images[i].astype(np.float32) / 255.0
            action = cifar_sub.process(obs) % 5
            correct += int(action == task_labels[i])
        cifar_acc_after.append(correct / len(idx))
    avg_acc_after = float(np.mean(cifar_acc_after))
    bwt = avg_acc_after - avg_acc_before
print(f"  avg_accuracy={avg_acc_after} elapsed={time.time()-t0:.2f}s")
print(f"  CIFAR backward transfer (after chain): {bwt}")

# ---- Static R3 audit ----
print("\n-- Static R3 audit --")
elems = B8_D1D3(n_actions=4, seed=0).frozen_elements()
m_names = [e["name"] for e in elems if e["class"] == "M"]
i_names = [e["name"] for e in elems if e["class"] == "I"]
u_names = [e["name"] for e in elems if e["class"] == "U"]
print(f"  M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"  M elements: {m_names}")
print(f"  U elements: {u_names}")
r3_verdict = "R3_PASS" if len(u_names) == 0 else f"R3_FAIL ({len(u_names)} U elements)"
print(f"  Verdict: {r3_verdict}")

# ---- Summary ----
print("\n" + "=" * 65)
print("B8 CHAIN SUMMARY")
print("=" * 65)
print(f"Static R3: M={len(m_names)} I={len(i_names)} U={len(u_names)} verdict={r3_verdict}")
print(f"CIFAR before: avg_acc={avg_acc_before}")
for game in ["LS20", "FT09", "VC33"]:
    d = all_phase_data[game]
    r3 = d.get("r3_score")
    prof = d.get("r3_profile", "N/A")
    vm = d.get("r3_verified_M", [])
    print(f"{game:<8}: l1={d['l1']} R3_dyn={r3} profile={prof} verified_M={vm}")
print(f"CIFAR after:  avg_acc={avg_acc_after} BWT={bwt}")

# ---- Prop 18 verdict ----
r3_scores = [all_phase_data[g].get("r3_score") for g in ["LS20", "FT09", "VC33"]
             if all_phase_data[g].get("r3_score") is not None]
max_r3 = max(r3_scores) if r3_scores else 0
if max_r3 > 0:
    print("\nPROP 18: CONFIRMED — R3 > 0 at phase transitions")
else:
    print("\nPROP 18: FALSIFIED — R3 = 0 throughout")
print("=" * 65)
print("STEP 737 DONE")
print("=" * 65)
