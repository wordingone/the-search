"""
Step 730 (B1): D1 channel-weighted encoding on LS20.

R3 hypothesis: per-channel transition variance selects task-relevant channels.
channel_weights is an M element (modified by dynamics — variance of transitions).
Replaces the 'channel_0_only' U element in 674 with a system-driven M element.

674 + adaptive channel weights w_c = Var(transitions_c) / max(Var).
For ARC games (1 channel): weights are scalar, but M element still tracked.
For CIFAR/Atari (3ch): meaningful differentiation.

20 seeds, 25s each. Report R3 audit + dynamic score.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import DIM, K_NAV, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 730 (B1) — D1 CHANNEL-WEIGHTED ENCODING ON LS20")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 20
PER_SEED_TIME = 25
N_STEPS = 100_000   # step limit (time is the real cap)


def _enc_d1(frame, ch_weights):
    """D1 encoding: weighted channel combination instead of channel_0_only."""
    frame = np.array(frame, dtype=np.float32)
    if frame.ndim == 3:
        if frame.shape[0] <= 4 and frame.shape[1] > 4:
            frame = frame.transpose(1, 2, 0)
        h, w = frame.shape[:2]
        n_ch = frame.shape[2]
        if n_ch == 1:
            a = frame[:, :, 0]
            if a.max() > 1:
                a = a / 15.0
        else:
            a = np.zeros((h, w), np.float32)
            w_sum = ch_weights[:n_ch].sum()
            for c in range(min(n_ch, 3)):
                ch = frame[:, :, c].astype(np.float32)
                if ch.max() > 1:
                    ch = ch / 255.0
                a += ch_weights[c] * ch
            if w_sum > 1e-8:
                a /= w_sum
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


class D1_674(BaseSubstrate):
    """674 bootloader with D1: adaptive channel weights as M element."""

    def __init__(self, n_actions=4, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self._n_actions = n_actions
        # 674 state
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
        # D1 state
        self.ch_weights = np.array([1.0, 0.0, 0.0], np.float32)  # channel_0 default
        self.ch_var_ema = np.ones(3, np.float32) * 0.01
        self.last_frame = None
        self.n_d1_obs = 0

    def _update_d1(self, frame):
        frame = np.array(frame, dtype=np.float32)
        if frame.ndim == 3:
            if frame.shape[0] <= 4 and frame.shape[1] > 4:
                frame = frame.transpose(1, 2, 0)
            n_ch = frame.shape[2]
            if self.last_frame is not None and self.last_frame.shape == frame.shape:
                self.n_d1_obs += 1
                alpha = min(0.05, 1.0 / max(self.n_d1_obs, 1))
                for c in range(min(n_ch, 3)):
                    diff = frame[:, :, c].astype(np.float32) - self.last_frame[:, :, c].astype(np.float32)
                    self.ch_var_ema[c] = (1-alpha)*self.ch_var_ema[c] + alpha*float(np.var(diff))
                total = self.ch_var_ema[:n_ch].sum()
                if total > 1e-8 and self.n_d1_obs >= 50:
                    self.ch_weights[:n_ch] = self.ch_var_ema[:n_ch] / total
            self.last_frame = frame.copy()

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
        self._update_d1(observation)
        x = _enc_d1(observation, self.ch_weights)
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
            "ch_var_ema": self.ch_var_ema.copy(),
            "n_d1_obs": self.n_d1_obs,
        }

    def frozen_elements(self):
        return [
            {"name": "avgpool16", "class": "U",
             "justification": "16x16 average pooling. System doesn't choose pool size."},
            {"name": "channel_weights", "class": "M",
             "justification": "D1: per-channel variance weights. Updated by dynamics. Replaces channel_0_only."},
            {"name": "mean_centering", "class": "U",
             "justification": "Subtract mean. System doesn't choose normalization."},
            {"name": "H_nav_planes", "class": "U",
             "justification": "k=12 random LSH planes. System doesn't choose count or direction."},
            {"name": "H_fine_planes", "class": "U",
             "justification": "k=20 random planes for aliased cells. System doesn't choose."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection. Irreducible."},
            {"name": "argmin_edge_count", "class": "I",
             "justification": "Argmin of outgoing edges. Irreducible."},
            {"name": "fine_graph_priority", "class": "U",
             "justification": "Use fine graph at aliased cells. System doesn't choose."},
            {"name": "min_visits_alias", "class": "U",
             "justification": "MIN_VISITS=3. System doesn't choose threshold."},
            {"name": "h_split_threshold", "class": "U",
             "justification": "H_SPLIT=0.05. System doesn't choose."},
            {"name": "refine_every", "class": "U",
             "justification": "REFINE_EVERY=5000. System doesn't choose."},
            {"name": "edge_count_update", "class": "M",
             "justification": "G updated by each transition. System-driven."},
            {"name": "aliased_set", "class": "M",
             "justification": "Aliased cells grow with ambiguous transitions. System-driven."},
            {"name": "ref_hyperplanes", "class": "M",
             "justification": "Refinement planes derived from frame diffs. System-derived."},
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
        self.ch_weights = np.array([1.0, 0.0, 0.0], np.float32)
        self.ch_var_ema = np.ones(3, np.float32) * 0.01
        self.last_frame = None
        self.n_d1_obs = 0

    def on_level_transition(self):
        self._pn = None
        self._pfn = None
        self._px = None

    @property
    def n_actions(self):
        return self._n_actions


def _make_env(game="LS20"):
    try:
        import arcagi3
        return arcagi3.make(game)
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game)


judge = ConstitutionalJudge()

# ---- Static R3 audit ----
print("\n-- Static R3 audit --")
elems = D1_674(n_actions=4, seed=0).frozen_elements()
m_count = sum(1 for e in elems if e["class"] == "M")
i_count = sum(1 for e in elems if e["class"] == "I")
u_count = sum(1 for e in elems if e["class"] == "U")
m_names = [e["name"] for e in elems if e["class"] == "M"]
u_names = [e["name"] for e in elems if e["class"] == "U"]
print(f"  D1_674: M={m_count} I={i_count} U={u_count}")
print(f"  M elements: {m_names}")
print(f"  U elements: {u_names}")
print(f"  Compare 674: M=3 I=3 U=9 → D1: M={m_count} I={i_count} U={u_count}")

# ---- R3 dynamics (random obs) ----
print("\n-- R3 dynamics (2000 random obs) --")
class _D1(D1_674):
    def __init__(self): super().__init__(n_actions=4, seed=0)
r3_rand = judge.measure_r3_dynamics(_D1, n_steps=2000, n_checkpoints=10)
print(f"  R3 score: {r3_rand.get('r3_dynamic_score')} profile: {r3_rand.get('dynamics_profile')}")
print(f"  Verified M: {r3_rand.get('verified_m_elements', [])}")

# ---- 20 seeds on LS20 ----
print(f"\n-- LS20 {N_SEEDS} seeds x {PER_SEED_TIME}s --")
results = []
obs_seq_all = []

for seed_i in range(N_SEEDS):
    seed = SEED_BASE + seed_i * 100
    try:
        env = _make_env("LS20")
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 4
        sub = D1_674(n_actions=n_valid, seed=seed)
        sub.reset(seed)
        obs = env.reset(seed=seed)
        level = 0
        l1_step = None
        steps = 0
        fresh = True
        t_start = time.time()

        while (time.time() - t_start) < PER_SEED_TIME:
            if obs is None:
                obs = env.reset(seed=seed)
                sub.on_level_transition()
                fresh = True
                continue
            obs_arr = np.array(obs, dtype=np.float32)
            if seed_i == 0:
                obs_seq_all.append(obs_arr)
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
                obs = env.reset(seed=seed)
                sub.on_level_transition()
                fresh = True

        final = sub.get_state()
        results.append({
            "seed": seed, "l1": l1_step, "steps": steps,
            "G_size": final["G_size"], "aliased": final["aliased_count"],
            "ch_weights": final["ch_weights"].tolist(),
            "n_d1_obs": final["n_d1_obs"],
        })
        status = "L1" if l1_step else "  "
        print(f"  seed={seed:>4} {status} steps={steps:>5} "
              f"G={final['G_size']:>4} aliased={final['aliased_count']:>3} "
              f"chW={[round(w,3) for w in final['ch_weights'].tolist()]}")
    except Exception as e:
        print(f"  seed={seed:>4} ERROR: {e}")
        results.append({"seed": seed, "l1": None, "steps": 0, "error": str(e)})

l1_count = sum(1 for r in results if r.get("l1"))
print(f"\n  L1 success: {l1_count}/{N_SEEDS}")

# ---- R3 dynamics on real obs ----
if obs_seq_all:
    print("\n-- R3 dynamics (real LS20 obs, seed 0) --")
    r3_real = judge.measure_r3_dynamics(_D1, obs_sequence=obs_seq_all[:2000],
                                        n_steps=2000, n_checkpoints=10)
    print(f"  R3 score: {r3_real.get('r3_dynamic_score')} profile: {r3_real.get('dynamics_profile')}")
    print(f"  Verified M: {r3_real.get('verified_m_elements', [])}")

print("\n" + "=" * 65)
print("B1 SUMMARY")
print("=" * 65)
print(f"D1_674 R3 static: M={m_count} I={i_count} U={u_count}")
print(f"Compare 674: M=3 I=3 U=9 — D1 adds 1M, removes 1U (channel_0_only)")
print(f"LS20 L1 success: {l1_count}/{N_SEEDS} {'PASS' if l1_count>=17 else 'KILL' if l1_count<14 else 'MARGINAL'}")
print(f"R3 dynamic score: {r3_rand.get('r3_dynamic_score')} (random obs)")
if obs_seq_all:
    print(f"R3 dynamic score: {r3_real.get('r3_dynamic_score')} (real LS20 obs)")
print("=" * 65)
print("STEP 730 DONE")
print("=" * 65)
