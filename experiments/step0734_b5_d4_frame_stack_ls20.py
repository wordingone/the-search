"""
Step 734 (B5): D4 Frame stacking on LS20.

R3 hypothesis: temporal context resolves time-dependent transition ambiguity.
hash(concat(frame_t, frame_{t-1})). 256+256=512 dim input, same K=12.
frame_buffer is an M element (changes every step).

20 seeds, 25s. Compare R3 to 674 baseline (single frame).
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM, K_NAV, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 734 (B5) — D4 FRAME STACKING ON LS20")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 20
PER_SEED_TIME = 25

DIM2 = DIM * 2   # 512 for stacked encoding


class D4_674(BaseSubstrate):
    """674 + D4: frame stacking (2 consecutive frames → 512-dim input).

    Replaces single-frame encoding with 2-frame temporal stack.
    frame_buffer is an M element (changes each step by dynamics).
    H_nav and H_fine use DIM2=512 for stacked input.
    """

    def __init__(self, n_actions=4, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM2).astype(np.float32)   # 512-dim planes
        self.H_fine = rng.randn(K_FINE, DIM2).astype(np.float32)
        self._n_actions = n_actions
        # 674 core
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
        # D4: frame buffer
        self.frame_buffer = np.zeros(DIM, np.float32)  # previous frame encoding

    def _enc_stacked(self, observation):
        """Stack current + previous frame encodings."""
        x_cur = _enc_frame(observation)
        x_stacked = np.concatenate([x_cur, self.frame_buffer])  # 512-dim
        return x_stacked, x_cur

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
        x_stacked, x_cur = self._enc_stacked(observation)
        n = self._node(x_stacked)
        fn = self._hash_fine(x_stacked)
        self.live.add(n)
        self.t += 1

        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(DIM2, np.float64), 0))
            self.C[k] = (s + x_stacked.astype(np.float64), c + 1)
            succ = self.G.get((self._pn, self._pa), {})
            if sum(succ.values()) >= MIN_VISITS_ALIAS and len(succ) >= 2:
                self.aliased.add(self._pn)
            if self._pn in self.aliased and self._pfn is not None:
                df = self.G_fine.setdefault((self._pfn, self._pa), {})
                df[fn] = df.get(fn, 0) + 1

        self._px = x_stacked
        self._cn = n
        self._fn = fn
        self.frame_buffer = x_cur.copy()  # update buffer for next step
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
            "frame_buffer_norm": float(np.linalg.norm(self.frame_buffer)),
        }

    def frozen_elements(self):
        return [
            {"name": "frame_buffer", "class": "M",
             "justification": "D4: previous frame encoding. Updated every step by observed dynamics."},
            {"name": "avgpool16", "class": "U",
             "justification": "16x16 pooling for each frame. System doesn't choose pool size."},
            {"name": "channel_0_only", "class": "U",
             "justification": "First channel only. System doesn't choose."},
            {"name": "mean_centering", "class": "U",
             "justification": "Subtract mean. System doesn't choose."},
            {"name": "H_nav_planes_512", "class": "U",
             "justification": "k=12 random planes for 512-dim stacked input. System doesn't choose."},
            {"name": "H_fine_planes_512", "class": "U",
             "justification": "k=20 planes for 512-dim input. System doesn't choose."},
            {"name": "stack_size_2", "class": "U",
             "justification": "Stack 2 frames. Could be 3 or 4. System doesn't choose."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection. Irreducible."},
            {"name": "argmin_edge_count", "class": "I",
             "justification": "Argmin of outgoing edges. Irreducible."},
            {"name": "fine_graph_priority", "class": "U",
             "justification": "Fine graph at aliased cells. System doesn't choose."},
            {"name": "min_visits_alias", "class": "U",
             "justification": "MIN_VISITS=3. System doesn't choose."},
            {"name": "h_split_threshold", "class": "U",
             "justification": "H_SPLIT=0.05. System doesn't choose."},
            {"name": "refine_every", "class": "U",
             "justification": "REFINE_EVERY=5000. System doesn't choose."},
            {"name": "edge_count_update", "class": "M",
             "justification": "G updated by transitions. System-driven."},
            {"name": "aliased_set", "class": "M",
             "justification": "Aliased cells grow. System-driven."},
            {"name": "ref_hyperplanes", "class": "M",
             "justification": "Refinement planes from frame diffs. System-derived."},
        ]

    def reset(self, seed):
        rng = np.random.RandomState(seed * 1000)
        self.H_nav = rng.randn(K_NAV, DIM2).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM2).astype(np.float32)
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
        self.frame_buffer = np.zeros(DIM, np.float32)

    def on_level_transition(self):
        self._pn = None
        self._pfn = None
        self._px = None
        self.frame_buffer = np.zeros(DIM, np.float32)  # reset buffer on transition

    @property
    def n_actions(self):
        return self._n_actions


def _make_env():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("LS20")


judge = ConstitutionalJudge()

# ---- Static R3 audit ----
print("\n-- Static R3 audit --")
elems = D4_674(n_actions=4, seed=0).frozen_elements()
m_names = [e["name"] for e in elems if e["class"] == "M"]
i_names = [e["name"] for e in elems if e["class"] == "I"]
u_names = [e["name"] for e in elems if e["class"] == "U"]
print(f"  D4_674: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"  M elements: {m_names}")
print(f"  U elements: {u_names}")
print(f"  Compare 674: M=3 I=3 U=9 — D4 adds 1M (frame_buffer)")

# ---- R3 dynamics ----
print("\n-- R3 dynamics (2000 random obs) --")
class _D4(D4_674):
    def __init__(self): super().__init__(n_actions=4, seed=0)
r3_rand = judge.measure_r3_dynamics(_D4, n_steps=2000, n_checkpoints=10)
print(f"  R3 score: {r3_rand.get('r3_dynamic_score')} profile: {r3_rand.get('dynamics_profile')}")
print(f"  Verified M: {r3_rand.get('verified_m_elements', [])}")

# ---- 20 seeds on LS20 ----
print(f"\n-- LS20 {N_SEEDS} seeds x {PER_SEED_TIME}s --")
results = []

for seed_i in range(N_SEEDS):
    seed = SEED_BASE + seed_i * 100
    try:
        env = _make_env()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 4
        sub = D4_674(n_actions=n_valid, seed=seed)
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

        state = sub.get_state()
        results.append({"seed": seed, "l1": l1_step, "steps": steps,
                        "G": state["G_size"], "aliased": state["aliased_count"]})
        status = "L1" if l1_step else "  "
        print(f"  seed={seed:>4} {status} steps={steps:>5} "
              f"G={state['G_size']:>4} aliased={state['aliased_count']:>3}")
    except Exception as e:
        print(f"  seed={seed:>4} ERROR: {e}")
        results.append({"seed": seed, "l1": None, "steps": 0, "error": str(e)})

l1_count = sum(1 for r in results if r.get("l1"))

print("\n" + "=" * 65)
print("B5 SUMMARY")
print("=" * 65)
print(f"D4_674 R3 static: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
verdict = "PASS" if l1_count >= 17 else ("KILL" if l1_count < 14 else "MARGINAL")
print(f"LS20 L1 success: {l1_count}/{N_SEEDS} {verdict}")
print(f"R3 dynamic (random): {r3_rand.get('r3_dynamic_score')}")
print("=" * 65)
print("STEP 734 DONE")
print("=" * 65)
