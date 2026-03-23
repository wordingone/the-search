"""
Step 740 (D1): Multi-episode graph retention on LS20 L2.

R3 hypothesis: full graph persistence (including aliased status) across game-overs
helps reach L2. 674 normally resets aliased cells on game-over (on_level_transition
clears _pn but not the aliased set). This experiment: NEVER reset aliased set.

Note: Jun directive says 300s budget requires explicit approval.
Using 60s (12x normal 5s) as intermediate — if L2 is reached at all, report.
5 seeds, 60s each.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM, K_NAV, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT

print("=" * 65)
print("STEP 740 (D1) — MULTI-EPISODE GRAPH RETENTION LS20 L2")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 5
PER_SEED_TIME = 60   # 60s per seed (within 5-min cap: 5x60=5min)


class D1_Retain(BaseSubstrate):
    """674 with persistent aliased set across game-overs.

    M element: aliased_set — persists across episodes. In 674, this effectively
    resets because _pn=None breaks the transition update chain. Here we preserve
    full graph structure including aliased status across game-overs.
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
        x = _enc_frame(observation)
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
        }

    def frozen_elements(self):
        return [
            {"name": "edge_count_update", "class": "M",
             "justification": "G updated by transitions. System-driven."},
            {"name": "aliased_set", "class": "M",
             "justification": "Aliased cells persist across episodes (D1 modification). System-driven."},
            {"name": "ref_hyperplanes", "class": "M",
             "justification": "Refinement planes. System-derived."},
            {"name": "avgpool16", "class": "U",
             "justification": "16x16 pooling. System doesn't choose."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection. Irreducible."},
            {"name": "argmin_edge_count", "class": "I",
             "justification": "Argmin. Irreducible."},
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

    def on_level_transition(self):
        # KEY: only reset pointers, NOT aliased set (retention across episodes)
        self._pn = None
        self._pfn = None
        self._px = None

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


print(f"\n-- LS20 {N_SEEDS} seeds x {PER_SEED_TIME}s --")
results = []

for seed_i in range(N_SEEDS):
    seed = SEED_BASE + seed_i * 100
    try:
        env = _make_env()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        sub = D1_Retain(n_actions=n_valid, seed=seed)
        sub.reset(seed)
        obs = env.reset(seed=seed)
        level = 0
        l1_step = l2_step = None
        steps = 0
        fresh = True
        t_start = time.time()
        aliased_log = []

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
                if cl == 2 and l2_step is None:
                    l2_step = steps
                level = cl
                sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed)
                sub.on_level_transition()
                fresh = True
            if steps % 10000 == 0:
                state = sub.get_state()
                aliased_log.append((steps, state["aliased_count"]))

        state = sub.get_state()
        results.append({"seed": seed, "l1": l1_step, "l2": l2_step, "steps": steps,
                        "G": state["G_size"], "aliased": state["aliased_count"]})
        status = f"L{level}" if level > 0 else "  "
        print(f"  seed={seed:>4} {status} steps={steps:>6} G={state['G_size']:>5} "
              f"aliased={state['aliased_count']:>3} l1={l1_step} l2={l2_step}")
    except Exception as e:
        print(f"  seed={seed:>4} ERROR: {e}")
        results.append({"seed": seed, "l1": None, "l2": None, "steps": 0, "error": str(e)})

l1_count = sum(1 for r in results if r.get("l1"))
l2_count = sum(1 for r in results if r.get("l2"))
verdict = "PASS" if l1_count >= 4 else ("KILL" if l1_count < 3 else "MARGINAL")

print("\n" + "=" * 65)
print("D1 SUMMARY (MULTI-EPISODE RETENTION)")
print("=" * 65)
print(f"LS20 L1: {l1_count}/{N_SEEDS} {verdict}")
print(f"LS20 L2: {l2_count}/{N_SEEDS}")
print(f"Note: 674 baseline at 60s would be ~{N_SEEDS}/5 L1 (all seeds reach L1 by 25s)")
print("Key question: does retention help reach L2?")
print("=" * 65)
print("STEP 740 DONE")
print("=" * 65)
