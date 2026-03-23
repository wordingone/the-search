"""
Step 755 - Adaptive REFINE_EVERY from aliasing rate.

R3 hypothesis: REFINE_EVERY (currently 5000, U element) can become M —
derived from aliasing velocity. More aliasing → faster refinement.
REFINE_EVERY = clamp(1000 / aliasing_rate, 1000, 10000).

If aliasing_rate=0.3 (typical 674): period ≈ 3333 (faster than 5000).
If aliasing_rate=0.1 (low): period = 10000 (slower than 5000).

Prediction: equivalent to fixed 5000 on LS20 (aliasing is uniform).
Kill: if worse than fixed 5000 by 3+ seeds → refinement timing is sensitive.
LS20, 10 seeds, 25s.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM, K_NAV, K_FINE, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT

print("=" * 65)
print("STEP 755 - ADAPTIVE REFINE_EVERY FROM ALIASING RATE")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 10
PER_SEED_TIME = 25
REFINE_MIN = 1000
REFINE_MAX = 10000


class Step755_AdaptiveRefine(BaseSubstrate):
    """674 with REFINE_EVERY = clamp(1000 / aliasing_rate, 1000, 10000).

    M element: refine_period (changes with aliasing_rate, which grows from dynamics).
    aliasing_rate = len(aliased) / max(len(live), 1).
    """

    def __init__(self, n_actions=7, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self._n_actions = n_actions
        self.ref = {}; self.G = {}; self.C = {}
        self.live = set(); self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None
        self._pfn = None; self.t = 0; self._cn = self._fn = None
        self._steps_since_refine = 0
        self.refine_period_log = []

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

    def _check_refine(self):
        self._steps_since_refine += 1
        alias_rate = len(self.aliased) / max(len(self.live), 1)
        period = max(REFINE_MIN, min(REFINE_MAX, int(1000 / max(alias_rate, 0.01))))
        if self._steps_since_refine >= period:
            self._refine()
            self._steps_since_refine = 0
            self.refine_period_log.append(period)

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
        self._px = x; self._cn = n; self._fn = fn
        self._check_refine()
        action = self._select()
        self._pn = n; self._pfn = fn; self._pa = action
        return action

    def get_state(self):
        avg_period = float(np.mean(self.refine_period_log)) if self.refine_period_log else None
        return {"G_size": len(self.G), "live_count": len(self.live),
                "aliased_count": len(self.aliased), "t": self.t,
                "ref_count": len(self.ref),
                "avg_refine_period": avg_period,
                "n_refines": len(self.refine_period_log)}

    def frozen_elements(self):
        return [
            {"name": "refine_period", "class": "M", "justification": "Derived from aliasing_rate, which grows from game dynamics."},
            {"name": "edge_count_update", "class": "M", "justification": "G grows by transitions."},
            {"name": "aliased_set", "class": "M", "justification": "Aliased cells grow."},
            {"name": "ref_hyperplanes", "class": "M", "justification": "Refinement planes from passive update."},
            {"name": "REFINE_MIN", "class": "U", "justification": "REFINE_MIN=1000. System doesn't choose."},
            {"name": "REFINE_MAX", "class": "U", "justification": "REFINE_MAX=10000. System doesn't choose."},
            {"name": "binary_hash", "class": "I", "justification": "Sign projection. Irreducible."},
            {"name": "argmin_edge_count", "class": "I", "justification": "Argmin. Irreducible."},
        ]

    def reset(self, seed):
        rng = np.random.RandomState(seed * 1000)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}
        self.live = set(); self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None
        self._pfn = None; self.t = 0; self._cn = self._fn = None
        self._steps_since_refine = 0
        self.refine_period_log = []

    def on_level_transition(self):
        self._pn = None; self._pfn = None; self._px = None

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


print(f"\n-- LS20 {N_SEEDS} seeds x {PER_SEED_TIME}s (adaptive REFINE_EVERY) --")
results = []

for seed_i in range(N_SEEDS):
    seed = SEED_BASE + seed_i * 100
    try:
        env = _make_env()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        sub = Step755_AdaptiveRefine(n_actions=n_valid, seed=seed)
        sub.reset(seed)
        obs = env.reset(seed=seed)
        level = 0; l1_step = None; steps = 0; fresh = True
        t_start = time.time()
        while (time.time() - t_start) < PER_SEED_TIME:
            if obs is None:
                obs = env.reset(seed=seed); sub.on_level_transition(); fresh = True; continue
            obs_arr = np.array(obs, dtype=np.float32)
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
                obs = env.reset(seed=seed); sub.on_level_transition(); fresh = True
        state = sub.get_state()
        results.append({"seed": seed, "l1": l1_step, "steps": steps,
                        "avg_period": state["avg_refine_period"],
                        "n_refines": state["n_refines"]})
        status = "L1" if l1_step else "  "
        print(f"  seed={seed:>4} {status} steps={steps:>5} avg_period={state['avg_refine_period']} "
              f"n_refines={state['n_refines']} ref={state['ref_count']}")
    except Exception as e:
        print(f"  seed={seed:>4} ERROR: {e}")
        import traceback; traceback.print_exc()
        results.append({"seed": seed, "l1": None, "steps": 0})

l1_count = sum(1 for r in results if r.get("l1"))
avg_periods = [r["avg_period"] for r in results if r.get("avg_period") is not None]
verdict = "PASS" if l1_count >= 7 else ("KILL" if l1_count < 5 else "MARGINAL")

print("\n" + "=" * 65)
print("STEP 755 SUMMARY - ADAPTIVE REFINE_EVERY")
print("=" * 65)
print(f"LS20 L1: {l1_count}/{N_SEEDS} {verdict}")
print(f"Avg refine period: {float(np.mean(avg_periods)):.0f} (674 fixed: 5000)")
print(f"If period ≈ 3000-4000: aliasing rate ≈0.25-0.33 (matches observed)")
print(f"Compare 674 (fixed 5000): need fresh 10-seed baseline for comparison")
print("=" * 65)
print("STEP 755 DONE")
print("=" * 65)
