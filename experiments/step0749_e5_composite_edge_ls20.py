"""
Step 749 (E5): Composite edge data (visits + deaths) on LS20.

R3 hypothesis: argmin over composite data E_total = visit_count + λ*death_count
combines exploration + safety. λ=1.0 baseline.
Compare: pure argmin (674) vs composite argmin (E5).
20 seeds, 25s. Success: ≥17/20 AND lower death rate. Kill: <14/20.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM, K_NAV, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT

print("=" * 65)
print("STEP 749 (E5) — COMPOSITE EDGE DATA (VISITS + DEATHS) LS20")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 20
PER_SEED_TIME = 25
LAMBDA = 1.0   # death weight


class E5_Composite(BaseSubstrate):
    """674 + composite E_total = visit_count + λ*death_count for argmin.

    M elements: edge_count_update (G), death_edges (new), aliased_set, ref_hyperplanes.
    Same as D2_Death but with LAMBDA=1.0 (not 5.0) — lighter penalty.
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
        self.death_edges = {}
        self._pending_death = False
        self._total_deaths = 0

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
                v = sum(self.G_fine.get((self._fn, a), {}).values())
                d = self.death_edges.get((self._fn, a), 0)
                if v + LAMBDA * d < best_s:
                    best_s = v + LAMBDA * d
                    best_a = a
            return best_a
        best_a, best_s = 0, float('inf')
        for a in range(self._n_actions):
            v = sum(self.G.get((self._cn, a), {}).values())
            d = self.death_edges.get((self._cn, a), 0)
            if v + LAMBDA * d < best_s:
                best_s = v + LAMBDA * d
                best_a = a
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
            if self._pending_death:
                k = (self._pn, self._pa)
                self.death_edges[k] = self.death_edges.get(k, 0) + 1
                if self._pn in self.aliased and self._pfn is not None:
                    kf = (self._pfn, self._pa)
                    self.death_edges[kf] = self.death_edges.get(kf, 0) + 1
                self._pending_death = False
            else:
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
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        action = self._select()
        self._pn = n; self._pfn = fn; self._pa = action
        return action

    def mark_death(self):
        self._pending_death = True
        self._total_deaths += 1

    def get_state(self):
        return {"G_size": len(self.G), "live_count": len(self.live),
                "aliased_count": len(self.aliased), "t": self.t,
                "total_deaths": self._total_deaths,
                "death_edge_count": len(self.death_edges)}

    def frozen_elements(self):
        return [
            {"name": "death_edges", "class": "M", "justification": "death_edges grow on game-over. System-driven."},
            {"name": "edge_count_update", "class": "M", "justification": "G updated by transitions."},
            {"name": "aliased_set", "class": "M", "justification": "Aliased cells grow."},
            {"name": "ref_hyperplanes", "class": "M", "justification": "Refinement planes."},
            {"name": "lambda", "class": "U", "justification": "LAMBDA=1.0. System doesn't choose."},
            {"name": "binary_hash", "class": "I", "justification": "Sign projection. Irreducible."},
            {"name": "argmin_E_total", "class": "I", "justification": "Argmin composite. Irreducible."},
        ]

    def reset(self, seed):
        rng = np.random.RandomState(seed * 1000)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}
        self.live = set(); self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None
        self._pfn = None; self.t = 0; self._cn = self._fn = None
        self.death_edges = {}; self._pending_death = False; self._total_deaths = 0

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


print(f"\n-- LS20 {N_SEEDS} seeds x {PER_SEED_TIME}s --")
results = []

for seed_i in range(N_SEEDS):
    seed = SEED_BASE + seed_i * 100
    try:
        env = _make_env()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        sub = E5_Composite(n_actions=n_valid, seed=seed)
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
                sub.mark_death()
                obs = env.reset(seed=seed); sub.on_level_transition(); fresh = True
        state = sub.get_state()
        results.append({"seed": seed, "l1": l1_step, "steps": steps,
                        "deaths": state["total_deaths"]})
        status = "L1" if l1_step else "  "
        print(f"  seed={seed:>4} {status} steps={steps:>5} deaths={state['total_deaths']:>3}")
    except Exception as e:
        print(f"  seed={seed:>4} ERROR: {e}")
        results.append({"seed": seed, "l1": None, "steps": 0})

l1_count = sum(1 for r in results if r.get("l1"))
avg_deaths = np.mean([r["deaths"] for r in results if "deaths" in r])
verdict = "PASS" if l1_count >= 17 else ("KILL" if l1_count < 14 else "MARGINAL")

print("\n" + "=" * 65)
print("E5 SUMMARY (COMPOSITE EDGE DATA)")
print("=" * 65)
print(f"LS20 L1: {l1_count}/{N_SEEDS} {verdict}")
print(f"Avg deaths: {avg_deaths:.1f}")
print("=" * 65)
print("STEP 749 DONE")
print("=" * 65)
