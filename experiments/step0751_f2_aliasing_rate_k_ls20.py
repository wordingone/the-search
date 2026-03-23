"""
Step 751 (F2): K grows from aliasing rate — correct signal for hash resolution.

R3 hypothesis: K_current (M element) grows when aliasing rate exceeds threshold.
Aliasing = cell has >=2 distinct successors with >= MIN_VISITS_ALIAS total visits.
High aliasing rate means K is too small (hash collisions). Adding a plane reduces
collisions, lowering aliasing rate. K stabilizes when aliasing rate < ALIAS_THRESH.

F1 (step750) failure: entropy-based growth cleared G on each step, causing entropy
to stay high from fresh sparsity rather than hash insufficiency. Entropy measures
exploration sparsity, not collision. Aliasing measures collision directly.

No G clearing: when K grows, new observations get longer hash IDs (prefix property).
Old orphaned entries in G just become stale. K grows until alias_rate < ALIAS_THRESH.

If K stabilizes at 10-14: substrate discovered K_NAV=12 is the game's resolution.
20 seeds, 25s. Compare to 674 (K_NAV=12 fixed).
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT

print("=" * 65)
print("STEP 751 (F2) — K GROWS FROM ALIASING RATE")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 20
PER_SEED_TIME = 25
K_START = 4        # start with 4 planes
K_MAX = 24         # ceiling
CHECK_EVERY = 2000  # check aliasing rate every N steps
ALIAS_THRESH = 0.3  # if aliased/live > 0.3, grow K


class F2_AliasingRateK(BaseSubstrate):
    """674 with K_NAV growing from aliasing rate instead of entropy.

    When aliased_count / live_count > ALIAS_THRESH, add a random plane.
    No G clearing on growth — old entries become stale naturally.
    New observations automatically use the longer hash (prefix property).

    K_current is M: grows from aliasing rate, which is driven by game dynamics.
    K stabilizes when aliasing rate drops below ALIAS_THRESH.

    M elements: edge_count_update (G), aliased_set, ref_hyperplanes, K_current.
    U elements: K_START=4 (floor), K_MAX=24 (ceiling), ALIAS_THRESH=0.3.
    I elements: binary_hash, argmin_edge_count, fine_graph_priority.
    """

    def __init__(self, n_actions=7, seed=0):
        rng = np.random.RandomState(seed)
        self._rng = rng
        self.H_nav = rng.randn(K_START, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self._n_actions = n_actions
        self.ref = {}; self.G = {}; self.C = {}
        self.live = set(); self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None
        self._pfn = None; self.t = 0; self._cn = self._fn = None
        self.k_history = [K_START]
        self.alias_rate_log = []

    def _hash_nav(self, x):
        bits = (self.H_nav @ x > 0).astype(np.uint8)
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val

    def _hash_fine(self, x):
        return int(np.packbits((self.H_fine @ x > 0).astype(np.uint8),
                               bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_nav(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _maybe_grow(self):
        """Grow K if aliasing rate is too high. NO G clearing."""
        if len(self.H_nav) >= K_MAX:
            return
        if not self.live:
            return
        alias_rate = len(self.aliased) / max(len(self.live), 1)
        self.alias_rate_log.append((self.t, len(self.H_nav), alias_rate))
        if alias_rate > ALIAS_THRESH:
            new_plane = self._rng.randn(1, DIM).astype(np.float32)
            nm = np.linalg.norm(new_plane)
            if nm > 1e-8:
                self.H_nav = np.vstack([self.H_nav, new_plane / nm])
                self.k_history.append(len(self.H_nav))
                # NO G clearing — old entries become stale, new observations
                # use longer hash IDs (prefix property of binary LSH)

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
        self._px = x; self._cn = n; self._fn = fn
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        if self.t > 0 and self.t % CHECK_EVERY == 0:
            self._maybe_grow()
        action = self._select()
        self._pn = n; self._pfn = fn; self._pa = action
        return action

    def get_state(self):
        return {"G_size": len(self.G), "live_count": len(self.live),
                "aliased_count": len(self.aliased), "t": self.t,
                "K_current": len(self.H_nav), "k_history": self.k_history[:],
                "alias_rate_log": self.alias_rate_log[-5:]}

    def frozen_elements(self):
        return [
            {"name": "K_current", "class": "M", "justification": "K grows from aliasing rate. System-driven."},
            {"name": "edge_count_update", "class": "M", "justification": "G grows by transitions."},
            {"name": "aliased_set", "class": "M", "justification": "Aliased cells grow."},
            {"name": "ref_hyperplanes", "class": "M", "justification": "Refinement planes from passive update."},
            {"name": "K_START", "class": "U", "justification": "K_START=4. System doesn't choose."},
            {"name": "K_MAX", "class": "U", "justification": "K_MAX=24. System doesn't choose."},
            {"name": "ALIAS_THRESH", "class": "U", "justification": "ALIAS_THRESH=0.3. System doesn't choose."},
            {"name": "binary_hash", "class": "I", "justification": "Sign projection. Irreducible."},
            {"name": "argmin_edge_count", "class": "I", "justification": "Argmin. Irreducible."},
            {"name": "fine_graph_priority", "class": "I", "justification": "Fine graph at aliased. Irreducible."},
        ]

    def reset(self, seed):
        rng = np.random.RandomState(seed * 1000)
        self._rng = rng
        self.H_nav = rng.randn(K_START, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}
        self.live = set(); self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None
        self._pfn = None; self.t = 0; self._cn = self._fn = None
        self.k_history = [K_START]
        self.alias_rate_log = []

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


print(f"\n-- LS20 {N_SEEDS} seeds x {PER_SEED_TIME}s (F2 Aliasing Rate K) --")
results = []

for seed_i in range(N_SEEDS):
    seed = SEED_BASE + seed_i * 100
    try:
        env = _make_env()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        sub = F2_AliasingRateK(n_actions=n_valid, seed=seed)
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
                        "K_final": state["K_current"], "k_hist": state["k_history"]})
        status = "L1" if l1_step else "  "
        alias_log = state["alias_rate_log"]
        last_rate = f"{alias_log[-1][2]:.2f}" if alias_log else "?"
        print(f"  seed={seed:>4} {status} steps={steps:>5} K={state['K_current']:>2} "
              f"k_hist={state['k_history']} alias_rate={last_rate}")
    except Exception as e:
        print(f"  seed={seed:>4} ERROR: {e}")
        import traceback; traceback.print_exc()
        results.append({"seed": seed, "l1": None, "steps": 0})

l1_count = sum(1 for r in results if r.get("l1"))
l1_steps = [r["l1"] for r in results if r.get("l1")]
avg_l1 = int(np.mean(l1_steps)) if l1_steps else None
k_finals = [r["K_final"] for r in results if "K_final" in r]
avg_k = float(np.mean(k_finals)) if k_finals else None
k_range = f"[{min(k_finals)}, {max(k_finals)}]" if k_finals else "?"
verdict = "PASS" if l1_count >= 17 else ("KILL" if l1_count < 14 else "MARGINAL")

print("\n" + "=" * 65)
print("F2 SUMMARY (ALIASING RATE K)")
print("=" * 65)
print(f"LS20 L1: {l1_count}/{N_SEEDS} {verdict}")
print(f"Avg steps to L1: {avg_l1}")
print(f"K_final avg: {avg_k} range: {k_range}")
print(f"If K converges to 10-14: substrate found game's natural resolution.")
print(f"If K stays at K_MAX=24: aliasing never drops — ALIAS_THRESH too low.")
print(f"If K stays near K_START=4: game is already well-resolved at 4 planes.")
print("=" * 65)
print("STEP 751 DONE")
print("=" * 65)
