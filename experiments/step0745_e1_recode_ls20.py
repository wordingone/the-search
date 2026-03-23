"""
Step 745 (E1): Recode substrate on new gym.

R3 hypothesis: Recode (LSH k=16 + passive self-refinement) achieved 5/5 L1 on
old gym. Confirm on new gym (action space 7, new game version ls20/9607627b).
Success: ≥4/10. Kill: 0/10 (likely API issue, not mechanism failure).
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM

print("=" * 65)
print("STEP 745 (E1) — RECODE SUBSTRATE ON NEW GYM LS20")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 10
PER_SEED_TIME = 25
K_RECODE = 16   # Recode used k=16
REFINE_EVERY_RECODE = 1000
MIN_H = 0.3     # entropy threshold for refinement


class Recode(BaseSubstrate):
    """Recode: LSH k=16 + passive self-refinement (no aliasing, no fine graph).

    Simpler than 674: single hash, refine when entropy > MIN_H.
    M elements: ref (refinement planes from passive update), G (edges).
    """

    def __init__(self, n_actions=7, seed=0):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(K_RECODE, DIM).astype(np.float32)
        self._n_actions = n_actions
        self.ref = {}       # hyperplane refinements
        self.G = {}         # (cell, action) -> {successor: count}
        self.C = {}         # (cell, action, successor) -> (sum_x, count) for refinement
        self.live = set()
        self._pn = self._pa = None
        self.t = 0
        self._cn = None

    def _hash(self, x):
        bits = (self.H @ x > 0).astype(np.uint8)
        n = self._hash_raw(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _hash_raw(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8),
                               bitorder='big').tobytes().hex(), 16)

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref:
                continue
            if len(d) < 2 or sum(d.values()) < 5:
                continue
            v = np.array(list(d.values()), np.float64)
            p = v / v.sum()
            h = float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))
            if h < MIN_H:
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
        n = self._hash(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(DIM, np.float64), 0))
            self.C[k] = (s + x.astype(np.float64), c + 1)
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY_RECODE == 0:
            self._refine()
        best_a, best_s = 0, float('inf')
        for a in range(self._n_actions):
            s = sum(self.G.get((n, a), {}).values())
            if s < best_s:
                best_s, best_a = s, a
        self._pn = n
        self._pa = best_a
        return best_a

    def get_state(self):
        return {"G_size": len(self.G), "live_count": len(self.live),
                "ref_count": len(self.ref), "t": self.t}

    def frozen_elements(self):
        return [
            {"name": "edge_count_update", "class": "M", "justification": "G grows by transitions."},
            {"name": "ref_hyperplanes", "class": "M", "justification": "Refinement planes passive."},
            {"name": "K_16", "class": "U", "justification": "K=16. System doesn't choose."},
            {"name": "binary_hash", "class": "I", "justification": "Sign projection. Irreducible."},
            {"name": "argmin_edges", "class": "I", "justification": "Argmin. Irreducible."},
        ]

    def reset(self, seed):
        rng = np.random.RandomState(seed * 1000)
        self.H = rng.randn(K_RECODE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}; self.live = set()
        self._pn = self._pa = None; self.t = 0; self._cn = None

    def on_level_transition(self):
        self._pn = None

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


print(f"\n-- LS20 {N_SEEDS} seeds x {PER_SEED_TIME}s (Recode) --")
results = []

for seed_i in range(N_SEEDS):
    seed = SEED_BASE + seed_i * 100
    try:
        env = _make_env()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
        sub = Recode(n_actions=n_valid, seed=seed)
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
        results.append({"seed": seed, "l1": l1_step, "steps": steps, "G": state["G_size"]})
        status = "L1" if l1_step else "  "
        print(f"  seed={seed:>4} {status} steps={steps:>5} G={state['G_size']:>4} ref={state['ref_count']:>2}")
    except Exception as e:
        print(f"  seed={seed:>4} ERROR: {e}")
        results.append({"seed": seed, "l1": None, "steps": 0})

l1_count = sum(1 for r in results if r.get("l1"))
verdict = "PASS" if l1_count >= 4 else ("KILL" if l1_count == 0 else "MARGINAL")

print("\n" + "=" * 65)
print("E1 SUMMARY (RECODE ON NEW GYM)")
print("=" * 65)
print(f"LS20 L1: {l1_count}/{N_SEEDS} {verdict}")
print(f"Expected: ≥4/10 if mechanism transfers. Prior: 5/5 on old gym.")
print("If 0/10: API/version issue. If low: mechanism degraded on new action space.")
print("=" * 65)
print("STEP 745 DONE")
print("=" * 65)
