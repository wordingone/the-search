"""
Step 733 (B4): D3 Continuous K per cell on LS20.

R3 hypothesis: binary K=12/20 is coarse. Continuous K provides smoother
self-directed attention — cells with more inconsistency get finer hashing.
K(n) = K_min + 2 * floor(I(n)/I_step), K ∈ {8,10,12,14,16,18,20,24}.
inconsistency_map is an M element (grows with distinct successors).

20 seeds, 25s each. Report R3 audit + dynamic score vs 674 baseline.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 733 (B4) — D3 CONTINUOUS K PER CELL ON LS20")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 20
PER_SEED_TIME = 25

K_MIN = 8
K_MAX = 24
I_STEP = 1   # 1 distinct successor per action → +2 bits


def _hash_k(x, H, k):
    """First k bits of hash from H planes."""
    bits = (H[:k] @ x > 0).astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


class D3_674(BaseSubstrate):
    """D3: Continuous K per cell — K grows with cell inconsistency.

    K(n) = K_min + 2 * floor(I(n) / I_step), capped at K_max.
    I(n) = max distinct successors across all actions from base cell n.

    M elements: edge_count_update (G grows), inconsistency_map (I(n) grows).
    The inconsistency_map is the novel R3 element — it self-directs hash resolution.
    """

    def __init__(self, n_actions=4, seed=0):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(K_MAX, DIM).astype(np.float32)  # K_MAX shared planes
        self._n_actions = n_actions
        self.G = {}              # (cell, action) -> {successor: count}
        self.live = set()
        self.inconsistency = {}  # base_cell -> max distinct successors seen
        self._pn = None
        self._pbase = None
        self._pa = None
        self.t = 0

    def _get_k(self, base_cell):
        inc = self.inconsistency.get(base_cell, 0)
        return min(K_MIN + 2 * (inc // max(I_STEP, 1)), K_MAX)

    def process(self, observation):
        x = _enc_frame(observation)
        base = _hash_k(x, self.H, K_MIN)   # stable 8-bit base
        k = self._get_k(base)
        cell = _hash_k(x, self.H, k)        # adaptive K-bit cell

        self.live.add(cell)
        self.t += 1

        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[cell] = d.get(cell, 0) + 1
            # Update inconsistency for prev base
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
            "inconsistency_map_size": len(self.inconsistency),
        }

    def frozen_elements(self):
        return [
            {"name": "H_24planes", "class": "U",
             "justification": "24 random LSH planes. System doesn't choose count or direction."},
            {"name": "K_min", "class": "U",
             "justification": "K_MIN=8. System doesn't choose minimum resolution."},
            {"name": "I_step", "class": "U",
             "justification": "I_STEP=1. System doesn't choose K increment threshold."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection for cell identity. Removing destroys graph. Irreducible."},
            {"name": "argmin_edge_count", "class": "I",
             "justification": "Argmin of outgoing edges. Removing causes random walk. Irreducible."},
            {"name": "edge_count_update", "class": "M",
             "justification": "G updated by each (cell, action, successor) transition. System-driven."},
            {"name": "inconsistency_map", "class": "M",
             "justification": "I(n) grows as distinct successors are observed. Determines K per cell. System-driven."},
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

    def on_level_transition(self):
        self._pn = None
        self._pbase = None
        self._pa = None

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
elems = D3_674(n_actions=4, seed=0).frozen_elements()
m_names = [e["name"] for e in elems if e["class"] == "M"]
i_names = [e["name"] for e in elems if e["class"] == "I"]
u_names = [e["name"] for e in elems if e["class"] == "U"]
print(f"  D3_674: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"  M elements: {m_names}")
print(f"  U elements: {u_names}")
print(f"  Compare 674: M=3 I=3 U=9 (9U → D3: {len(u_names)}U)")

# ---- R3 dynamics (random obs) ----
print("\n-- R3 dynamics (2000 random obs, 10 ckpts) --")
class _D3(D3_674):
    def __init__(self): super().__init__(n_actions=4, seed=0)
r3_rand = judge.measure_r3_dynamics(_D3, n_steps=2000, n_checkpoints=10)
print(f"  R3 score: {r3_rand.get('r3_dynamic_score')} profile: {r3_rand.get('dynamics_profile')}")
print(f"  Verified M: {r3_rand.get('verified_m_elements', [])}")

# ---- 20 seeds on LS20 ----
print(f"\n-- LS20 {N_SEEDS} seeds x {PER_SEED_TIME}s --")
results = []
obs_seq_seed0 = []

for seed_i in range(N_SEEDS):
    seed = SEED_BASE + seed_i * 100
    try:
        env = _make_env()
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 4
        sub = D3_674(n_actions=n_valid, seed=seed)
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
                obs_seq_seed0.append(obs_arr)
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
        results.append({
            "seed": seed, "l1": l1_step, "steps": steps,
            "G_size": state["G_size"], "avg_k": state["avg_k"], "max_k": state["max_k"],
            "n_base": state["n_base_cells"],
        })
        status = "L1" if l1_step else "  "
        print(f"  seed={seed:>4} {status} steps={steps:>5} G={state['G_size']:>4} "
              f"base={state['n_base_cells']:>3} avgK={state['avg_k']:.1f} maxK={state['max_k']:.0f}")
    except Exception as e:
        print(f"  seed={seed:>4} ERROR: {e}")
        results.append({"seed": seed, "l1": None, "steps": 0, "error": str(e)})

l1_count = sum(1 for r in results if r.get("l1"))
avg_k_final = np.mean([r["avg_k"] for r in results if "avg_k" in r])

# ---- R3 dynamics on real LS20 obs ----
if obs_seq_seed0:
    print(f"\n-- R3 dynamics (real LS20 obs, {len(obs_seq_seed0)} steps) --")
    n_obs = min(len(obs_seq_seed0), 6000)
    r3_real = judge.measure_r3_dynamics(
        _D3, obs_sequence=obs_seq_seed0[:n_obs], n_steps=n_obs, n_checkpoints=10)
    print(f"  R3 score: {r3_real.get('r3_dynamic_score')} profile: {r3_real.get('dynamics_profile')}")
    print(f"  Verified M: {r3_real.get('verified_m_elements', [])}")

print("\n" + "=" * 65)
print("B4 SUMMARY")
print("=" * 65)
print(f"D3_674 R3 static: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"Compare 674: M=3 I=3 U=9 — D3: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"Key delta: -1U (k_12 removed) -1U (H_nav_planes), +1M (inconsistency_map)")
verdict = "PASS" if l1_count >= 17 else ("KILL" if l1_count < 14 else "MARGINAL")
print(f"LS20 L1 success: {l1_count}/{N_SEEDS} {verdict}")
print(f"Final avg K per cell: {avg_k_final:.2f} (started at {K_MIN})")
print(f"R3 dynamic score (random): {r3_rand.get('r3_dynamic_score')}")
if obs_seq_seed0:
    print(f"R3 dynamic score (real LS20): {r3_real.get('r3_dynamic_score')}")
print("=" * 65)
print("STEP 733 DONE")
print("=" * 65)
