"""
Step 746 (E2): Non-argmin action selection with 674 encoding on LS20.

R3 hypothesis (Prop 15): action selection mechanism doesn't matter for L1.
Test 3 alternatives to argmin:
1. Random uniform action selection
2. Softmax(T=1) over visit counts (prefer less-visited)
3. Epsilon-greedy(ε=0.1) argmin

674 encoding + each selector. LS20, 25s, 10 seeds each.
Success: all ≥5/10. Kill: any <3/10.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM, K_NAV, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT

print("=" * 65)
print("STEP 746 (E2) — NON-ARGMIN ACTION SELECTION LS20")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 10
PER_SEED_TIME = 25
EPSILON = 0.1
SOFTMAX_T = 1.0


def _make_674_core(n_actions, seed):
    """Build 674 core state."""
    rng = np.random.RandomState(seed)
    return {
        'H_nav': rng.randn(K_NAV, DIM).astype(np.float32),
        'H_fine': rng.randn(K_FINE, DIM).astype(np.float32),
        'n_actions': n_actions,
        'ref': {}, 'G': {}, 'C': {}, 'live': set(),
        'G_fine': {}, 'aliased': set(),
        '_pn': None, '_pa': None, '_px': None, '_pfn': None,
        't': 0, '_cn': None, '_fn': None,
    }


class _674Base(BaseSubstrate):
    """674 encoding without selection — subclasses override _select."""

    def __init__(self, n_actions=7, seed=0, selector='argmin'):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self._n_actions = n_actions
        self.selector = selector
        self._rng = np.random.RandomState(seed + 999)
        self.ref = {}; self.G = {}; self.C = {}
        self.live = set(); self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None
        self._pfn = None; self.t = 0; self._cn = self._fn = None

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

    def _get_counts(self):
        """Return visit counts for each action from current cell."""
        if self._cn in self.aliased and self._fn is not None:
            G_use = self.G_fine
            key_prefix = self._fn
        else:
            G_use = self.G
            key_prefix = self._cn
        return [sum(G_use.get((key_prefix, a), {}).values()) for a in range(self._n_actions)]

    def _select(self):
        counts = self._get_counts()
        if self.selector == 'argmin':
            return int(np.argmin(counts))
        elif self.selector == 'random':
            return int(self._rng.randint(0, self._n_actions))
        elif self.selector == 'softmax':
            # Softmax over NEGATIVE counts (prefer less visited)
            neg_counts = np.array([-c for c in counts], dtype=np.float64)
            neg_counts -= neg_counts.max()
            probs = np.exp(neg_counts / max(SOFTMAX_T, 1e-8))
            probs /= probs.sum()
            return int(self._rng.choice(self._n_actions, p=probs))
        elif self.selector == 'epsilon_greedy':
            if self._rng.random() < EPSILON:
                return int(self._rng.randint(0, self._n_actions))
            return int(np.argmin(counts))
        return 0

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
        return {"G_size": len(self.G), "live_count": len(self.live), "t": self.t}

    def frozen_elements(self):
        return [
            {"name": "edge_count_update", "class": "M", "justification": "G updated by transitions."},
            {"name": "aliased_set", "class": "M", "justification": "Aliased cells grow."},
            {"name": "ref_hyperplanes", "class": "M", "justification": "Refinement planes."},
            {"name": "avgpool16", "class": "U", "justification": "16x16 pooling."},
            {"name": "binary_hash", "class": "I", "justification": "Sign projection. Irreducible."},
            {"name": f"action_select_{self.selector}", "class": "U",
             "justification": f"{self.selector} selection — tested as alternative to argmin."},
        ]

    def reset(self, seed):
        rng = np.random.RandomState(seed * 1000)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}
        self.live = set(); self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None
        self._pfn = None; self.t = 0; self._cn = self._fn = None
        self._rng = np.random.RandomState(seed + 999)

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


selectors = ['random', 'softmax', 'epsilon_greedy']
all_results = {}

for sel in selectors:
    print(f"\n-- Selector: {sel} ({N_SEEDS} seeds x {PER_SEED_TIME}s) --")
    results = []
    for seed_i in range(N_SEEDS):
        seed = SEED_BASE + seed_i * 100
        try:
            env = _make_env()
            n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
            sub = _674Base(n_actions=n_valid, seed=seed, selector=sel)
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
            results.append({"seed": seed, "l1": l1_step, "steps": steps, "G": state["G_size"]})
            status = "L1" if l1_step else "  "
            print(f"  seed={seed:>4} {status} steps={steps:>5} G={state['G_size']:>4}")
        except Exception as e:
            print(f"  seed={seed:>4} ERROR: {e}")
            results.append({"seed": seed, "l1": None, "steps": 0})
    l1_count = sum(1 for r in results if r.get("l1"))
    verdict = "PASS" if l1_count >= 5 else ("KILL" if l1_count < 3 else "MARGINAL")
    all_results[sel] = {"l1": l1_count, "verdict": verdict}
    print(f"  {sel}: L1={l1_count}/{N_SEEDS} {verdict}")

print("\n" + "=" * 65)
print("E2 SUMMARY (NON-ARGMIN ACTION SELECTION)")
print("=" * 65)
for sel, r in all_results.items():
    print(f"  {sel:<20}: L1={r['l1']}/{N_SEEDS} {r['verdict']}")
print("Prop 15: all selectors should work if encoding is sufficient")
prop15 = all(r["l1"] >= 5 for r in all_results.values())
print(f"Prop 15 {'CONFIRMED' if prop15 else 'FALSIFIED'}")
print("=" * 65)
print("STEP 746 DONE")
print("=" * 65)
