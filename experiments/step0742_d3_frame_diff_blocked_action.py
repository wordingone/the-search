"""
Step 742 (D3): Frame-diff blocked-action detection on LS20.

R3 hypothesis: blocked actions (frame_diff < threshold) waste steps. Skipping them
is self-directed attention on actions — the substrate modifies WHICH actions it
counts (its counting mechanism), not just WHAT it observes.

If frame_diff(obs_t, obs_{t+1}) < BLOCK_THRESH after action a from cell n,
don't increment G[(n,a)][successor]. Instead, immediately try the next action.
Blocked action tracker is an M element (grows as blocked actions are discovered).

LS20, 25s, 20 seeds.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM, K_NAV, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT

print("=" * 65)
print("STEP 742 (D3) — FRAME-DIFF BLOCKED ACTION DETECTION LS20")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 20
PER_SEED_TIME = 25
BLOCK_THRESH = 0.082   # from prior step — mean pixel diff threshold for blocked action


class D3_Blocked(BaseSubstrate):
    """674 + blocked action detection via frame diff.

    M element: blocked_actions — set of (cell, action) pairs known to be no-ops.
    When frame_diff < BLOCK_THRESH, mark (prev_cell, action) as blocked.
    Argmin skips blocked actions; explores unblocked actions first.
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
        # D3: blocked actions
        self.blocked = set()   # (cell, action) pairs known to be no-ops
        self._last_x = None    # last encoding for frame diff

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
        # Try unblocked actions first with argmin; fall back to blocked if all blocked
        if self._cn in self.aliased and self._fn is not None:
            unblocked = [a for a in range(self._n_actions)
                         if (self._fn, a) not in self.blocked]
            pool = unblocked if unblocked else list(range(self._n_actions))
            best_a, best_s = pool[0], float('inf')
            for a in pool:
                s = sum(self.G_fine.get((self._fn, a), {}).values())
                if s < best_s:
                    best_s, best_a = s, a
            return best_a
        unblocked = [a for a in range(self._n_actions)
                     if (self._cn, a) not in self.blocked]
        pool = unblocked if unblocked else list(range(self._n_actions))
        best_a, best_s = pool[0], float('inf')
        for a in pool:
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
            # Compute frame diff to check if previous action was blocked
            frame_diff = 0.0
            if self._last_x is not None:
                frame_diff = float(np.mean(np.abs(x - self._last_x)))

            if frame_diff < BLOCK_THRESH:
                # Previous action produced no real change — mark as blocked
                self.blocked.add((self._pn, self._pa))
                if self._pn in self.aliased and self._pfn is not None:
                    self.blocked.add((self._pfn, self._pa))
            else:
                # Valid transition — update graph normally
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

        self._last_x = x
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
            "blocked_count": len(self.blocked),
        }

    def frozen_elements(self):
        return [
            {"name": "blocked_actions", "class": "M",
             "justification": "Set of (cell,action) no-ops discovered via frame diff. System-driven."},
            {"name": "edge_count_update", "class": "M",
             "justification": "G updated (only for non-blocked actions). System-driven."},
            {"name": "aliased_set", "class": "M",
             "justification": "Aliased cells grow. System-driven."},
            {"name": "ref_hyperplanes", "class": "M",
             "justification": "Refinement planes. System-derived."},
            {"name": "block_threshold", "class": "U",
             "justification": "BLOCK_THRESH=0.082. System doesn't choose."},
            {"name": "avgpool16", "class": "U",
             "justification": "16x16 pooling. System doesn't choose."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection. Irreducible."},
            {"name": "argmin_unblocked", "class": "I",
             "justification": "Argmin over unblocked actions. Irreducible."},
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
        self.blocked = set()
        self._last_x = None

    def on_level_transition(self):
        self._pn = None
        self._pfn = None
        self._px = None
        self._last_x = None

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
        sub = D3_Blocked(n_actions=n_valid, seed=seed)
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
                        "G": state["G_size"], "blocked": state["blocked_count"]})
        status = "L1" if l1_step else "  "
        print(f"  seed={seed:>4} {status} steps={steps:>5} G={state['G_size']:>4} "
              f"blocked={state['blocked_count']:>3} l1={l1_step}")
    except Exception as e:
        print(f"  seed={seed:>4} ERROR: {e}")
        results.append({"seed": seed, "l1": None, "steps": 0, "error": str(e)})

l1_count = sum(1 for r in results if r.get("l1"))
avg_blocked = np.mean([r["blocked"] for r in results if "blocked" in r])
verdict = "PASS" if l1_count >= 17 else ("KILL" if l1_count < 14 else "MARGINAL")

print("\n" + "=" * 65)
print("D3 SUMMARY (BLOCKED ACTION DETECTION)")
print("=" * 65)
print(f"LS20 L1: {l1_count}/{N_SEEDS} {verdict}")
print(f"Avg blocked pairs: {avg_blocked:.1f}")
print(f"Compare: 674 baseline on LS20 (25s, 7 actions) needed")
print("=" * 65)
print("STEP 742 DONE")
print("=" * 65)
