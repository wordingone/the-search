"""
Step 732 (B3): D2 Adaptive spatial resolution on LS20.

R3 hypothesis: inconsistent regions need finer spatial resolution.
Divide 64x64 into 16 regions (4x4 blocks of 16x16 pixels each).
I(region) = avg frame-to-frame pixel variance in that region.
High I → 2x2 pool (finer, 4 cells per region); Low I → 4x4 pool (coarser, 1 cell).

region_pool_sizes is an M element (modified by observed transition variance).
Replaces the 'avgpool16' U element in 674 with a system-driven M element.

20 seeds, 25s. Compare R3 to 674 baseline.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import K_NAV, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 732 (B3) — D2 ADAPTIVE SPATIAL RESOLUTION ON LS20")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 20
PER_SEED_TIME = 25

DIM = 256         # target encoding dimension
N_REGIONS = 16   # 4x4 grid of regions (each 16x16 pixels of 64x64)
HIGH_I_THRESH = 0.01   # variance threshold for "high inconsistency"


def _enc_d2(frame, region_pool_sizes, last_frame):
    """D2: adaptive spatial resolution encoding."""
    frame = np.array(frame, dtype=np.float32)
    if frame.ndim == 3:
        if frame.shape[0] <= 4 and frame.shape[1] > 4:
            frame = frame.transpose(1, 2, 0)
        a = frame[:, :, 0].astype(np.float32)
        if a.max() > 1:
            a = a / 15.0
    else:
        # Fallback: standard 16x16 pool
        x = frame.flatten()[:DIM].astype(np.float32)
        if len(x) < DIM:
            x = np.pad(x, (0, DIM - len(x)))
        return x - x.mean(), None

    h, w = a.shape
    # Ensure 64x64 (pad if needed)
    if h < 64 or w < 64:
        buf = np.zeros((64, 64), np.float32)
        buf[:min(h, 64), :min(w, 64)] = a[:min(h, 64), :min(w, 64)]
        a = buf

    # 4x4 grid of regions, each 16x16 pixels
    # region_pool_sizes: array of 16 floats (2.0=fine=2x2, 4.0=coarse=4x4)
    features = []
    region_vars = np.zeros(N_REGIONS, np.float32)

    for reg_i in range(N_REGIONS):
        ry = (reg_i // 4) * 16   # row start (0, 16, 32, 48)
        rx = (reg_i % 4) * 16    # col start
        region = a[ry:ry+16, rx:rx+16]

        pool_size = int(region_pool_sizes[reg_i])  # 2 or 4
        # Pool within region
        ph = pw = 16 // pool_size
        pooled = region.reshape(pool_size, ph, pool_size, pw).mean(axis=(1, 3))
        features.extend(pooled.flatten().tolist())

        # Compute frame-to-frame variance for this region
        if last_frame is not None and last_frame.shape == a.shape:
            region_var = float(np.var(a[ry:ry+16, rx:rx+16] - last_frame[ry:ry+16, rx:rx+16]))
        else:
            region_var = 0.0
        region_vars[reg_i] = region_var

    x = np.array(features, np.float32)[:DIM]
    if len(x) < DIM:
        x = np.pad(x, (0, DIM - len(x)))
    return x - x.mean(), region_vars


class D2_674(BaseSubstrate):
    """674 bootloader + D2: adaptive spatial resolution.

    region_pool_sizes[i] ∈ {2, 4}: 2=fine (high variance), 4=coarse (low variance).
    Updated by EMA of per-region frame variance. M element: region_pool_sizes.
    """

    def __init__(self, n_actions=4, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self._n_actions = n_actions
        # 674 core state
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
        # D2 state
        self.region_pool_sizes = np.full(N_REGIONS, 4.0, np.float32)  # start coarse
        self.region_var_ema = np.zeros(N_REGIONS, np.float32)
        self.last_a = None  # stored pooled frame for variance computation
        self.n_d2_obs = 0

    def _update_d2(self, region_vars):
        if self.n_d2_obs > 0:
            alpha = min(0.1, 1.0 / self.n_d2_obs)
            self.region_var_ema = (1-alpha)*self.region_var_ema + alpha*region_vars
            # Update pool sizes based on EMA variance
            for i in range(N_REGIONS):
                self.region_pool_sizes[i] = 2.0 if self.region_var_ema[i] > HIGH_I_THRESH else 4.0
        self.n_d2_obs += 1

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
        obs_arr = np.array(observation, dtype=np.float32)
        # Get D2 encoding with adaptive resolution
        x, region_vars = _enc_d2(obs_arr, self.region_pool_sizes, self.last_a)
        # Update last_a: store the 64x64 float frame for next step's variance
        if obs_arr.ndim == 3:
            a_frame = obs_arr.transpose(1, 2, 0)[:, :, 0] if obs_arr.shape[0] <= 4 else obs_arr[:, :, 0]
            if a_frame.max() > 1:
                a_frame = a_frame / 15.0
            h, w = a_frame.shape
            if h < 64 or w < 64:
                buf = np.zeros((64, 64), np.float32)
                buf[:min(h, 64), :min(w, 64)] = a_frame[:min(h, 64), :min(w, 64)]
                self.last_a = buf
            else:
                self.last_a = a_frame[:64, :64]
        if region_vars is not None:
            self._update_d2(region_vars)

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
        n_fine = int((self.region_pool_sizes == 2.0).sum())
        return {
            "G_size": len(self.G),
            "live_count": len(self.live),
            "aliased_count": len(self.aliased),
            "ref_count": len(self.ref),
            "t": self.t,
            "n_fine_regions": n_fine,   # D2: how many regions are in fine mode
            "region_pool_sizes": self.region_pool_sizes.copy(),
            "region_var_ema": self.region_var_ema.copy(),
        }

    def frozen_elements(self):
        return [
            {"name": "region_pool_sizes", "class": "M",
             "justification": "D2: pool size per region updated by observed transition variance. System-driven."},
            {"name": "channel_0_only", "class": "U",
             "justification": "Uses only first channel. System doesn't choose."},
            {"name": "mean_centering", "class": "U",
             "justification": "Subtract mean. System doesn't choose."},
            {"name": "HIGH_I_THRESH", "class": "U",
             "justification": "Threshold 0.01 for fine pooling. System doesn't choose."},
            {"name": "H_nav_planes", "class": "U",
             "justification": "k=12 random planes. System doesn't choose."},
            {"name": "H_fine_planes", "class": "U",
             "justification": "k=20 random planes. System doesn't choose."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection. Irreducible."},
            {"name": "argmin_edge_count", "class": "I",
             "justification": "Argmin of outgoing edges. Irreducible."},
            {"name": "fine_graph_priority", "class": "U",
             "justification": "Use fine graph at aliased cells. System doesn't choose."},
            {"name": "min_visits_alias", "class": "U",
             "justification": "MIN_VISITS=3. System doesn't choose."},
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
        self.region_pool_sizes = np.full(N_REGIONS, 4.0, np.float32)
        self.region_var_ema = np.zeros(N_REGIONS, np.float32)
        self.last_a = None
        self.n_d2_obs = 0

    def on_level_transition(self):
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


judge = ConstitutionalJudge()

# ---- Static R3 audit ----
print("\n-- Static R3 audit --")
elems = D2_674(n_actions=4, seed=0).frozen_elements()
m_names = [e["name"] for e in elems if e["class"] == "M"]
i_names = [e["name"] for e in elems if e["class"] == "I"]
u_names = [e["name"] for e in elems if e["class"] == "U"]
print(f"  D2_674: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
print(f"  M elements: {m_names}")
print(f"  U elements: {u_names}")
print(f"  Compare 674: M=3 I=3 U=9 — D2 adds 1M (region_pool_sizes), removes 1U (avgpool16)")

# ---- R3 dynamics (random obs) ----
print("\n-- R3 dynamics (2000 random obs) --")
class _D2(D2_674):
    def __init__(self): super().__init__(n_actions=4, seed=0)
r3_rand = judge.measure_r3_dynamics(_D2, n_steps=2000, n_checkpoints=10)
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
        sub = D2_674(n_actions=n_valid, seed=seed)
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
            "G": state["G_size"], "n_fine": state["n_fine_regions"]
        })
        status = "L1" if l1_step else "  "
        print(f"  seed={seed:>4} {status} steps={steps:>5} G={state['G_size']:>4} "
              f"fine_regions={state['n_fine_regions']:>2}/16")
    except Exception as e:
        print(f"  seed={seed:>4} ERROR: {e}")
        results.append({"seed": seed, "l1": None, "steps": 0, "error": str(e)})

l1_count = sum(1 for r in results if r.get("l1"))
avg_fine = np.mean([r.get("n_fine", 0) for r in results if "n_fine" in r])

print("\n" + "=" * 65)
print("B3 SUMMARY")
print("=" * 65)
print(f"D2_674 R3 static: M={len(m_names)} I={len(i_names)} U={len(u_names)}")
verdict = "PASS" if l1_count >= 17 else ("KILL" if l1_count < 14 else "MARGINAL")
print(f"LS20 L1 success: {l1_count}/{N_SEEDS} {verdict}")
print(f"Final avg fine regions: {avg_fine:.1f}/16")
print(f"R3 dynamic (random): {r3_rand.get('r3_dynamic_score')}")
print("=" * 65)
print("STEP 732 DONE")
print("=" * 65)
