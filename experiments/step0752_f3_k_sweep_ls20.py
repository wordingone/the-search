"""
Step 752 (F3): K sweep — find minimum sufficient K_NAV on LS20.

R3 hypothesis (diagnostic): K_NAV=12 is a U element. Is it the minimum
sufficient value, or is it over-specified? If minimum K=8, then K_NAV=12
has headroom and is not principled. If minimum K=12, it's justified as
the minimum sufficient plane count for LS20 discrimination.

Run 674 with K_NAV in {4, 6, 8, 10, 12, 16} × 10 seeds × 25s.
Find minimum K achieving ≥7/10. Report L1 rates across K values.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame, DIM, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT

print("=" * 65)
print("STEP 752 (F3) — K SWEEP: MINIMUM SUFFICIENT K_NAV")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 10
PER_SEED_TIME = 25
K_VALUES = [4, 6, 8, 10, 12, 16]


def _hash_k_bits(x, H, k):
    bits = (H[:k] @ x > 0).astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def run_674_with_k(k_nav, n_seeds, seed_base, per_seed_time):
    """Run 674 with fixed K_NAV=k_nav. Returns list of l1 results."""
    results = []
    for seed_i in range(n_seeds):
        seed = seed_base + seed_i * 100
        try:
            try:
                import arcagi3
                env = arcagi3.make("LS20")
            except ImportError:
                import util_arcagi3
                env = util_arcagi3.make("LS20")

            n_valid = len(env._action_space) if hasattr(env, '_action_space') else 7
            rng = np.random.RandomState(seed)
            H_nav = rng.randn(k_nav, DIM).astype(np.float32)
            H_fine = rng.randn(K_FINE, DIM).astype(np.float32)

            ref = {}; G = {}; C = {}
            live = set(); G_fine = {}; aliased = set()
            pn = pa = px = pfn = cn = fn = None
            t = 0

            def _node(x):
                n = _hash_k_bits(x, H_nav, k_nav)
                while n in ref:
                    n = (n, int(ref[n] @ x > 0))
                return n

            def _hash_fine(x):
                return int(np.packbits((H_fine @ x > 0).astype(np.uint8),
                                       bitorder='big').tobytes().hex(), 16)

            def _select(cn_, fn_):
                if cn_ in aliased and fn_ is not None:
                    best_a, best_s = 0, float('inf')
                    for a in range(n_valid):
                        s = sum(G_fine.get((fn_, a), {}).values())
                        if s < best_s:
                            best_s, best_a = s, a
                    return best_a
                best_a, best_s = 0, float('inf')
                for a in range(n_valid):
                    s = sum(G.get((cn_, a), {}).values())
                    if s < best_s:
                        best_s, best_a = s, a
                return best_a

            def _refine():
                did = 0
                for (n, a), d in list(G.items()):
                    if n not in live or n in ref:
                        continue
                    if len(d) < 2 or sum(d.values()) < MIN_OBS:
                        continue
                    v = np.array(list(d.values()), np.float64)
                    p = v / v.sum()
                    h = float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))
                    if h < H_SPLIT:
                        continue
                    top = sorted(d, key=d.get, reverse=True)[:2]
                    r0 = C.get((n, a, top[0]))
                    r1 = C.get((n, a, top[1]))
                    if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3:
                        continue
                    diff = r0[0]/r0[1] - r1[0]/r1[1]
                    nm = np.linalg.norm(diff)
                    if nm < 1e-8:
                        continue
                    ref[n] = (diff / nm).astype(np.float32)
                    live.discard(n)
                    did += 1
                    if did >= 3:
                        break

            obs = env.reset(seed=seed)
            level = 0; l1_step = None; steps = 0; fresh = True
            t_start = time.time()
            while (time.time() - t_start) < per_seed_time:
                if obs is None:
                    obs = env.reset(seed=seed); pn = pfn = px = None; fresh = True; continue
                x = _enc_frame(np.array(obs, dtype=np.float32))
                n = _node(x)
                fn_val = _hash_fine(x)
                live.add(n)
                t += 1
                if pn is not None:
                    d = G.setdefault((pn, pa), {})
                    d[n] = d.get(n, 0) + 1
                    k_key = (pn, pa, n)
                    s, c = C.get(k_key, (np.zeros(DIM, np.float64), 0))
                    C[k_key] = (s + x.astype(np.float64), c + 1)
                    succ = G.get((pn, pa), {})
                    if sum(succ.values()) >= MIN_VISITS_ALIAS and len(succ) >= 2:
                        aliased.add(pn)
                    if pn in aliased and pfn is not None:
                        df = G_fine.setdefault((pfn, pa), {})
                        df[fn_val] = df.get(fn_val, 0) + 1
                cn = n; fn = fn_val
                if t > 0 and t % REFINE_EVERY == 0:
                    _refine()
                action = _select(cn, fn)
                pn = n; pfn = fn_val; pa = action
                obs_new, reward, done, info = env.step(action % n_valid)
                obs = obs_new; steps += 1
                if fresh:
                    fresh = False; continue
                cl = info.get('level', 0) if isinstance(info, dict) else 0
                if cl > level:
                    if cl == 1 and l1_step is None:
                        l1_step = steps
                    level = cl; pn = pfn = px = None
                if done:
                    obs = env.reset(seed=seed); pn = pfn = px = None; fresh = True
            results.append({"seed": seed, "l1": l1_step, "steps": steps,
                            "aliased": len(aliased), "live": len(live)})
            status = "L1" if l1_step else "  "
            print(f"    seed={seed:>4} {status} steps={steps:>5} aliased={len(aliased):>3}/{len(live):>3}")
        except Exception as e:
            print(f"    seed={seed:>4} ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append({"seed": seed, "l1": None, "steps": 0})
    return results


all_results = {}

for k in K_VALUES:
    print(f"\n-- K_NAV={k} ({N_SEEDS} seeds x {PER_SEED_TIME}s) --")
    r = run_674_with_k(k, N_SEEDS, SEED_BASE, PER_SEED_TIME)
    l1_count = sum(1 for x in r if x.get("l1"))
    all_results[k] = {"results": r, "l1_count": l1_count}
    print(f"  K={k}: L1={l1_count}/{N_SEEDS}")

print("\n" + "=" * 65)
print("F3 SUMMARY (K SWEEP)")
print("=" * 65)
print(f"{'K_NAV':<8} {'L1':>5} {'Rate':>8} {'Status':>10}")
print("-" * 35)
for k in K_VALUES:
    cnt = all_results[k]["l1_count"]
    rate = cnt / N_SEEDS
    status = "PASS" if cnt >= 7 else ("FAIL" if cnt == 0 else "MARGINAL")
    print(f"K={k:<6} {cnt:>3}/{N_SEEDS:<3} {rate:>7.1%} {status:>10}")
print("-" * 35)
min_k_pass = next((k for k in K_VALUES if all_results[k]["l1_count"] >= 7), None)
print(f"Minimum sufficient K (>=7/10): {min_k_pass}")
print(f"674 baseline K_NAV=12: compare to K=12 row above")
print(f"If min_k < 12: K_NAV=12 is over-specified (has headroom).")
print(f"If min_k = 12: K_NAV=12 is the minimum sufficient value.")
print("=" * 65)
print("STEP 752 DONE")
print("=" * 65)
