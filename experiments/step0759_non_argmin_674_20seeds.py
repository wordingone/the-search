"""
Step 759 - Non-argmin selectors on 674+running-mean, 20 seeds.

Definitive selector comparison at good perception quality (674 encoding).
Step 746 was 10 seeds. This rerun uses 20 seeds for statistical power.

Five selectors: argmin, random, epsilon(0.1), softmax(T=0.5), softmax(T=1.0).
LS20, 20 seeds, 25s each per selector (5 runs of 20 seeds).

R3 note: selector choice is I (argmin) in 674. This tests whether argmin
advantage holds at n=20 with 674's perception quality vs Step 746 at n=10.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame, DIM, K_NAV, K_FINE, REFINE_EVERY, MIN_VISITS_ALIAS, MIN_OBS, H_SPLIT

print("=" * 65)
print("STEP 759 - NON-ARGMIN SELECTORS ON 674, 20 SEEDS")
print("=" * 65)

SEED_BASE = 0
N_SEEDS = 20
PER_SEED_TIME = 25

SELECTORS = ["argmin", "random", "epsilon", "softmax05", "softmax10"]


def run_selector(selector_name, n_seeds, seed_base, per_seed_time):
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
            rng_s = np.random.RandomState(seed + 99999)  # for stochastic selectors

            rng = np.random.RandomState(seed)
            H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
            H_fine = rng.randn(K_FINE, DIM).astype(np.float32)

            ref = {}; G = {}; C = {}
            live = set(); G_fine = {}; aliased = set()
            pn = pa = pfn = cn = fn_ = None
            t = 0

            def _node(x):
                n = int(np.packbits((H_nav @ x > 0).astype(np.uint8),
                                    bitorder='big').tobytes().hex(), 16)
                while n in ref:
                    n = (n, int(ref[n] @ x > 0))
                return n

            def _hash_fine(x):
                return int(np.packbits((H_fine @ x > 0).astype(np.uint8),
                                       bitorder='big').tobytes().hex(), 16)

            def _select(cn_, fn_):
                if cn_ in aliased and fn_ is not None:
                    cands = {a: sum(G_fine.get((fn_, a), {}).values()) for a in range(n_valid)}
                else:
                    cands = {a: sum(G.get((cn_, a), {}).values()) for a in range(n_valid)}

                if selector_name == "argmin":
                    return min(cands, key=cands.get)
                elif selector_name == "random":
                    return int(rng_s.randint(n_valid))
                elif selector_name == "epsilon":
                    if rng_s.random() < 0.1:
                        return int(rng_s.randint(n_valid))
                    return min(cands, key=cands.get)
                elif selector_name in ("softmax05", "softmax10"):
                    T = 0.5 if selector_name == "softmax05" else 1.0
                    vals = np.array([cands[a] for a in range(n_valid)], np.float64)
                    inv = -vals / T
                    inv -= inv.max()
                    probs = np.exp(inv); probs /= probs.sum()
                    return int(rng_s.choice(n_valid, p=probs))
                return 0

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
                    r0 = C.get((n, a, top[0])); r1 = C.get((n, a, top[1]))
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
                    obs = env.reset(seed=seed); pn = pfn = None; fresh = True; continue
                x = _enc_frame(np.array(obs, dtype=np.float32))
                n = _node(x); fn_v = _hash_fine(x)
                live.add(n); t += 1
                if pn is not None:
                    d = G.setdefault((pn, pa), {}); d[n] = d.get(n, 0) + 1
                    k = (pn, pa, n)
                    s, c = C.get(k, (np.zeros(DIM, np.float64), 0))
                    C[k] = (s + x.astype(np.float64), c + 1)
                    succ = G.get((pn, pa), {})
                    if sum(succ.values()) >= MIN_VISITS_ALIAS and len(succ) >= 2:
                        aliased.add(pn)
                    if pn in aliased and pfn is not None:
                        df = G_fine.setdefault((pfn, pa), {})
                        df[fn_v] = df.get(fn_v, 0) + 1
                cn = n; fn_ = fn_v
                if t > 0 and t % REFINE_EVERY == 0:
                    _refine()
                action = _select(cn, fn_)
                pn = n; pfn = fn_v; pa = action
                obs, _, done, info = env.step(action % n_valid); steps += 1
                if fresh:
                    fresh = False; continue
                cl = info.get('level', 0) if isinstance(info, dict) else 0
                if cl > level:
                    if cl == 1 and l1_step is None:
                        l1_step = steps
                    level = cl; pn = pfn = None
                if done:
                    obs = env.reset(seed=seed); pn = pfn = None; fresh = True
            results.append({"seed": seed, "l1": l1_step, "steps": steps})
            status = "L1" if l1_step else "  "
            print(f"    seed={seed:>4} {status} l1_step={l1_step}")
        except Exception as e:
            print(f"    seed={seed:>4} ERROR: {e}")
            results.append({"seed": seed, "l1": None, "steps": 0})
    return results


all_results = {}

for sel in SELECTORS:
    print(f"\n-- Selector: {sel} ({N_SEEDS} seeds x {PER_SEED_TIME}s) --")
    r = run_selector(sel, N_SEEDS, SEED_BASE, PER_SEED_TIME)
    l1_count = sum(1 for x in r if x.get("l1"))
    all_results[sel] = l1_count
    print(f"  {sel}: L1={l1_count}/{N_SEEDS}")

print("\n" + "=" * 65)
print("STEP 759 SUMMARY - NON-ARGMIN SELECTORS ON 674 (20 SEEDS)")
print("=" * 65)
print(f"{'Selector':<14} {'L1':>5} {'Rate':>8}")
print("-" * 30)
for sel in SELECTORS:
    cnt = all_results[sel]
    print(f"{sel:<14} {cnt:>3}/{N_SEEDS:<3} {cnt/N_SEEDS:>7.1%}")
print("-" * 30)
argmin_l1 = all_results["argmin"]
random_l1 = all_results["random"]
advantage = argmin_l1 - random_l1
print(f"Argmin advantage over random: {advantage:+d} seeds")
print(f"E2 (Step 746, 10 seeds): argmin=5, random=4, eps=5")
print(f"Step 653 (plain k=12): argmin≈random≈3/20 — perception-gated")
print(f"If argmin >> random: perception quality unlocks argmin advantage.")
print("=" * 65)
print("STEP 759 DONE")
print("=" * 65)
