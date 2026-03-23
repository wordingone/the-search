"""
Step 729 (C5): Encoding-statistics coupling (T9).

R3 hypothesis: aliased cell set A(t) stabilizes within phases, changes at
phase transitions. Test: autocorrelation and Jaccard similarity A(t) vs
A(t+Δt) should be high within phases (stable encoding) and low at transitions
(phase change disrupts aliasing pattern).

674 on chain (LS20 → FT09 → VC33). Record A(t) every 100 steps.
A(t) = aliased_count (proxy for aliased set size, since we can't serialize
the full set from get_state). Also track live_count and G_size.
Δt = 100 steps. Compute rolling Jaccard(A(t), A(t+100)) as proxy for set
similarity (|min(a,b)| / |max(a,b)| for count-based proxy).
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674

print("=" * 65)
print("STEP 729 (C5) — ENCODING-STATISTICS COUPLING (T9)")
print("=" * 65)

SEED = 0
N_STEPS_PER_GAME = 3_000
RECORD_EVERY = 100

timeline = []  # global: {global_step, phase, aliased_count, live_count, G_size}
phase_starts = {}


def _make_env(game):
    try:
        import arcagi3
        return arcagi3.make(game)
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game)


def run_phase_tracked(game, global_offset):
    print(f"\n-- {game} (offset={global_offset}) --")
    try:
        env = _make_env(game)
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 4
        sub = TransitionTriggered674(n_actions=n_valid, seed=SEED)
        sub.reset(SEED)
        obs = env.reset(seed=SEED)
        level = 0
        l1_step = l2_step = None
        steps = 0
        fresh = True
        t_start = time.time()
        prev_aliased = 0

        while steps < N_STEPS_PER_GAME and (time.time() - t_start) < 120:
            if obs is None:
                obs = env.reset(seed=SEED)
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
                if cl == 2 and l2_step is None:
                    l2_step = steps
                level = cl
                sub.on_level_transition()

            if done:
                obs = env.reset(seed=SEED)
                sub.on_level_transition()
                fresh = True

            if steps % RECORD_EVERY == 0:
                state = sub.get_state()
                cur_aliased = state.get("aliased_count", 0)
                # Jaccard proxy: min(a,b)/max(a,b) — count-based set similarity
                jac = min(cur_aliased, prev_aliased) / max(max(cur_aliased, prev_aliased), 1)
                timeline.append({
                    "phase": game,
                    "local_step": steps,
                    "global_step": global_offset + steps,
                    "aliased_count": cur_aliased,
                    "live_count": state.get("live_count", 0),
                    "G_size": state.get("G_size", 0),
                    "ref_count": state.get("ref_count", 0),
                    "jaccard": round(jac, 3),
                })
                prev_aliased = cur_aliased

        elapsed = time.time() - t_start
        print(f"  steps={steps} elapsed={elapsed:.1f}s l1={l1_step}")
        return steps

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return 0


global_offset = 0
for game in ["LS20", "FT09", "VC33"]:
    phase_starts[game] = global_offset
    n = run_phase_tracked(game, global_offset)
    global_offset += n

# --- Analysis ---
print("\n-- A(t) Timeline and Jaccard Analysis --")
print(f"  {'Global':>7} | {'Phase':>5} | {'Aliased':>8} | {'Jaccard':>8} | {'Live':>6}")
print("  " + "-" * 50)
for r in timeline:
    print(f"  {r['global_step']:>7} | {r['phase']:>5} | {r['aliased_count']:>8} | "
          f"{r['jaccard']:>8.3f} | {r['live_count']:>6}")

# Autocorrelation of aliased_count series
if len(timeline) > 2:
    counts = np.array([r["aliased_count"] for r in timeline], dtype=float)
    if counts.std() > 0:
        # Lag-1 autocorrelation
        acf1 = np.corrcoef(counts[:-1], counts[1:])[0, 1]
        # Lag-5 autocorrelation
        acf5 = np.corrcoef(counts[:-5], counts[5:])[0, 1] if len(counts) > 10 else None
        print(f"\n  Autocorrelation of A(t):")
        print(f"    lag-1: {acf1:.3f}")
        if acf5 is not None:
            print(f"    lag-5: {acf5:.3f}")
    else:
        print(f"\n  A(t) is constant = {counts[0]:.0f} (static aliasing)")

# Within-phase Jaccard
print(f"\n  Within-phase average Jaccard:")
for game in ["LS20", "FT09", "VC33"]:
    phase_data = [r for r in timeline if r["phase"] == game]
    if phase_data:
        jacs = [r["jaccard"] for r in phase_data[1:]]
        max_a = max(r["aliased_count"] for r in phase_data)
        print(f"    {game}: avg_jaccard={np.mean(jacs):.3f} max_aliased={max_a}")

# Phase-transition Jaccard (last point of phase N vs first of phase N+1)
games = ["LS20", "FT09", "VC33"]
print(f"\n  Phase-transition aliased counts:")
for i in range(len(games) - 1):
    g1 = [r for r in timeline if r["phase"] == games[i]]
    g2 = [r for r in timeline if r["phase"] == games[i+1]]
    if g1 and g2:
        a_end = g1[-1]["aliased_count"]
        a_start = g2[0]["aliased_count"]
        jac_trans = min(a_end, a_start) / max(max(a_end, a_start), 1)
        print(f"    {games[i]}→{games[i+1]}: A_end={a_end} A_start={a_start} "
              f"Jaccard={jac_trans:.3f}")

print(f"\n  T9 interpretation:")
print(f"    Within-phase high Jaccard -> A(t) stable within phases")
print(f"    Phase-transition low Jaccard -> A(t) resets at transitions")
print(f"    This would SUPPORT T9 (encoding-statistics coupling).")

print("\n" + "=" * 65)
print("STEP 729 DONE")
print("=" * 65)
