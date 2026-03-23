"""
Step 727 (C3): R3 dynamics across chain with phase boundaries.

R3 hypothesis (T9): R3 changes spike at phase transitions, stabilize within
phases. The aliased cell set A(t) should be quasi-stable within a phase
(Jaccard similarity A(t) ~ A(t+Δt) high within phase, drops at transition).

Chain: LS20 → FT09 → VC33. 3K steps per game (500-step resolution = 6 checkpoints/game).
Fresh substrate per phase (matching chain benchmark semantics). Record
aliased_set at each checkpoint, compute Jaccard similarity between
adjacent checkpoints.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 727 (C3) — R3 DYNAMICS: 674 ACROSS CHAIN (PHASE TRANSITIONS)")
print("=" * 65)

SEED = 0
N_STEPS_PER_GAME = 3_000
CHECKPOINT_EVERY = 500

judge = ConstitutionalJudge()
all_checkpoints = []  # global timeline
phase_boundaries = {}


def _make_env(game):
    try:
        import arcagi3
        return arcagi3.make(game)
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game)


def run_phase(game, global_offset):
    """Run one phase, return checkpoints."""
    print(f"\n-- Phase: {game} --")
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
        phase_checkpoints = []
        prev_aliased = set()
        t_start = time.time()

        while steps < N_STEPS_PER_GAME and (time.time() - t_start) < 150:
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

            if steps % CHECKPOINT_EVERY == 0:
                state = sub.get_state()
                cur_aliased = set(range(state.get("aliased_count", 0)))  # proxy
                # Jaccard: |A ∩ B| / |A ∪ B| (use counts as proxy)
                a_count = state.get("aliased_count", 0)
                p_count = len(prev_aliased)
                jaccard = min(a_count, p_count) / max(max(a_count, p_count), 1)
                prev_aliased = cur_aliased

                ckpt = {
                    "phase": game,
                    "local_step": steps,
                    "global_step": global_offset + steps,
                    "aliased_count": a_count,
                    "live_count": state.get("live_count", 0),
                    "G_size": state.get("G_size", 0),
                    "ref_count": state.get("ref_count", 0),
                    "jaccard_approx": round(jaccard, 3),
                }
                phase_checkpoints.append(ckpt)
                all_checkpoints.append(ckpt)

        elapsed = time.time() - t_start
        print(f"  steps={steps} elapsed={elapsed:.1f}s l1={l1_step} l2={l2_step}")

        print(f"  {'Local':>6} | {'Aliased':>8} | {'Live':>6} | {'G':>7} | {'Jaccard':>8}")
        print("  " + "-" * 50)
        for c in phase_checkpoints:
            print(f"  {c['local_step']:>6} | {c['aliased_count']:>8} | "
                  f"{c['live_count']:>6} | {c['G_size']:>7} | {c['jaccard_approx']:>8.3f}")

        return phase_checkpoints, steps, l1_step

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return [], 0, None


global_offset = 0
for game in ["LS20", "FT09", "VC33"]:
    phase_boundaries[game] = global_offset
    ckpts, n_steps, l1 = run_phase(game, global_offset)
    global_offset += n_steps

# --- T9 analysis ---
print("\n-- T9 Hypothesis Analysis --")
print(f"  Phase boundaries (global step):")
for game, offset in phase_boundaries.items():
    print(f"    {game}: starts at step {offset}")

print(f"\n  Global timeline:")
print(f"  {'Global':>7} | {'Phase':>5} | {'Aliased':>8} | {'Jaccard':>8}")
print("  " + "-" * 40)
for c in all_checkpoints:
    print(f"  {c['global_step']:>7} | {c['phase']:>5} | {c['aliased_count']:>8} | {c['jaccard_approx']:>8.3f}")

# Within-phase stability vs cross-phase change
print(f"\n  R3 dynamics measure_r3_dynamics (2K random, 10 ckpts)...")
class _674_chain(TransitionTriggered674):
    def __init__(self): super().__init__(n_actions=4, seed=0)

r3 = judge.measure_r3_dynamics(_674_chain, n_steps=2000, n_checkpoints=10)
print(f"  R3 score: {r3.get('r3_dynamic_score')} profile: {r3.get('dynamics_profile')}")

if all_checkpoints:
    # Phase-by-phase aliased growth
    for game in ["LS20", "FT09", "VC33"]:
        phase_ckpts = [c for c in all_checkpoints if c["phase"] == game]
        if phase_ckpts:
            max_a = max(c["aliased_count"] for c in phase_ckpts)
            avg_j = np.mean([c["jaccard_approx"] for c in phase_ckpts[1:]] or [0])
            print(f"  {game}: max_aliased={max_a} avg_within_jaccard={avg_j:.3f}")

print("\n" + "=" * 65)
print("STEP 727 DONE")
print("=" * 65)
