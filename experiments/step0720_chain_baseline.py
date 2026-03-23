"""
Step 720 (A1): 674+running-mean full chain baseline.

R3 hypothesis: 674 achieves partial R3 via transition-triggered refinement.
Measure R3 dynamics across chain with state checkpoints every 1000 steps.

Chain: Split-CIFAR-100 -> LS20 -> FT09 -> VC33 -> Split-CIFAR-100 (backward).
5 min per phase, 25 min total (Leo-approved for chain benchmark).
1 seed. Track aliased_cells, live_count, G_size at each 1000-step checkpoint.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates import ConstitutionalJudge
from substrates.step0674 import TransitionTriggered674
from substrates.chain import SplitCIFAR100Wrapper

print("=" * 65)
print("STEP 720 (A1) — 674 FULL CHAIN BASELINE")
print("=" * 65)

SEED = 0
PER_PHASE_TIME = 300   # 5 min per phase
N_STEPS_GAME = 10_000  # game steps per phase
N_STEPS_CIFAR = 500    # images per CIFAR task


def _make_env(game):
    try:
        import arcagi3
        return arcagi3.make(game)
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game)


def run_game_tracked(env, substrate, n_steps, per_seed_time, checkpoint_every=1000):
    """Run game with state snapshots every checkpoint_every steps."""
    checkpoints = []
    obs_seq = []
    obs = env.reset(seed=SEED)
    level = 0
    l1_step = l2_step = None
    steps = 0
    fresh = True
    n_valid = len(env._action_space) if hasattr(env, '_action_space') else 4
    t_start = time.time()

    while steps < n_steps and (time.time() - t_start) < per_seed_time:
        if obs is None:
            obs = env.reset(seed=SEED)
            substrate.on_level_transition()
            fresh = True
            continue

        obs_arr = np.array(obs, dtype=np.float32)
        obs_seq.append(obs_arr)
        action = substrate.process(obs_arr)
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
            substrate.on_level_transition()

        if done:
            obs = env.reset(seed=SEED)
            substrate.on_level_transition()
            fresh = True

        if steps % checkpoint_every == 0:
            state = substrate.get_state()
            checkpoints.append({
                "step": steps,
                "aliased_count": state.get("aliased_count", 0),
                "live_count": state.get("live_count", 0),
                "G_size": state.get("G_size", 0),
                "ref_count": state.get("ref_count", 0),
            })

    elapsed = time.time() - t_start
    return {
        "steps": steps,
        "elapsed": round(elapsed, 2),
        "l1": l1_step,
        "l2": l2_step,
        "level_reached": level,
        "checkpoints": checkpoints,
        "obs_sequence": obs_seq,
    }


# --- Phase 1: CIFAR before ---
print("\n-- Phase 1: Split-CIFAR-100 (before) --")
wrapper = SplitCIFAR100Wrapper(n_images_per_task=N_STEPS_CIFAR)
if not wrapper._load():
    print("  ERROR: CIFAR-100 not available")
    cifar_before = {"avg_accuracy": None, "task_accuracies": []}
else:
    sub_cifar = TransitionTriggered674(n_actions=5, seed=SEED)
    cifar_before = wrapper.run_seed(sub_cifar, SEED)
    print(f"  avg_accuracy={cifar_before.get('avg_accuracy')} "
          f"tasks={cifar_before.get('tasks_completed')} "
          f"elapsed={cifar_before.get('elapsed')}s")

# --- Phases 2-4: Games ---
game_results = {}
judge = ConstitutionalJudge()

for game in ["LS20", "FT09", "VC33"]:
    print(f"\n-- Phase: {game} --")
    try:
        env = _make_env(game)
        sub_game = TransitionTriggered674(n_actions=4, seed=SEED)
        sub_game.reset(SEED)

        result = run_game_tracked(env, sub_game, N_STEPS_GAME, PER_PHASE_TIME)
        game_results[game] = result

        print(f"  steps={result['steps']} elapsed={result['elapsed']}s "
              f"l1={result['l1']} l2={result['l2']} level={result['level_reached']}")

        # Checkpoint table
        print(f"  {'Step':>6} | {'Aliased':>8} | {'Live':>6} | {'G':>7} | {'Ref':>5}")
        for ckpt in result["checkpoints"]:
            print(f"  {ckpt['step']:>6} | {ckpt['aliased_count']:>8} | "
                  f"{ckpt['live_count']:>6} | {ckpt['G_size']:>7} | {ckpt['ref_count']:>5}")

        # R3 dynamics (using collected obs_sequence)
        print(f"\n  measure_r3_dynamics ({min(len(result['obs_sequence']), 2000)} obs, 10 ckpts)...")
        obs_for_r3 = result["obs_sequence"][:2000]
        n_valid_g = len(env._action_space) if hasattr(env, '_action_space') else 4

        class _674_game(TransitionTriggered674):
            def __init__(self): super().__init__(n_actions=n_valid_g, seed=0)

        r3_dyn = judge.measure_r3_dynamics(
            _674_game,
            obs_sequence=obs_for_r3,
            n_steps=len(obs_for_r3),
            n_checkpoints=10
        )
        game_results[game]["r3_dynamics"] = r3_dyn
        print(f"  R3 dynamic score: {r3_dyn.get('r3_dynamic_score')} "
              f"profile: {r3_dyn.get('dynamics_profile')}")
        print(f"  Verified M: {r3_dyn.get('verified_M_elements')}")

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        game_results[game] = {"error": str(e)}

# --- Phase 5: CIFAR after ---
print("\n-- Phase 5: Split-CIFAR-100 (after) --")
if wrapper._data is not None:
    sub_cifar_after = TransitionTriggered674(n_actions=5, seed=SEED)
    cifar_after = wrapper.run_seed(sub_cifar_after, SEED)
    backward_transfer = None
    if cifar_before.get("avg_accuracy") and cifar_after.get("avg_accuracy"):
        backward_transfer = round(cifar_after["avg_accuracy"] - cifar_before["avg_accuracy"], 4)
    print(f"  avg_accuracy={cifar_after.get('avg_accuracy')} "
          f"elapsed={cifar_after.get('elapsed')}s")
    print(f"  CIFAR backward transfer (after chain): {backward_transfer}")
else:
    cifar_after = {"avg_accuracy": None}
    backward_transfer = None

# --- Summary ---
print("\n" + "=" * 65)
print("CHAIN SUMMARY")
print("=" * 65)
print(f"CIFAR before:     avg_acc={cifar_before.get('avg_accuracy')}")
for game, r in game_results.items():
    if "error" not in r:
        r3_score = r.get("r3_dynamics", {}).get("r3_dynamic_score", "N/A")
        print(f"{game:<10}: l1={r.get('l1')} l2={r.get('l2')} "
              f"steps={r.get('steps')} R3_dyn={r3_score}")
    else:
        print(f"{game:<10}: ERROR — {r['error'][:50]}")
print(f"CIFAR after:      avg_acc={cifar_after.get('avg_accuracy')}")
print(f"Chain BWT:        {backward_transfer}")

# R1 audit
print("\n-- Static R3 audit --")
audit = judge.audit(TransitionTriggered674, chain_results=None, n_audit_steps=200)
r3_static = audit.get("R3", {})
print(f"R3 static: M={r3_static.get('M_count')} I={r3_static.get('I_count')} "
      f"U={r3_static.get('U_count')} verdict={audit['summary']['verdict']}")

print("=" * 65)
print("STEP 720 DONE")
print("=" * 65)
