"""
Constraint Map Validation — 6 MinimalLVQ variants on LS20 50K.
Mail 1747: Jun-approved long run.

Tests whether 9 of 10 "unjustified" elements are actually forced.
Each variant changes exactly one element. Failure = element is forced (I).

Leo's predictions:
  Run 0 (baseline):     Level 1 ~26K
  Run 1 (L1 norm):      FAIL — timer dominates
  Run 2 (fixed lr=0.5): might navigate (weak kill)
  Run 3 (mean thresh):  navigates (mean ~= median)
  Run 4 (content act):  FAIL — first dims have no meaning
  Run 5 (lr=1-sim^2):   navigates (valid convex weight)
"""
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from minimal_lvq_variants import (
    MinimalLVQ,
    MinimalLVQ_L1,
    MinimalLVQ_FixedLR,
    MinimalLVQ_MeanThresh,
    MinimalLVQ_ContentAction,
    MinimalLVQ_LrSq,
)

MAX_STEPS = 50000
CHECKPOINTS = {10000, 25000, 50000}


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, V):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if V.shape[0] > 2:
        mean_V = V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit


def run_variant(run_id, name, substrate, arc, ls20_game):
    print(f"\n{'='*60}")
    print(f"RUN {run_id}: {name}")
    print(f"{'='*60}", flush=True)

    env = arc.make(ls20_game.game_id)
    obs = env.reset()

    steps = go = levels = 0
    unique = set()
    action_counts = [0] * 4
    run_lengths = []
    cur_run = 0
    cur_action = -1
    t0 = time.time()

    while steps < MAX_STEPS:
        if obs is None:
            obs = env.reset()
            if obs is None:
                break
            continue

        from arcengine import GameState
        if obs.state == GameState.GAME_OVER:
            go += 1
            obs = env.reset()
            if obs is None:
                break
            continue

        if obs.state == GameState.WIN:
            print(f"WIN at step {steps}!")
            break

        action_space = env.action_space
        n_acts = len(action_space)

        pooled = avgpool16(obs.frame)
        unique.add(hash(pooled.tobytes()))

        x = centered_enc(pooled, substrate.V)
        idx = substrate.step(x, n_actions=n_acts)

        if idx == cur_action:
            cur_run += 1
        else:
            if cur_run > 0:
                run_lengths.append(cur_run)
            cur_run = 1
            cur_action = idx

        action = action_space[idx % n_acts]
        action_counts[idx % n_acts] += 1

        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        steps += 1

        if obs is not None and obs.levels_completed > obs_before:
            levels = obs.levels_completed
            print(f"LEVEL {levels} at step {steps}  cb={substrate.V.shape[0]}", flush=True)

        if steps in CHECKPOINTS:
            elapsed = time.time() - t0
            total = sum(action_counts)
            dom = max(action_counts) / total * 100 if total else 0
            mean_run = float(np.mean(run_lengths[-100:])) if run_lengths else 0

            print(f"\n--- step {steps} | {elapsed:.0f}s ---")
            print(f"  unique={len(unique)}  levels={levels}  go={go}  dom={dom:.0f}%")
            print(f"  cb={substrate.V.shape[0]}  thresh={substrate.last_thresh:.4f}")
            print(f"  acts={action_counts}  mean_run={mean_run:.1f}", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total else 0
    baseline_unique = 1536  # Phase 1 LVQ baseline at 10K
    fail = len(unique) < baseline_unique * 0.5 and levels == 0

    print(f"\n--- FINAL {name} ---")
    print(f"  unique={len(unique)}  levels={levels}  go={go}  dom={dom:.0f}%")
    print(f"  cb={substrate.V.shape[0]}  elapsed={elapsed:.0f}s")
    print(f"  verdict: {'FAIL (element forced -> I)' if fail else 'PASS (element unjustified -> U)'}")
    return {
        'run': run_id,
        'name': name,
        'unique': len(unique),
        'levels': levels,
        'cb': substrate.V.shape[0],
        'dom': dom,
        'elapsed': elapsed,
        'fail': fail,
    }


def main():
    try:
        import arc_agi
    except ImportError:
        print("SKIP: arc_agi not available")
        return

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found")
        return

    print("Constraint Map Validation — 6 MinimalLVQ variants on LS20 50K")
    print(f"Baseline (Phase 1 LVQ): unique=1536 at 10K")
    print(f"Fail criterion: unique < 768 AND levels == 0 at 50K")

    variants = [
        (0, "MinimalLVQ (baseline)", MinimalLVQ(d=256)),
        (1, "MinimalLVQ_L1 (L1 norm)", MinimalLVQ_L1(d=256)),
        (2, "MinimalLVQ_FixedLR (lr=0.5)", MinimalLVQ_FixedLR(d=256)),
        (3, "MinimalLVQ_MeanThresh (mean)", MinimalLVQ_MeanThresh(d=256)),
        (4, "MinimalLVQ_ContentAction (V[w][:n])", MinimalLVQ_ContentAction(d=256)),
        (5, "MinimalLVQ_LrSq (1-sim^2)", MinimalLVQ_LrSq(d=256)),
    ]

    results = []
    for run_id, name, substrate in variants:
        r = run_variant(run_id, name, substrate, arc, ls20)
        results.append(r)

    print(f"\n{'='*60}")
    print("SUMMARY — Constraint Map Validation")
    print(f"{'='*60}")
    print(f"{'Run':<4} {'Name':<35} {'unique':>7} {'levels':>7} {'cb':>6} {'dom':>6} {'verdict'}")
    print(f"{'-'*80}")
    for r in results:
        verdict = "FAIL->I" if r['fail'] else "PASS->U"
        print(f"{r['run']:<4} {r['name']:<35} {r['unique']:>7} {r['levels']:>7} {r['cb']:>6} {r['dom']:>5.0f}% {verdict}")

    forced = [r for r in results if r['fail']]
    unjustified = [r for r in results if not r['fail']]
    print(f"\nForced elements (U->I): {len(forced)}/6 runs fail")
    print(f"Unjustified elements (confirmed U): {len(unjustified)}/6 runs pass")
    if results[0]['fail']:
        print("WARNING: Run 0 (baseline) failed. Constraint analysis is invalid.")
    elif results[0]['levels'] > 0:
        print(f"Run 0 navigated Level {results[0]['levels']}. Baseline confirmed.")


if __name__ == '__main__':
    main()
