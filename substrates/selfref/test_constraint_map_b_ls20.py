"""
Constraint Map Validation Round B — SelfRef (depth 2) baseline + 5 variants.
Mail 1749: Approved long run.

Run 0b: SelfRef exact — MUST navigate Level 1
Run 1b: L1 norm at depth 2
Run 2b: fixed lr=0.5 at depth 2
Run 3b: SKIP (mean=median confirmed)
Run 4b: content-based action at depth 2
Run 5b: lr=1-sim^2 at depth 2 (KEY RUN — is lr formula forced?)
Run 6b: depth 1 + random non-winner attract (isolate chain selection)

Predictions:
  0b: Level 1 (must navigate)
  1b: fails (L1 catastrophic)
  2b: fails (adaptive lr drives growth)
  4b: fails (content-based fails)
  5b: KEY — if navigates, 1-sim is genuine U
  6b: fails (random != chain-selected)
"""
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from selfref import SelfRef
from selfref_b_variants import (
    SelfRef_L1,
    SelfRef_FixedLR,
    SelfRef_ContentAction,
    SelfRef_LrSq,
    SelfRef_Depth1RandAttract,
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
    last_thresh = 0.0

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

        if hasattr(substrate, 'last_thresh'):
            last_thresh = substrate.last_thresh

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
            print(f"  cb={substrate.V.shape[0]}  thresh={last_thresh:.4f}")
            print(f"  acts={action_counts}  mean_run={mean_run:.1f}", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total else 0

    # Fail: unique < 50% of SelfRef baseline AND no levels
    baseline_unique = 1125  # SelfRef at 10K from memory
    fail = len(unique) < baseline_unique * 0.5 and levels == 0

    print(f"\n--- FINAL {name} ---")
    print(f"  unique={len(unique)}  levels={levels}  go={go}  dom={dom:.0f}%")
    print(f"  cb={substrate.V.shape[0]}  elapsed={elapsed:.0f}s")
    verdict = "FAIL (forced->I)" if fail else ("PASS->U (unjustified)" if levels == 0 else f"NAVIGATE (Level {levels})")
    print(f"  verdict: {verdict}")
    return {
        'run': run_id, 'name': name,
        'unique': len(unique), 'levels': levels,
        'cb': substrate.V.shape[0], 'dom': dom,
        'elapsed': elapsed, 'fail': fail,
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

    print("Constraint Map Validation Round B -- SelfRef (depth 2) baseline + 5 variants")
    print("Fail criterion: unique < 562 AND levels == 0 at 50K")

    variants = [
        ("0b", "SelfRef (depth 2 baseline)", SelfRef(d=256)),
        ("1b", "SelfRef_L1 (L1 norm depth 2)", SelfRef_L1(d=256)),
        ("2b", "SelfRef_FixedLR (lr=0.5 depth 2)", SelfRef_FixedLR(d=256)),
        ("4b", "SelfRef_ContentAction (V[w1][:n])", SelfRef_ContentAction(d=256)),
        ("5b", "SelfRef_LrSq (1-sim^2 depth 2)", SelfRef_LrSq(d=256)),
        ("6b", "SelfRef_Depth1RandAttract", SelfRef_Depth1RandAttract(d=256)),
    ]

    results = []
    for run_id, name, substrate in variants:
        r = run_variant(run_id, name, substrate, arc, ls20)
        results.append(r)

    print(f"\n{'='*60}")
    print("SUMMARY -- Constraint Map Validation Round B")
    print(f"{'='*60}")
    print(f"{'Run':<4} {'Name':<40} {'unique':>7} {'levels':>7} {'cb':>6} {'dom':>6} {'verdict'}")
    print(f"{'-'*85}")
    for r in results:
        if r['levels'] > 0:
            verdict = f"NAVIGATE L{r['levels']}"
        elif r['fail']:
            verdict = "FAIL->I"
        else:
            verdict = "PASS->U"
        print(f"{r['run']:<4} {r['name']:<40} {r['unique']:>7} {r['levels']:>7} {r['cb']:>6} {r['dom']:>5.0f}% {verdict}")

    if results[0]['levels'] == 0:
        print("\nWARNING: Run 0b (baseline) did not navigate. Analysis invalid.")
    else:
        print(f"\nRun 0b navigated Level {results[0]['levels']}. Baseline confirmed.")
        forced = sum(1 for r in results[1:] if r['fail'])
        print(f"Forced elements confirmed: {forced}/5")


if __name__ == '__main__':
    main()
