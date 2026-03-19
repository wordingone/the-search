"""LS20 10K: TemporalPerAction. All diagnostics per Leo's spec."""
import sys, time
import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from temporal_pa import TemporalPerAction

MAX_STEPS = 10000


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def run():
    try:
        import arc_agi
        from arcengine import GameState
    except ImportError:
        print("SKIP: arc_agi not available")
        return None

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found")
        return None

    s = TemporalPerAction(d=256, n_actions=4)
    env = arc.make(ls20.game_id)
    obs = env.reset()

    steps = go = levels = 0
    unique = set()
    action_counts = [0] * 4
    action_history = []
    t0 = time.time()

    while steps < MAX_STEPS:
        if obs is None:
            obs = env.reset()
            if obs is None:
                break
            continue
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

        x = torch.from_numpy(pooled.astype(np.float32))
        idx = s.step(x)
        action = action_space[idx % n_acts]
        action_counts[idx % n_acts] += 1
        action_history.append(idx % n_acts)

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
            print(f"LEVEL {levels} at step {steps}", flush=True)

        if steps in (1000, 5000, 10000):
            errs = [f"{e:.4f}" for e in s.pred_err]
            print(f"  step {steps}: unique={len(unique)}  pred_err={errs}  acts={action_counts}", flush=True)

    elapsed = time.time() - t0

    # Directional runs: count sequences of >5 consecutive same action
    runs_gt5 = 0
    cur_run = 1
    for i in range(1, len(action_history)):
        if action_history[i] == action_history[i-1]:
            cur_run += 1
        else:
            if cur_run > 5:
                runs_gt5 += 1
            cur_run = 1
    if cur_run > 5:
        runs_gt5 += 1

    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total > 0 else 0
    print(f"TemporalPerAction LS20: levels={levels}  unique={len(unique)}  go={go}  dom={dom:.0f}%  acts={action_counts}  runs>5={runs_gt5}  {elapsed:.0f}s")
    print(f"Baseline (1 W):         levels=0          unique=1536          dom=54%  acts=4/4")
    print(f"Leo kill criterion:     unique <= 1536 → kill")
    return levels


if __name__ == '__main__':
    run()
