"""LS20 10K: TemporalPrediction. Report levels, unique, pred_err, W_rank, W_norm at 1K/5K/10K."""
import sys, time
import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from temporal import TemporalPrediction

MAX_STEPS = 10000


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def w_stats(s):
    rank = torch.linalg.matrix_rank(s.W, atol=1e-4).item()
    norm = s.W.norm().item()
    return rank, norm


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

    s = TemporalPrediction(d=256, n_actions=4)
    env = arc.make(ls20.game_id)
    obs = env.reset()

    steps = go = levels = 0
    unique = set()
    action_counts = {}
    pred_errors = []
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
        pred_errors.append(s.pred_err)
        action = action_space[idx % n_acts]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1

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
            rank, norm = w_stats(s)
            recent_err = float(np.mean(pred_errors[-100:])) if pred_errors else 0.0
            print(f"  step {steps}: pred_err={recent_err:.4f}  W_rank={rank}  W_norm={norm:.3f}  unique={len(unique)}", flush=True)

    elapsed = time.time() - t0
    vals = list(action_counts.values())
    dom = max(vals) / sum(vals) * 100 if vals else 0
    acts_used = len(action_counts)
    rank, norm = w_stats(s)
    print(f"TemporalPrediction LS20: levels={levels}  unique={len(unique)}  go={go}  dom={dom:.0f}%  acts={acts_used}/4  W_rank={rank}  W_norm={norm:.3f}  {elapsed:.0f}s")
    print(f"SelfRef baseline:        levels=0          unique=1125          dom=40%  acts=4/4")
    return levels


if __name__ == '__main__':
    run()
