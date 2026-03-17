"""LS20 10K: ExprU20 (temporal_smoothness x coverage). 5-min cap."""
import sys, time
import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from expr_u20 import ExprU20

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

    s = ExprU20(n_dims=256, n_actions=4)
    env = arc.make(ls20.game_id)
    obs = env.reset()

    steps = go = levels = 0
    unique = set()
    action_counts = {}
    smoothness_samples = []
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
        idx = s.step(x, n_actions=n_acts)
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
            print(f"LEVEL {levels} at step {steps}  size={s.size}", flush=True)

        # Track smoothness every 1000 steps
        if steps % 1000 == 0 and len(s.history) > 1:
            from expr import evaluate
            recent = s.history[-s.window:]
            acts = [evaluate(s.pop[s.best], o) % n_acts for o, _ in recent]
            same = sum(a == b for a, b in zip(acts[:-1], acts[1:]))
            sm = same / max(len(acts) - 1, 1)
            smoothness_samples.append(sm)
            print(f"  step {steps}: smoothness={sm:.2f}  scores={[f'{x:.2f}' for x in s.scores]}"
                  f"  unique={len(unique)}", flush=True)

    elapsed = time.time() - t0
    vals = list(action_counts.values())
    dom = max(vals) / sum(vals) * 100 if vals else 0
    acts_used = len(action_counts)
    avg_sm = float(np.mean(smoothness_samples)) if smoothness_samples else 0.0
    print(f"ExprU20:  levels={levels}  size={s.size}  unique={len(unique)}"
          f"  go={go}  dom={dom:.0f}%  acts={acts_used}/4  sm_avg={avg_sm:.2f}  {elapsed:.0f}s")
    print(f"Baseline: levels=0    cb=164    unique=1125  go=?   dom=40%  acts=4/4")
    return levels


if __name__ == '__main__':
    run()
