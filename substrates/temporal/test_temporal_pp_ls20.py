"""PredictPersist on LS20 50K. Full diagnostics per Spec."""
import sys, time
import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from temporal_pp import PredictPersist

MAX_STEPS = 50000
CHECKPOINTS = {1000, 5000, 10000, 25000, 50000}


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def svd_top3(W):
    _, S, _ = torch.linalg.svd(W.cpu())
    S = S.numpy()
    total_var = (S ** 2).sum()
    v1 = (S[0] ** 2) / total_var
    return S[:3], v1


def run():
    try:
        import arc_agi
        from arcengine import GameState
    except ImportError:
        print("SKIP: arc_agi not available")
        return

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found")
        return

    s = PredictPersist(d=256, n_actions=4)
    env = arc.make(ls20.game_id)
    obs = env.reset()

    steps = go = levels = 0
    unique = set()
    action_counts = [0] * 4
    pred_errs_window = []
    run_lengths = []
    cur_run = 0
    cur_action = 0
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
        pred_errs_window.append(s.pred_err)

        # Track run lengths
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
            print(f"LEVEL {levels} at step {steps}", flush=True)

        if steps in CHECKPOINTS:
            elapsed = time.time() - t0
            rank = torch.linalg.matrix_rank(s.W.cpu(), atol=1e-4).item()
            norm = s.W.norm().item()
            sv3, var1 = svd_top3(s.W)
            mean_err = float(np.mean(pred_errs_window[-1000:])) if pred_errs_window else 0
            mean_run = float(np.mean(run_lengths[-100:])) if run_lengths else 0
            max_run = max(run_lengths[-100:]) if run_lengths else 0
            total = sum(action_counts)
            dom = max(action_counts) / total * 100 if total else 0
            print(f"  step {steps}: unique={len(unique)}  levels={levels}  go={go}  dom={dom:.0f}%"
                  f"  acts={action_counts}  mean_run={mean_run:.1f}  max_run={max_run}"
                  f"  pred_err={mean_err:.4f}  W_rank={rank}  W_norm={norm:.2f}"
                  f"  SV3={[f'{v:.2f}' for v in sv3]}  timer_var={var1*100:.1f}%"
                  f"  {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total else 0
    mean_run = float(np.mean(run_lengths)) if run_lengths else 0
    max_run = max(run_lengths) if run_lengths else 0

    print(f"\nPredictPersist 50K: levels={levels}  unique={len(unique)}  go={go}  dom={dom:.0f}%")
    print(f"  action_counts={action_counts}  mean_run={mean_run:.1f}  max_run={max_run}")
    print(f"  {elapsed:.0f}s")
    print(f"\nPredictions: unique@10K=800-1200, unique@50K=2500-4000, level1=70%@20K-35K")
    print(f"Predictions: unique@10K=600-900,  unique@50K=2000-3500, level1=25%")
    print(f"Kill: unique<500 at 50K OR one action >90%")
    return levels


if __name__ == '__main__':
    run()
