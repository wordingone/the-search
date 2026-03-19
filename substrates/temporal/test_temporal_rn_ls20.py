"""TemporalRowNorm on LS20 50K. Decisive test for temporal family.

SVD of both W (raw) and W_normalized at each checkpoint.
Timer variance fraction in normalized W.
"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from temporal_rn import TemporalRowNorm

MAX_STEPS = 50000
CHECKPOINTS = {1000, 5000, 10000, 25000, 50000}
TIMER_ROWS = slice(208, 256)  # rows 13-15 of 16x16 = timer/score region


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def svd_diag(W, label, top_n=5):
    """SVD diagnostics for a matrix. Returns timer_var_frac of top SV input dir."""
    U, S, Vh = torch.linalg.svd(W.cpu())
    S = S.numpy()
    total_var = (S ** 2).sum()
    sv_strs = [f'{v:.3f}' for v in S[:top_n]]
    var_fracs = [(S[i]**2)/total_var*100 for i in range(min(top_n, len(S)))]
    var_strs = [f'{v:.1f}%' for v in var_fracs]
    print(f"  {label} SVD: [{', '.join(sv_strs)}]  var=[{', '.join(var_strs)}]")

    # Timer variance in top-1 input direction (row of Vh = right singular vector)
    v1 = Vh[0].numpy()  # shape (d,) — input direction
    v1_energy = v1 ** 2
    timer_frac = v1_energy[TIMER_ROWS].sum() / v1_energy.sum()
    print(f"  {label} SV1 timer_frac={timer_frac*100:.1f}%  "
          f"(rows 13-15 energy vs total)")
    return timer_frac


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

    s = TemporalRowNorm(d=256, n_actions=4)
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
            mean_err = float(np.mean(pred_errs_window[-1000:])) if pred_errs_window else 0
            mean_run = float(np.mean(run_lengths[-100:])) if run_lengths else 0
            max_run = max(run_lengths[-100:]) if run_lengths else 0
            total = sum(action_counts)
            dom = max(action_counts) / total * 100 if total else 0

            print(f"\n--- step {steps} | {elapsed:.0f}s ---")
            print(f"  unique={len(unique)}  levels={levels}  go={go}  dom={dom:.0f}%")
            print(f"  acts={action_counts}  mean_run={mean_run:.1f}  max_run={max_run}")
            print(f"  pred_err={mean_err:.4f}")

            # SVD of W (raw)
            svd_diag(s.W, "W_raw", top_n=5)

            # SVD of W_normalized
            W_n = F.normalize(s.W.cpu(), dim=1)
            svd_diag(W_n, "W_norm", top_n=5)

    elapsed = time.time() - t0
    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total else 0
    mean_run = float(np.mean(run_lengths)) if run_lengths else 0
    max_run = max(run_lengths) if run_lengths else 0

    print(f"\n{'='*60}")
    print(f"TemporalRowNorm 50K: levels={levels}  unique={len(unique)}  go={go}  dom={dom:.0f}%")
    print(f"  action_counts={action_counts}  mean_run={mean_run:.1f}  max_run={max_run}")
    print(f"  {elapsed:.0f}s")
    print(f"\nLeo prediction: 60% NO navigation, 40% qualitative improvement")
    print(f"Kill: unique <= LVQ baseline AND 0 levels @ 50K → temporal family dead")
    print(f"Live: Level 1 OR clearly spatial coverage pattern")
    return levels


if __name__ == '__main__':
    run()
