"""
SVD(W) diagnostic on LS20 baseline TemporalPrediction.
At steps 1K/5K/10K: report top-10 singular values, variance fractions,
and spatial structure of top-3 input/output directions (16x16 heatmaps).

the prediction: top-1 timer-dominated (>30% variance), top 4-10 sprite-region.
"""
import sys, time, os
import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from temporal import TemporalPrediction

MAX_STEPS = 10000
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'svd_diagnostics')
os.makedirs(SAVE_DIR, exist_ok=True)


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def svd_report(W, step):
    """Compute SVD of W (d×d), report top-10 singular values and variance fractions."""
    U, S, Vh = torch.linalg.svd(W.cpu())
    S = S.numpy()
    U = U.numpy()
    Vh = Vh.numpy()

    total_var = (S ** 2).sum()
    var1 = (S[0] ** 2) / total_var
    var3 = (S[:3] ** 2).sum() / total_var
    var10 = (S[:10] ** 2).sum() / total_var

    print(f"\n--- SVD at step {step} ---")
    print(f"  Top-10 singular values: {[f'{v:.3f}' for v in S[:10]]}")
    print(f"  Variance fraction: top-1={var1*100:.1f}%  top-3={var3*100:.1f}%  top-10={var10*100:.1f}%")

    # Input directions (rows of Vh = right singular vectors)
    # Output directions (columns of U = left singular vectors)
    for k in range(3):
        v_dir = Vh[k].reshape(16, 16)   # input direction k
        u_dir = U[:, k].reshape(16, 16)  # output direction k

        # Print row sums (shows which rows carry most energy)
        v_row_energy = (v_dir ** 2).sum(axis=1)
        u_row_energy = (u_dir ** 2).sum(axis=1)

        v_peak_row = v_row_energy.argmax()
        u_peak_row = u_row_energy.argmax()

        print(f"  SV{k+1} (S={S[k]:.3f}):")
        print(f"    Input dir  peak row={v_peak_row}  row_energy={[f'{r:.3f}' for r in v_row_energy[:4]]}...{[f'{r:.3f}' for r in v_row_energy[12:]]}")
        print(f"    Output dir peak row={u_peak_row}  row_energy={[f'{r:.3f}' for r in u_row_energy[:4]]}...{[f'{r:.3f}' for r in u_row_energy[12:]]}")

        # Save arrays
        np.save(os.path.join(SAVE_DIR, f'v_dir_step{step}_k{k+1}.npy'), v_dir)
        np.save(os.path.join(SAVE_DIR, f'u_dir_step{step}_k{k+1}.npy'), u_dir)

    print(f"  (Saved top-3 U/V dirs to {SAVE_DIR}/)")
    return var1, var3, var10


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

    s = TemporalPrediction(d=256, n_actions=4)
    env = arc.make(ls20.game_id)
    obs = env.reset()

    steps = go = levels = 0
    unique = set()
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
            svd_report(s.W, steps)
            print(f"  unique={len(unique)}  pred_err={s.pred_err:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"\nFinal: levels={levels}  unique={len(unique)}  {elapsed:.0f}s")
    print(f"Prediction: top-1 >30% variance (timer), top-4..10 sprite region")


if __name__ == '__main__':
    run()
