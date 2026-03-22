"""
10K LS20 test: SelfRef (argmax) vs SelfRefArgmin (argmin). One line diff.
Within 5-min cap.
"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from selfref import SelfRef
from selfref_argmin import SelfRefArgmin

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_STEPS = 10000


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


def run_ls20(substrate_cls, label):
    try:
        import arc_agi
        from arcengine import GameState
    except ImportError:
        print(f"  {label}: SKIP (arc_agi not available)")
        return None

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if ls20 is None:
        return None

    s = substrate_cls(d=256, device=DEVICE)
    env = arc.make(ls20.game_id)
    obs = env.reset()

    total_steps = 0
    game_over_count = 0
    total_levels = 0
    unique_states = set()
    action_counts = {}
    t0 = time.time()

    while total_steps < MAX_STEPS:
        if obs is None:
            obs = env.reset()
            if obs is None: break
            continue
        if obs.state == GameState.GAME_OVER:
            game_over_count += 1
            obs = env.reset()
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            print(f"  [{label}] WIN at step {total_steps}!")
            break

        action_space = env.action_space
        n_acts = len(action_space)

        pooled = avgpool16(obs.frame)
        enc = centered_enc(pooled, s.V)
        unique_states.add(hash(pooled.tobytes()))

        action_idx = s.step(enc, n_actions=n_acts)
        action = action_space[action_idx % n_acts]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1

        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        total_steps += 1

        if obs is not None and obs.levels_completed > obs_before:
            total_levels = obs.levels_completed
            print(f"  [{label}] LEVEL {total_levels} at step {total_steps}  cb={s.V.shape[0]}", flush=True)

    elapsed = time.time() - t0
    vals = list(action_counts.values())
    dom = max(vals) / sum(vals) * 100 if vals else 0
    print(f"  {label}: levels={total_levels}  cb={s.V.shape[0]}  unique={len(unique_states)}"
          f"  go={game_over_count}  dom={dom:.0f}%  {elapsed:.0f}s")
    return total_levels


print(f"LS20 10K: argmax vs argmin  device={DEVICE}")
print()
r_argmax = run_ls20(SelfRef, "argmax (original)")
r_argmin = run_ls20(SelfRefArgmin, "argmin (exploration)")
print()
print(f"argmax={r_argmax}  argmin={r_argmin}")
if r_argmin is not None and r_argmax is not None:
    print(f"Hypothesis: {'CONFIRMED' if r_argmin > r_argmax else 'REFUTED'}")
