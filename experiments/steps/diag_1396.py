"""Diagnostic for dolphin v1 failures — unique frames and graph growth per game."""
import os, sys, json

sys.path.insert(0, 'B:/M/the-search/experiments/steps')
sys.path.insert(0, 'B:/M/the-search/experiments/environments')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search')

with open(r'C:\Users\Admin\.secrets\.env') as f:
    for line in f:
        if line.strip().startswith('ARC_API_KEY='):
            os.environ['ARC_API_KEY'] = line.strip().split('=', 1)[1].strip()

import arc_agi
import numpy as np
from util_arcagi3 import _Env
from step1396_dolphin_explorer import DolphinExplorer

N_STEPS = 1000

games_to_test = [
    ('R11L', 'failed'), ('SB26', 'failed'), ('TN36', 'failed'),
    ('WA30', 'failed'), ('TU93', 'failed'), ('DC22', 'failed'),
    ('G50T', 'failed'), ('SC25', 'failed'), ('SU15', 'failed'),
    ('RE86', 'failed'),
    ('VC33', 'pass'), ('FT09', 'pass'), ('LP85', 'pass'),
]

arc = arc_agi.Arcade()
games_info = arc.get_environments()

print(f"{'Game':6s} | {'status':6s} | {'n_act':5s} | {'click':5s} | {'f@500':6s} | {'f@1K':5s} | {'new_last200':11s} | {'L1':3s}")
print('-' * 70)

for gname, status in games_to_test:
    key = gname.lower()
    try:
        info = next(g for g in games_info if key in g.game_id.lower())
        env = _Env(info.game_id)
        sub = DolphinExplorer()
        sub.set_game(env.n_actions)

        obs = env.reset(seed=0)
        level = 0
        frame_counts = []

        for step in range(N_STEPS):
            if obs is None:
                obs = env.reset(seed=0)
                sub.on_level_transition()
                continue

            arr = np.array(obs, dtype=np.float32)
            action = sub.process(arr)
            obs, reward, done, info_d = env.step(action % env.n_actions)

            frame_counts.append(len(sub._nodes))

            cl = info_d.get('level', 0) if isinstance(info_d, dict) else 0
            if cl > level:
                level = cl
            if done:
                obs = env.reset(seed=0)
                sub.on_level_transition()

        n500 = frame_counts[499] if len(frame_counts) > 499 else '?'
        n1k  = frame_counts[-1] if frame_counts else 0
        new_last200 = frame_counts[-1] - frame_counts[-201] if len(frame_counts) >= 201 else '?'
        print(f"{gname:6s} | {status:6s} | {env.n_actions:5d} | {str(env._supports_click):5s} | {n500!s:6s} | {n1k!s:5s} | {new_last200!s:11s} | {level>0}")
    except Exception as e:
        print(f"{gname}: ERROR {e}")
