"""Tier 2 test for step 1398 — dolphin v3 (change-based priority).
All 25 API games, 10 seeds, 5K steps.
Leo mail 4557, 2026-04-13.

Same config as v1 Tier 2 (2026-04-13_1396_DolphinExplorer.json) for direct comparison.
Kill: K < 4/25 (v1 baseline). Flag: K >= 6/25 (new Kaggle baseline candidate).
Growing games of interest: R11L, SB26, WA30, G50T, RE86.
"""
import os, sys, json, time

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
from step1398_dolphin_v3 import DolphinV3

N_STEPS = 5000
N_SEEDS = 10
# All 25 API games (v1 order, MBPP excluded)
GAMES = ['LP85', 'SP80', 'R11L', 'S5I5', 'SB26', 'TN36', 'LS20', 'WA30', 'VC33', 'TU93',
         'CN04', 'CD82', 'M0R0', 'BP35', 'TR87', 'FT09', 'SK48', 'AR25', 'DC22', 'G50T',
         'KA59', 'SC25', 'LF52', 'SU15', 'RE86']

GROWING_GAMES = {'R11L', 'SB26', 'WA30', 'G50T', 'RE86'}

arc = arc_agi.Arcade()
games_info = arc.get_environments()

results = {}

for gname in GAMES:
    key = gname.lower()
    info = next(g for g in games_info if key in g.game_id.lower())
    env = _Env(info.game_id)

    l1_solved = 0
    seed_results = []

    for seed in range(N_SEEDS):
        sub = DolphinV3()
        sub.set_game(env.n_actions)

        obs = env.reset(seed=seed)
        max_level = 0
        episodes = 0

        for step in range(N_STEPS):
            if obs is None:
                obs = env.reset(seed=seed)
                sub.on_level_transition()
                episodes += 1
                continue

            arr = np.array(obs, dtype=np.float32)
            action = sub.process(arr)
            obs, reward, done, info_d = env.step(action % env.n_actions)

            cl = info_d.get('level', 0) if isinstance(info_d, dict) else 0
            if cl > max_level:
                max_level = cl
                sub.on_level_transition()

            if done:
                obs = env.reset(seed=seed)
                sub.on_level_transition()
                episodes += 1

        nodes_at_end = len(sub._nodes)
        solved = (max_level >= 1)
        if solved:
            l1_solved += 1

        seed_results.append({
            'seed': seed,
            'max_level': max_level,
            'solved': solved,
            'n_nodes': nodes_at_end,
            'episodes': episodes,
        })

    l1_pct = l1_solved / N_SEEDS
    results[gname] = {
        'l1_solved': l1_solved,
        'l1_pct': l1_pct,
        'n_actions': env.n_actions,
        'seeds': seed_results,
    }
    avg_nodes = sum(s['n_nodes'] for s in seed_results) / N_SEEDS
    avg_eps = sum(s['episodes'] for s in seed_results) / N_SEEDS
    flag = ' [GROWING]' if gname in GROWING_GAMES else ''
    print(f"{gname}: L1={l1_solved}/{N_SEEDS} ({l1_pct:.0%}) | avg_nodes={avg_nodes:.1f} | avg_eps={avg_eps:.1f}{flag}")

k = sum(1 for g in results.values() if g['l1_pct'] == 1.0)
k_any = sum(1 for g in results.values() if g['l1_solved'] > 0)
growing_wins = {g: results[g]['l1_solved'] for g in GROWING_GAMES if g in results}
print(f"\nK (100%) = {k}/{len(GAMES)}")
print(f"K (any)  = {k_any}/{len(GAMES)}")
print(f"Growing game wins: {growing_wins}")
print(f"v1 baseline: K=4/25 (LP85, SP80, VC33, FT09)")
print(f"Flag for Kaggle: K >= 6/25")

out_path = 'B:/M/the-search/chain_results/runs/tier2_1398_dolphin_v3.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump({'results': results, 'k_100pct': k, 'k_any': k_any}, f, indent=2)
print(f"Results saved: {out_path}")
