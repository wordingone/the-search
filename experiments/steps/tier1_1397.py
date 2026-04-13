"""Tier 1 test for step 1397 — dolphin v2 (persist graph across GAME_OVER).
Games: SC25 (GAME_OVER-cycling, key variable), VC33 (passing control), WA30 (KB-only control).
10 seeds, 5K steps.
Leo mail 4550, 2026-04-13.

Kill: K_tier1 <= 1/3 → KILL v2 iteration.
Expected: VC33=10/10, WA30~0/10 (unchanged), SC25 improves vs v1.
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
from step1397_dolphin_v2 import DolphinV2

N_STEPS = 5000
N_SEEDS = 10
GAMES = ['SC25', 'VC33', 'WA30']

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
        sub = DolphinV2()
        sub.set_game(env.n_actions)

        obs = env.reset(seed=seed)
        level = 0
        max_level = 0
        episodes = 0
        nodes_at_end = 0

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
    print(f"{gname}: L1={l1_solved}/{N_SEEDS} ({l1_pct:.0%}) | avg_nodes={avg_nodes:.1f} | avg_eps={avg_eps:.1f} | n_actions={env.n_actions}")

# Summary
k = sum(1 for g in results.values() if g['l1_pct'] == 1.0)
print(f"\nK = {k}/{len(GAMES)}")
print(f"Kill criterion: K <= 1 → KILL | K >= 2 → continue")

# Save results
out_path = 'B:/M/the-search/chain_results/runs/tier1_1397_dolphin_v2.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {out_path}")
