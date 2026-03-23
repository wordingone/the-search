"""
Step 766 - Random agent baseline on Atari 100K.

Control baseline for Step 763 (674 on Atari 100K). Random uniform action
selection. Same 26 games, same protocol (1 seed, 100K steps, 5 min cap).

R3 hypothesis: N/A — pure control. Establishes random floor for comparison.
Expected: lower total reward than any structured agent.
Compare: Step 763 (674 R1 mode). If 674 ≈ random: no exploration structure.
If 674 > random: self-organization finds reward even without reward signal.

26 standard Atari 100K games (Bellemare et al. 2013 / Kaiser et al. 2019).
Budget: 5 min per game (100K steps). 1 seed per game.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np

print("=" * 65)
print("STEP 766 - RANDOM BASELINE ON ATARI 100K (26 GAMES)")
print("=" * 65)

ATARI_100K_GAMES = [
    "Alien", "Amidar", "Assault", "Asterix", "BankHeist",
    "BattleZone", "Boxing", "Breakout", "ChopperCommand", "CrazyClimber",
    "DemonAttack", "Freeway", "Frostbite", "Gopher", "Hero",
    "Jamesbond", "Kangaroo", "Krull", "KungFuMaster", "MsPacman",
    "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"
]

N_SEEDS = 1
N_STEPS = 100_000
PER_GAME_TIME = 300   # 5 min max

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)


def run_random_game(game_name, seed=0, n_steps=N_STEPS, per_game_time=PER_GAME_TIME):
    """Random agent on one Atari game."""
    results = []
    env_id = f"ALE/{game_name}-v5"
    try:
        env = gym.make(env_id, render_mode=None, frameskip=4)
        n_actions = env.action_space.n
        rng = np.random.RandomState(seed)
        obs, info = env.reset(seed=seed)
        steps = 0; total_reward = 0.0
        unique_obs = set()
        t_start = time.time()
        while steps < n_steps and (time.time() - t_start) < per_game_time:
            action = rng.randint(0, n_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if steps % 100 == 0:
                h = hash(np.array(obs, dtype=np.float32).tobytes()[:64])
                unique_obs.add(h)
            if terminated or truncated:
                obs, info = env.reset(seed=seed)
        env.close()
        results.append({
            "seed": seed, "steps": steps,
            "total_reward": total_reward,
            "unique_states": len(unique_obs),
            "elapsed": round(time.time() - t_start, 1)
        })
        print(f"    seed={seed}: steps={steps} reward={total_reward:.0f} unique={len(unique_obs)}")
    except Exception as e:
        print(f"    seed={seed}: ERROR {e}")
        results.append({"seed": seed, "steps": 0, "total_reward": 0, "error": str(e)})
    return results


all_results = {}

for game in ATARI_100K_GAMES:
    print(f"\n-- {game} --")
    r = run_random_game(game, seed=0)
    all_results[game] = r
    valid = [x for x in r if "total_reward" in x and not x.get("error")]
    if valid:
        avg_r = float(np.mean([x["total_reward"] for x in valid]))
        avg_u = float(np.mean([x["unique_states"] for x in valid]))
        print(f"  {game}: avg_reward={avg_r:.1f} avg_unique={avg_u:.0f}")

print("\n" + "=" * 65)
print("STEP 766 SUMMARY - RANDOM BASELINE ON ATARI 100K")
print("=" * 65)
print(f"{'Game':<20} {'Reward (random)':>16} {'Unique States':>14}")
print("-" * 54)
for game in ATARI_100K_GAMES:
    r = all_results.get(game, [])
    valid = [x for x in r if "total_reward" in x and not x.get("error")]
    if valid:
        avg_r = float(np.mean([x["total_reward"] for x in valid]))
        avg_u = float(np.mean([x["unique_states"] for x in valid]))
        print(f"{game:<20} {avg_r:>16.1f} {avg_u:>14.0f}")
    else:
        print(f"{game:<20} {'ERROR':>16} {'N/A':>14}")
print("-" * 54)
print("Compare to Step 763 (674 R1 mode) for same games.")
print("Human-normalized score (HNS) = (agent - random) / (human - random).")
print("EfficientZero median HNS: 249%. We're establishing R1 floor.")
print("=" * 65)
print("STEP 766 DONE")
print("=" * 65)
