"""
Step 763 - 674+running-mean on Atari 100K benchmark.

Standard Atari 100K: 26 games, 100K steps per game.
R1 mode: substrate gets pixels (210×160×3), picks actions. No reward to substrate.
External measurement: total reward (human-normalized score), unique states visited.

674 encodes 210×160×3 frames via avgpool16 → 256D vector → K=12 bit hash.
Frame is ~6x bigger than LS20 — pooling will be coarser but still works.

Standard 26 Atari 100K games (Bellemare et al. 2013 / Kaiser et al. 2019):
Alien, Amidar, Assault, Asterix, BankHeist, BattleZone, Boxing, Breakout,
ChopperCommand, CrazyClimber, DemonAttack, Freeway, Frostbite, Gopher, Hero,
Jamesbond, Kangaroo, Krull, KungFuMaster, MsPacman, Pong, PrivateEye,
Qbert, RoadRunner, Seaquest, UpNDown.

Budget: 5 min per game (100K steps each). 3 seeds per game.
Run games sequentially. Report: per-game reward sum, states visited.

Leo's note: we're NOT trying to beat EfficientZero (249% median HNS).
We're establishing what R1-mode self-organization achieves.
Compare random baseline (Step 766).
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import TransitionTriggered674

print("=" * 65)
print("STEP 763 - 674 ON ATARI 100K (26 GAMES)")
print("=" * 65)

ATARI_100K_GAMES = [
    "Alien", "Amidar", "Assault", "Asterix", "BankHeist",
    "BattleZone", "Boxing", "Breakout", "ChopperCommand", "CrazyClimber",
    "DemonAttack", "Freeway", "Frostbite", "Gopher", "Hero",
    "Jamesbond", "Kangaroo", "Krull", "KungFuMaster", "MsPacman",
    "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"
]

N_SEEDS = 1      # 1 seed per game for initial scan; expand to 3 in Step 765
N_STEPS = 100_000
PER_GAME_TIME = 300   # 5 min max per game per seed

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)


def run_atari_game(game_name, n_seeds, n_steps, per_game_time):
    """Run 674 on one Atari game for n_seeds seeds."""
    results = []
    env_id = f"ALE/{game_name}-v5"

    for seed in range(n_seeds):
        try:
            env = gym.make(env_id, render_mode=None, frameskip=4)
            n_actions = env.action_space.n
            sub = TransitionTriggered674(n_actions=n_actions, seed=seed)
            sub.reset(seed)

            obs, info = env.reset(seed=seed)
            steps = 0; total_reward = 0.0
            unique_obs = set()
            t_start = time.time()

            while steps < n_steps and (time.time() - t_start) < per_game_time:
                obs_arr = np.array(obs, dtype=np.float32) / 255.0
                action = sub.process(obs_arr) % n_actions
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

                # Track unique state approximation via coarse hash
                if steps % 100 == 0:
                    h = hash(obs_arr.tobytes()[:64])  # coarse fingerprint
                    unique_obs.add(h)

                if terminated or truncated:
                    obs, info = env.reset(seed=seed)
                    sub.on_level_transition()

            env.close()
            results.append({
                "seed": seed, "steps": steps,
                "total_reward": total_reward,
                "unique_states": len(unique_obs),
                "elapsed": round(time.time() - t_start, 1)
            })
            print(f"    seed={seed}: steps={steps} reward={total_reward:.0f} "
                  f"unique={len(unique_obs)}")
        except Exception as e:
            print(f"    seed={seed}: ERROR {e}")
            results.append({"seed": seed, "steps": 0, "total_reward": 0, "error": str(e)})
    return results


all_results = {}

for game in ATARI_100K_GAMES:
    print(f"\n-- {game} --")
    r = run_atari_game(game, N_SEEDS, N_STEPS, PER_GAME_TIME)
    all_results[game] = r
    valid = [x for x in r if "total_reward" in x and not x.get("error")]
    if valid:
        avg_r = float(np.mean([x["total_reward"] for x in valid]))
        avg_u = float(np.mean([x["unique_states"] for x in valid]))
        print(f"  {game}: avg_reward={avg_r:.1f} avg_unique={avg_u:.0f}")

print("\n" + "=" * 65)
print("STEP 763 SUMMARY - 674 ON ATARI 100K")
print("=" * 65)
print(f"{'Game':<20} {'Avg Reward':>12} {'Unique States':>14}")
print("-" * 50)
for game in ATARI_100K_GAMES:
    r = all_results.get(game, [])
    valid = [x for x in r if "total_reward" in x and not x.get("error")]
    if valid:
        avg_r = float(np.mean([x["total_reward"] for x in valid]))
        avg_u = float(np.mean([x["unique_states"] for x in valid]))
        print(f"{game:<20} {avg_r:>12.1f} {avg_u:>14.0f}")
    else:
        print(f"{game:<20} {'ERROR':>12} {'N/A':>14}")
print("-" * 50)
print("R1 mode (no reward to substrate). Compare to random (Step 766).")
print("Published EfficientZero median HNS: 249%. We're not competing —")
print("establishing the R1 floor for self-organized exploration.")
print("=" * 65)
print("STEP 763 DONE")
print("=" * 65)
