"""
Step 722 (A3): 674 on Atari Montezuma R1 mode + R3 dynamics.

R3 hypothesis: encoding may need temporal integration (D4) for Atari partial
observability. Current channel-0 avgpool16 encoding loses motion and temporal
context. Expected: low room discovery, R3 dynamics similar to LS20.

Metrics:
  - Rooms visited (primary)
  - Unique observation patches (diversity proxy)
  - aliased_count, live_count at end
  - R3 dynamic score on collected Atari frames
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from substrates.step0674 import TransitionTriggered674
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 722 (A3) — ATARI MONTEZUMA R1 MODE + R3 DYNAMICS")
print("=" * 65)

N_STEPS = 5000
SEED = 0

# Check ale_py
try:
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    env = gym.make("ALE/MontezumaRevenge-v5", render_mode=None, frameskip=4)
    n_actions = env.action_space.n
    print(f"  Env: MontezumaRevenge-v5, n_actions={n_actions}, "
          f"obs={env.observation_space.shape}")
    ENV_OK = True
except Exception as e:
    print(f"  Atari not available: {e}")
    print("  Install: pip install ale-py && autorom --accept-license")
    ENV_OK = False

if not ENV_OK:
    sys.exit(0)

# Run 674 on Montezuma R1
print(f"\n  Running 674 (n_actions={n_actions}) on Montezuma, {N_STEPS} steps...")
sub = TransitionTriggered674(n_actions=n_actions, seed=SEED)
sub.reset(SEED)

obs, info = env.reset(seed=SEED)
t_start = time.time()
steps = 0
obs_patches = set()
rooms_visited = set()
first_novel_step = None
episode_count = 0
atari_obs_seq = []  # for R3 dynamics

while steps < N_STEPS and (time.time() - t_start) < 290:
    obs_arr = np.array(obs, dtype=np.float32) / 255.0
    atari_obs_seq.append(obs_arr)
    action = sub.process(obs_arr)
    action = action % n_actions
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1

    # Diversity: center patch hash
    patch = obs[80:96, 72:88, 0]
    patch_hash = hash(patch.tobytes())
    if patch_hash not in obs_patches:
        if not obs_patches:
            first_novel_step = steps
        obs_patches.add(patch_hash)

    # Room via RAM if available
    if isinstance(info, dict) and "ram" in info:
        rooms_visited.add(int(info["ram"][3]))

    if terminated or truncated:
        episode_count += 1
        obs, info = env.reset(seed=SEED + episode_count)
        sub.on_level_transition()

elapsed = time.time() - t_start
env.close()

state = sub.get_state()
print(f"\n  Steps: {steps}  Episodes: {episode_count}  Elapsed: {elapsed:.1f}s")
print(f"  Unique patches: {len(obs_patches)}  First novel: step {first_novel_step}")
print(f"  Rooms (RAM): {sorted(rooms_visited)[:10]}")
print(f"  674 state: live={state['live_count']} G={state['G_size']} "
      f"aliased={state.get('aliased_count',0)} ref={state.get('ref_count',0)}")

# R3 dynamics on Atari frames
print(f"\n  R3 dynamics on Atari frames ({min(len(atari_obs_seq), 2000)} frames, 10 ckpts)...")
judge = ConstitutionalJudge()

class _674_atari(TransitionTriggered674):
    def __init__(self): super().__init__(n_actions=n_actions, seed=0)

r3_dyn = judge.measure_r3_dynamics(
    _674_atari,
    obs_sequence=atari_obs_seq[:2000],
    n_steps=min(len(atari_obs_seq), 2000),
    n_checkpoints=10
)
print(f"  R3 dynamic score: {r3_dyn.get('r3_dynamic_score')}")
print(f"  Profile: {r3_dyn.get('dynamics_profile')}")
print(f"  Declared M: {r3_dyn.get('declared_M_elements')}")
print(f"  Verified M: {r3_dyn.get('verified_M_elements')}")
change_times = r3_dyn.get("component_change_times", {})
print(f"  Changed keys: {sorted(change_times.keys())[:5]}")

print(f"\n  L1 (rooms>1): {'PASS' if len(rooms_visited)>1 else 'FAIL'}")
print(f"  L1 (patches>50): {'PASS' if len(obs_patches)>50 else 'FAIL'}")
print(f"\n  Note: R1 mode (no reward) is harder than published Atari 100K.")
print(f"  D4 (temporal integration) hypothesis: partial observability may require")
print(f"  frame stacking or motion encoding beyond static avgpool16.")

print("\n" + "=" * 65)
print("STEP 722 DONE")
print("=" * 65)
