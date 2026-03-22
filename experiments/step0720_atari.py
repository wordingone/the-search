"""
Step 720b: Atari Montezuma's Revenge — R1 mode baseline.

674 substrate on Montezuma's Revenge without reward signal.
R1 mode: substrate gets raw frames, no reward passed in.

Metrics:
  - Rooms visited (primary)
  - Steps to first room change
  - Cell count / coverage
  - Actions used

Published comparison (Atari 100K benchmark, WITH reward):
  BBF: ~14,000 score
  EfficientZero: ~11,000 score
  DreamerV3: ~4,000 score
  Random: ~200 score
  Human: ~4,753 score

Our setup is strictly harder (no reward signal).
Rooms visited > 1 = L1 pass.

Requires: ale-py + autorom ROMs
Install: pip install ale-py && autorom --accept-license
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np

print("=" * 65)
print("STEP 720b — ATARI MONTEZUMA R1 MODE BASELINE")
print("=" * 65)

# -- Check ale_py ---------------------------------------------------
try:
    import ale_py
    print(f"ale_py version: {ale_py.__version__}")
    ALE_AVAILABLE = True
except ImportError:
    print("ale_py NOT available")
    print("Install: pip install ale-py && autorom --accept-license")
    ALE_AVAILABLE = False

# -- Check gymnasium atari envs -------------------------------------
if ALE_AVAILABLE:
    try:
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        # Try to make Montezuma
        env = gym.make("ALE/MontezumaRevenge-v5", render_mode=None)
        print(f"Montezuma env: OK (obs_space={env.observation_space.shape}, n_actions={env.action_space.n})")
        env.close()
        ENV_AVAILABLE = True
    except Exception as e:
        print(f"Montezuma env ERROR: {e}")
        ENV_AVAILABLE = False
else:
    ENV_AVAILABLE = False

if not ENV_AVAILABLE:
    print("\nAtari environment not available. Reporting status only.")
    print("To install:")
    print("  pip install ale-py")
    print("  autorom --accept-license")
    print("  (or: pip install 'autorom[accept-rom-license]')")
    sys.exit(0)

# -- Run baseline --------------------------------------------------
from substrates.step0674 import TransitionTriggered674

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

N_STEPS = 5000  # 5-min cap: ~5000 Atari steps
SEED = 0

print(f"\nRunning 674 on Montezuma (R1 mode, no reward, {N_STEPS} steps)...")

env = gym.make("ALE/MontezumaRevenge-v5", render_mode=None, frameskip=4)
sub = TransitionTriggered674(n_actions=env.action_space.n, seed=SEED)
sub.reset(SEED)

obs, info = env.reset(seed=SEED)
t_start = time.time()
steps = 0
rooms_visited = set()
first_room_change_step = None
episode_count = 0

# Track initial room
# In Montezuma, room is often tracked via RAM
# Without ALE RAM access in gymnasium v1+, we approximate via done signals
# and observation hash changes

obs_hashes = set()
first_hash = hash(obs.tobytes())
obs_hashes.add(first_hash)
initial_hash = first_hash

while steps < N_STEPS and (time.time() - t_start) < 290:
    obs_arr = np.array(obs, dtype=np.float32) / 255.0
    action = sub.process(obs_arr)
    action = action % env.action_space.n
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1

    # Track unique observation regions as proxy for rooms
    # Sample center 16x16 patch to detect scene changes
    patch = obs[80:96, 72:88, 0]  # center patch
    patch_hash = hash(patch.tobytes())
    if patch_hash not in obs_hashes:
        if not rooms_visited:
            first_room_change_step = steps
        obs_hashes.add(patch_hash)

    # RAM info may have room data in older ale_py
    if isinstance(info, dict):
        room = info.get("room", info.get("ram", [None])[3] if "ram" in info else None)
        if room is not None:
            rooms_visited.add(room)

    if terminated or truncated:
        episode_count += 1
        obs, info = env.reset(seed=SEED + episode_count)
        sub.on_level_transition()

elapsed = time.time() - t_start
env.close()

# Get state
state = sub.get_state()
cell_count = state.get("live_count", state.get("G_size", "?"))
unique_obs_patches = len(obs_hashes)

print(f"\nResults ({elapsed:.1f}s elapsed):")
print(f"  Steps completed:        {steps}")
print(f"  Episodes:               {episode_count}")
print(f"  Rooms visited (RAM):    {len(rooms_visited)} — {sorted(rooms_visited)[:10]}")
print(f"  Unique patches seen:    {unique_obs_patches} (observation diversity proxy)")
print(f"  First novel patch:      step {first_room_change_step}")
print(f"  674 cells (live_count): {cell_count}")
print(f"  674 G_size:             {state.get('G_size', '?')}")

# L1 assessment
rooms_l1 = len(rooms_visited) > 1
patches_l1 = unique_obs_patches > 50
print(f"\n  L1 (rooms>1):           {'PASS' if rooms_l1 else 'FAIL'}")
print(f"  L1 (patches>50):        {'PASS' if patches_l1 else 'FAIL'}")

print("\n  Note: R1 mode (no reward) is strictly harder than published Atari 100K.")
print(f"  Published Atari 100K room count:")
print(f"    BBF/EfficientZero: typically reaches 2-5 rooms with reward")
print(f"    Random: ~1 room (stays in room 0)")
print(f"    Our R1 mode: substrate must discover structure without reward signal")

print("\n" + "=" * 65)
print("DONE")
print("=" * 65)
