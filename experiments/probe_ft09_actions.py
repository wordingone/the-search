"""
Probe FT09/VC33 action interface.

Leo's diagnosis (id 2518): FT09 has 1 action TYPE (ACTION6=click) with integer PARAMETER 0-67.
Test: pass raw integers to self._env.step() — does the game accept them?
Report: which integers produce non-identical frames.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
import numpy as np
import arc_agi
from arcengine import GameState, GameAction

print("=" * 60)
print("PROBE: FT09/VC33 action interface")
print("=" * 60)

arc = arc_agi.Arcade()
games = arc.get_environments()

for game_key in ['ft09', 'vc33']:
    print(f"\n--- {game_key.upper()} ---")
    info = next(g for g in games if game_key in g.game_id.lower())
    env = arc.make(info.game_id)
    obs0 = env.reset()
    print(f"action_space: {env.action_space}")
    print(f"frame type: {type(obs0.frame) if obs0 else None}")
    if obs0 and obs0.frame:
        f0 = np.array(obs0.frame, dtype=np.float32)
        print(f"frame shape: {f0.shape}, min={f0.min():.1f}, max={f0.max():.1f}")

    # Test GameAction enum values
    print("\nGameAction enum test:")
    for ga in list(GameAction):
        try:
            env2 = arc.make(info.game_id)
            env2.reset()
            obs2 = env2.step(ga)
            if obs2 and obs2.frame:
                f2 = np.array(obs2.frame, dtype=np.float32)
                diff = float(np.mean(np.abs(f2 - f0)))
                print(f"  step({ga.name}) → diff={diff:.4f} state={obs2.state.name if obs2 else 'None'}")
            else:
                print(f"  step({ga.name}) → obs=None or no frame")
        except Exception as e:
            print(f"  step({ga.name}) → ERROR: {e}")

    # Test raw integers
    print("\nRaw integer test (0-8):")
    for i in range(9):
        try:
            env3 = arc.make(info.game_id)
            env3.reset()
            obs3 = env3.step(i)
            if obs3 and obs3.frame:
                f3 = np.array(obs3.frame, dtype=np.float32)
                diff = float(np.mean(np.abs(f3 - f0)))
                print(f"  step({i}) → diff={diff:.4f} state={obs3.state.name}")
            else:
                print(f"  step({i}) → obs=None or no frame")
        except Exception as e:
            print(f"  step({i}) → ERROR: {e}")

print("\n" + "=" * 60)
print("PROBE DONE")
print("=" * 60)
