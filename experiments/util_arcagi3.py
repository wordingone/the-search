"""
arcagi3 — thin wrapper around arc_agi 0.9.4 exposing a gym-style interface.

arcagi3.make("LS20") -> env
env.reset(seed=None) -> frame   (frame[0] is 64x64 grid, values 0-15)
env.step(action_int) -> (frame, reward, done, info)  info={'level': int}
"""
import arc_agi
from arcengine import GameState

_GAME_IDS = {
    'LS20': 'ls20',
    'FT09': 'ft09',
    'VC33': 'vc33',
}


class _Env:
    def __init__(self, game_name):
        self._arc = arc_agi.Arcade()
        games = self._arc.get_environments()
        key = _GAME_IDS.get(game_name, game_name.lower())
        info = next(g for g in games if key in g.game_id.lower())
        self._env = self._arc.make(info.game_id)
        # FIX 2026-03-23: _env.action_space = [GameAction.X] (1 element) for FT09/VC33
        # is the "default" action type, not the full action set. All GameAction values work.
        # FIX 2026-03-23b: Skip GameAction.RESET (index 0) — sending RESET resets FT09/VC33,
        # causing degenerate exploration (substrate keeps resetting the game).
        from arcengine import GameAction
        _GA_ALL = list(GameAction)  # [RESET, ACTION1..ACTION7]
        self._action_space = _GA_ALL[1:]  # 7 elements: ACTION1..ACTION7
        self._last_obs = None

    def reset(self, seed=None):
        # arc_agi doesn't support seeding — seed param is ignored
        self._last_obs = self._env.reset()
        return self._frame()

    def step(self, action_int):
        # Map action_int 0..6 → ACTION1..ACTION7 (skip RESET at index 0).
        # FT09/VC33: RESET resets the game → degenerate exploration if included.
        from arcengine import GameAction
        _GA_LIST = list(GameAction)[1:]  # ACTION1..ACTION7, len=7
        ga = _GA_LIST[action_int % len(_GA_LIST)]
        obs = self._env.step(ga)
        self._last_obs = obs
        if obs is None:
            return None, 0.0, True, {'level': 0}
        done = obs.state in (GameState.GAME_OVER, GameState.WIN)
        info = {'level': obs.levels_completed}
        frame = obs.frame if obs.frame else None
        return frame, 0.0, done, info

    def _frame(self):
        if self._last_obs is None or not self._last_obs.frame:
            return None
        return self._last_obs.frame


def make(game_name):
    return _Env(game_name)
