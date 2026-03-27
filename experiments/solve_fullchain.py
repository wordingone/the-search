"""
Full-chain solver for bp35, lf52, s5i5, r11l.

Key optimization: uses set_level() to skip prefix replay.
Each BFS state costs only ~4ms (create+set_level) + 0.01ms*depth.

Strategy per level:
1. set_level(N) to jump to level N directly
2. Find unique effective actions (grid scan)
3. BFS with frame hashing
"""
import json
import time
import sys
import os
import hashlib
import importlib
import numpy as np
from collections import deque
from arcengine import GameAction, ActionInput, GameState

GAME_PATHS = {
    'bp35': ('B:/M/the-search/environment_files/bp35/0a0ad940', 'bp35', 'Bp35'),
    'lf52': ('B:/M/the-search/environment_files/lf52/271a04aa', 'lf52', 'Lf52'),
    's5i5': ('B:/M/the-search/environment_files/s5i5/a48e4b1d', 's5i5', 'S5i5'),
    'r11l': ('B:/M/the-search/environment_files/r11l/aa269680', 'r11l', 'R11l'),
}

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'

L1_SOLUTIONS = {
    'bp35': [3, 3, 3, 3, 2097, 2463, 2, 2, 2, 2079, 2079, 3, 2085, 2, 2],
    'lf52': [1241, 1253, 1253, 1265, 1265, 2033, 2033, 2801],
    's5i5': [2847, 2847, 2847, 2847, 2847, 2847, 1396, 1396, 1396, 1396, 1396, 1396, 1396],
    'r11l': [5, 2583, 2729, 947, 3931, 6, 3614, 3419, 3, 6, 5, 3, 166, 946, 6, 2038, 3455, 0, 1, 427, 62, 1178, 1154, 2, 3937, 0, 4, 3, 5, 3082, 1867, 193, 3, 3444, 2, 376, 6, 2352, 4, 4, 204, 0, 4, 5, 2035, 266, 823, 5, 3229, 2024, 1, 4020, 690, 2, 1541],
}

GAME_LEVELS = {'bp35': 9, 'lf52': 10, 's5i5': 8, 'r11l': 6}

ARCAGI3_TO_GA = {
    0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3,
    3: GameAction.ACTION4, 4: GameAction.ACTION5, 5: GameAction.ACTION6,
    6: GameAction.ACTION7,
}

KB_ACTIONS = {
    'bp35': [2, 3],       # left, right
    'lf52': [0, 1, 2, 3], # ACTION1-4
    's5i5': [],
    'r11l': [],
}


def encode_click(x, y):
    return 7 + y * 64 + x

def decode_action(a):
    if a < 7: return f"KB{a}"
    ci = a - 7
    return f"CL({ci%64},{ci//64})"

def frame_hash(arr):
    if arr is None: return ''
    return hashlib.md5(arr.astype(np.uint8).tobytes()).hexdigest()


def load_game_class(game_id):
    path, mod_name, cls_name = GAME_PATHS[game_id]
    if path not in sys.path:
        sys.path.insert(0, path)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def game_step(game, action):
    """Execute arcagi3 action. Return (frame_2d, levels_completed, done)."""
    if action < 7:
        ga = ARCAGI3_TO_GA[action]
        if ga == GameAction.ACTION6:
            ai = ActionInput(id=GameAction.ACTION6, data={'x': 0, 'y': 0})
        else:
            ai = ActionInput(id=ga, data={})
    else:
        ci = action - 7
        ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})

    try:
        r = game.perform_action(ai, raw=True)
    except:
        return None, 0, False
    if r is None:
        return None, 0, False

    f = np.array(r.frame, dtype=np.uint8)
    if f.ndim == 3: f = f[-1]
    done = r.state in (GameState.GAME_OVER, GameState.WIN)
    return f, r.levels_completed, done


def make_at_level(game_cls, level_idx):
    """Create game set to specific level (0-indexed)."""
    g = game_cls()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)
    return g


def replay_at_level(game_cls, level_idx, actions):
    """Create game at level, replay actions. Return (frame, levels_completed, done)."""
    g = make_at_level(game_cls, level_idx)
    frame = None
    levels = 0
    done = False
    for a in actions:
        frame, levels, done = game_step(g, a)
        if done or levels > 0:
            break
    return frame, levels, done


def get_base_frame(game_cls, level_idx):
    """Get base frame at start of level."""
    g = make_at_level(game_cls, level_idx)
    # Do a no-op click to get frame
    frame, levels, done = game_step(g, encode_click(0, 0))
    return frame


def find_unique_actions(game_cls, level_idx, kb_actions, click_step=2):
    """Find unique effective actions at this level."""
    base_frame = get_base_frame(game_cls, level_idx)
    base_hash = frame_hash(base_frame)
    groups = {}

    # Keyboard
    for kb in kb_actions:
        frame, levels, _ = replay_at_level(game_cls, level_idx, [kb])
        if frame is None: continue
        if levels > 0:
            return [kb], True
        h = frame_hash(frame)
        if h != base_hash and h not in groups:
            groups[h] = kb

    # Click grid
    for y in range(0, 64, click_step):
        for x in range(0, 64, click_step):
            a = encode_click(x, y)
            frame, levels, _ = replay_at_level(game_cls, level_idx, [a])
            if frame is None: continue
            if levels > 0:
                return [a], True
            h = frame_hash(frame)
            if h != base_hash and h not in groups:
                groups[h] = a

    return list(groups.values()), False


def bfs_solve_level(game_cls, level_idx, game_id, max_depth=50, max_states=500000, time_limit=300):
    """BFS solve level using set_level + unique actions."""
    t0 = time.time()

    base_frame = get_base_frame(game_cls, level_idx)
    init_hash = frame_hash(base_frame)

    print(f"    Scanning actions...")
    kb = KB_ACTIONS[game_id]
    unique, instant = find_unique_actions(game_cls, level_idx, kb, click_step=2)

    if instant:
        print(f"    INSTANT!")
        return unique

    # If few actions, try finer grid
    if len(unique) < 2:
        print(f"    Only {len(unique)} actions, retrying step=1...")
        unique, instant = find_unique_actions(game_cls, level_idx, kb, click_step=1)
        if instant:
            return unique

    print(f"    {len(unique)} unique: {[decode_action(a) for a in unique]}")
    if not unique:
        return None

    # BFS
    queue = deque([()])
    visited = {init_hash}
    explored = 0
    depth = 0

    while queue:
        if time.time() - t0 > time_limit:
            print(f"    TIMEOUT ({time_limit}s, e={explored}, d={depth})")
            return None

        seq = queue.popleft()

        if len(seq) > depth:
            depth = len(seq)
            el = time.time() - t0
            rate = explored / max(el, 0.1)
            print(f"      d={depth} v={len(visited)} q={len(queue)} e={explored} t={el:.0f}s ({rate:.0f}/s)")

        if len(seq) >= max_depth:
            continue

        for action in unique:
            actions = list(seq) + [action]
            frame, levels, done = replay_at_level(game_cls, level_idx, actions)
            explored += 1

            if levels > 0:
                print(f"    SOLVED! {len(actions)} actions, {explored} states, {time.time()-t0:.1f}s")
                return actions

            if done or frame is None:
                continue

            h = frame_hash(frame)
            if h not in visited:
                visited.add(h)
                queue.append(tuple(actions))

            if explored >= max_states:
                print(f"    LIMIT ({max_states}, d={depth})")
                return None

    print(f"    EXHAUSTED ({explored})")
    return None


def replay_full_chain(game_cls, full_sequence):
    """Replay entire chain from L1 to verify. Return levels_completed."""
    g = game_cls()
    g.full_reset()
    levels = 0
    for a in full_sequence:
        _, levels, done = game_step(g, a)
        if done:
            break
    return levels


def solve_game(game_id):
    print(f"\n{'='*60}")
    print(f"SOLVING {game_id.upper()} ({GAME_LEVELS[game_id]} levels)")
    print(f"{'='*60}")

    game_cls = load_game_class(game_id)
    l1_seq = L1_SOLUTIONS[game_id]
    total = GAME_LEVELS[game_id]

    results = {
        'game': game_id, 'total_levels': total,
        'levels': {}, 'full_sequence': [],
        'method': 'bfs_set_level',
    }

    # Verify L1
    print(f"\nL1: Verifying ({len(l1_seq)} actions)...")
    g = game_cls()
    g.full_reset()
    levels = 0
    for a in l1_seq:
        _, levels, done = game_step(g, a)
        if done: break

    if levels >= 1:
        print(f"  L1 OK")
        results['levels']['L1'] = {'status': 'SOLVED', 'actions': l1_seq, 'length': len(l1_seq), 'method': 'known'}
        results['full_sequence'] = list(l1_seq)
    else:
        print(f"  L1 FAIL")
        results['max_level_solved'] = 0
        return results

    current = 1

    for lnum in range(2, total + 1):
        if current < lnum - 1:
            break

        print(f"\nL{lnum} (level_idx={lnum-1}):")
        t0 = time.time()

        sol = bfs_solve_level(game_cls, lnum - 1, game_id,
                               max_depth=50, max_states=500000, time_limit=300)
        elapsed = time.time() - t0

        if sol is not None:
            results['levels'][f'L{lnum}'] = {
                'status': 'SOLVED', 'actions': sol,
                'length': len(sol), 'time': round(elapsed, 2),
            }
            results['full_sequence'].extend(sol)
            current = lnum

            # Verify full chain
            vlev = replay_full_chain(game_cls, results['full_sequence'])
            print(f"  Verified: chain levels={vlev}")
            acts = [decode_action(a) for a in sol[:20]]
            print(f"  Actions: {acts}{'...' if len(sol) > 20 else ''}")
        else:
            results['levels'][f'L{lnum}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
            print(f"  UNSOLVED ({elapsed:.1f}s)")
            break

    results['max_level_solved'] = current
    results['total_actions'] = len(results['full_sequence'])
    return results


if __name__ == '__main__':
    games = sys.argv[1:] if len(sys.argv) > 1 else ['s5i5', 'bp35', 'lf52', 'r11l']

    all_results = {}
    for gid in games:
        t0 = time.time()
        res = solve_game(gid)
        total_t = time.time() - t0

        outpath = os.path.join(RESULTS_DIR, f'{gid}_fullchain.json')
        with open(outpath, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\n  SAVED: {outpath}")
        print(f"  {res.get('max_level_solved',0)}/{GAME_LEVELS[gid]} levels, {res.get('total_actions',0)} actions, {total_t:.0f}s")
        all_results[gid] = res

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for gid, res in all_results.items():
        print(f"  {gid}: {res.get('max_level_solved',0)}/{GAME_LEVELS[gid]} levels, {res.get('total_actions',0)} total actions")
