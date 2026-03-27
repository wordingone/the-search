"""
Full-chain solver using random search with game-specific action sets.

For games where BFS fails (too many actions or too deep), random search
with intelligent action selection works better.

Uses set_level() for direct level access.
"""
import json
import time
import sys
import os
import hashlib
import importlib
import random
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

# Additional known solutions from BFS
KNOWN_SOLUTIONS = {
    's5i5': {
        2: [3473]*8 + [3489]*8 + [3473] + [3503]*3 + [3519]*6,  # L2
    },
}

GAME_LEVELS = {'bp35': 9, 'lf52': 10, 's5i5': 8, 'r11l': 6}

ARCAGI3_TO_GA = {
    0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3,
    3: GameAction.ACTION4, 4: GameAction.ACTION5, 5: GameAction.ACTION6,
    6: GameAction.ACTION7,
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


def find_unique_actions(game_cls, level_idx, kb_actions, click_step=2):
    """Find unique effective actions at this level."""
    g = game_cls(); g.full_reset()
    if level_idx > 0: g.set_level(level_idx)
    base_f, _, _ = game_step(g, encode_click(0, 0))
    base_h = frame_hash(base_f)
    groups = {}

    for kb in kb_actions:
        g2 = game_cls(); g2.full_reset()
        if level_idx > 0: g2.set_level(level_idx)
        f, lev, _ = game_step(g2, kb)
        if f is None: continue
        if lev > 0: return [kb], True
        h = frame_hash(f)
        if h != base_h and h not in groups:
            groups[h] = kb

    for y in range(0, 64, click_step):
        for x in range(0, 64, click_step):
            a = encode_click(x, y)
            g2 = game_cls(); g2.full_reset()
            if level_idx > 0: g2.set_level(level_idx)
            f, lev, _ = game_step(g2, a)
            if f is None: continue
            if lev > 0: return [a], True
            h = frame_hash(f)
            if h != base_h and h not in groups:
                groups[h] = a

    return list(groups.values()), False


def random_search(game_cls, level_idx, actions, max_steps, n_trials, time_limit=60):
    """Random search: try random action sequences until level is solved."""
    t0 = time.time()
    best_seq = None
    best_len = max_steps + 1
    trials = 0

    while trials < n_trials and time.time() - t0 < time_limit:
        g = game_cls()
        g.full_reset()
        if level_idx > 0:
            g.set_level(level_idx)

        seq = []
        n_steps = random.randint(1, max_steps)
        for _ in range(n_steps):
            a = random.choice(actions)
            f, lev, done = game_step(g, a)
            seq.append(a)
            if lev > 0:
                if len(seq) < best_len:
                    best_len = len(seq)
                    best_seq = list(seq)
                    print(f"    Found: {len(seq)} actions (trial {trials})")
                break
            if done:
                break

        trials += 1
        if trials % 1000 == 0:
            el = time.time() - t0
            print(f"    {trials} trials, {el:.0f}s, best={best_len if best_seq else 'none'}")

    return best_seq


def trimmed_random_search(game_cls, level_idx, actions, max_steps, n_trials, time_limit=120):
    """Random search with solution trimming: find solution then trim unnecessary actions."""
    t0 = time.time()

    # Phase 1: Find any solution
    raw_sol = random_search(game_cls, level_idx, actions, max_steps, n_trials, time_limit)
    if raw_sol is None:
        return None

    print(f"    Raw solution: {len(raw_sol)} actions, trimming...")

    # Phase 2: Trim - try removing each action
    trimmed = list(raw_sol)
    improved = True
    while improved:
        improved = False
        for i in range(len(trimmed) - 1, -1, -1):
            candidate = trimmed[:i] + trimmed[i+1:]
            g = game_cls()
            g.full_reset()
            if level_idx > 0:
                g.set_level(level_idx)
            solved = False
            for a in candidate:
                _, lev, done = game_step(g, a)
                if lev > 0:
                    solved = True
                    break
                if done:
                    break
            if solved:
                trimmed = candidate
                improved = True
                break  # restart from end

    print(f"    Trimmed: {len(raw_sol)} -> {len(trimmed)} actions")
    return trimmed


def solve_level(game_cls, level_idx, game_id, time_limit=300):
    """Solve a single level using combined BFS + random search."""
    t0 = time.time()

    # Determine keyboard actions
    kb_map = {
        'bp35': [2, 3],
        'lf52': [0, 1, 2, 3],
        's5i5': [],
        'r11l': [],
    }
    kb = kb_map.get(game_id, [])

    # Find unique actions
    print(f"    Finding unique actions...")
    unique, instant = find_unique_actions(game_cls, level_idx, kb, click_step=2)
    if instant:
        return unique

    if not unique:
        unique, instant = find_unique_actions(game_cls, level_idx, kb, click_step=1)
        if instant:
            return unique

    n_actions = len(unique)
    print(f"    {n_actions} unique actions")

    if n_actions == 0:
        return None

    # Strategy depends on number of unique actions
    if n_actions <= 4:
        # BFS feasible up to depth ~30
        print(f"    Using BFS (small action space)...")
        sol = bfs_solve(game_cls, level_idx, unique, max_depth=50, max_states=500000,
                        time_limit=min(time_limit, 240))
        if sol:
            return sol
        # Fall back to random
        remaining = time_limit - (time.time() - t0)
        if remaining > 10:
            print(f"    BFS failed, trying random search...")
            return trimmed_random_search(game_cls, level_idx, unique, 100, 100000, remaining)
        return None

    elif n_actions <= 20:
        # BFS up to depth ~10, then random
        print(f"    Using BFS (medium) then random...")
        sol = bfs_solve(game_cls, level_idx, unique, max_depth=15, max_states=200000,
                        time_limit=min(time_limit/2, 120))
        if sol:
            return sol
        remaining = time_limit - (time.time() - t0)
        if remaining > 10:
            return trimmed_random_search(game_cls, level_idx, unique, 50, 200000, remaining)
        return None

    else:
        # Pure random search
        print(f"    Using random search (large action space: {n_actions})...")
        return trimmed_random_search(game_cls, level_idx, unique, 100, 500000, time_limit)


def bfs_solve(game_cls, level_idx, actions, max_depth=50, max_states=500000, time_limit=300):
    """BFS with set_level."""
    t0 = time.time()

    g = game_cls(); g.full_reset()
    if level_idx > 0: g.set_level(level_idx)
    base_f, _, _ = game_step(g, encode_click(63, 63))
    init_hash = frame_hash(base_f)

    queue = deque([()])
    visited = {init_hash}
    explored = 0
    depth = 0

    while queue:
        if time.time() - t0 > time_limit:
            return None

        seq = queue.popleft()
        if len(seq) > depth:
            depth = len(seq)
            el = time.time() - t0
            rate = explored / max(el, 0.1)
            print(f"      d={depth} v={len(visited)} q={len(queue)} e={explored} ({rate:.0f}/s)")

        if len(seq) >= max_depth:
            continue

        for action in actions:
            g2 = game_cls(); g2.full_reset()
            if level_idx > 0: g2.set_level(level_idx)
            done_flag = False
            for a in list(seq) + [action]:
                f, lev, done = game_step(g2, a)
                if lev > 0:
                    return list(seq) + [action]
                if done:
                    done_flag = True
                    break
            if done_flag or f is None:
                continue

            explored += 1
            h = frame_hash(f)
            if h not in visited:
                visited.add(h)
                queue.append(seq + (action,))

            if explored >= max_states:
                return None

    return None


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
        'method': 'bfs_random_hybrid',
    }

    # L1
    print(f"\nL1: Verifying...")
    g = game_cls(); g.full_reset()
    for a in l1_seq:
        _, lev, _ = game_step(g, a)
    if lev >= 1:
        print(f"  L1 OK")
        results['levels']['L1'] = {'status': 'SOLVED', 'actions': l1_seq, 'length': len(l1_seq)}
        results['full_sequence'] = list(l1_seq)
    else:
        print(f"  L1 FAIL")
        results['max_level_solved'] = 0
        return results

    current = 1

    # Check for known solutions
    known = KNOWN_SOLUTIONS.get(game_id, {})

    for lnum in range(2, total + 1):
        if current < lnum - 1:
            break

        print(f"\nL{lnum}:")
        t0 = time.time()

        if lnum in known:
            sol = known[lnum]
            print(f"  Using known solution ({len(sol)} actions)")
        else:
            sol = solve_level(game_cls, lnum - 1, game_id, time_limit=300)

        elapsed = time.time() - t0

        if sol is not None:
            results['levels'][f'L{lnum}'] = {
                'status': 'SOLVED', 'actions': sol,
                'length': len(sol), 'time': round(elapsed, 2),
            }
            results['full_sequence'].extend(sol)
            current = lnum

            # Verify chain
            g = game_cls(); g.full_reset()
            for a in results['full_sequence']:
                _, vlev, _ = game_step(g, a)
            print(f"  Verified chain: levels={vlev}")
            acts = [decode_action(a) for a in sol[:15]]
            print(f"  Actions: {acts}{'...' if len(sol) > 15 else ''}")
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
        tt = time.time() - t0

        outpath = os.path.join(RESULTS_DIR, f'{gid}_fullchain.json')
        with open(outpath, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\n  SAVED: {outpath}")
        print(f"  {res.get('max_level_solved',0)}/{GAME_LEVELS[gid]} levels, {res.get('total_actions',0)} actions, {tt:.0f}s")
        all_results[gid] = res

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for gid, res in all_results.items():
        print(f"  {gid}: {res.get('max_level_solved',0)}/{GAME_LEVELS[gid]} levels, {res.get('total_actions',0)} actions")
