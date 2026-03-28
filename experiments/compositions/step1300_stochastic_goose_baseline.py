"""
Step 1300 — StochasticGoose PRISM Baseline (Jun directive, Leo mail 3662/3663).

Exact port of DriesSmit/ARC3-solution run through PRISM protocol.
11 games × 20 draws = 220 runs. Same measurement protocol as Step 1298.

Research questions:
  1. What does the leaderboard leader ACTUALLY score on our specific games?
  2. Does it reach L2+ on any game?
  3. What's its RHAE?
  4. Does it show second-exposure speedup from genuine CNN learning?
  5. Where does it stand vs RANDOM/ARGMIN/PE-EMA (Step 1298 reference)?

Architecture notes:
  - CNN trains online: Adam lr=0.0001, batch=64, train every 5 steps
  - Binary frame-change reward (1.0 if frame changed, 0.0 if not)
  - Buffer reset + model reset on level transition
  - Can't access PRISM keyboard actions 5-6 (SG only has 5 discrete outputs)
  - MBPP: falls back to random (SG not designed for text)

Spec: Leo mail 3662/3663, 2026-03-28.
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

import numpy as np
from substrates.stochastic_goose import StochasticGooseSubstrate

# --- Config ---
ARC_GAMES = ['ls20', 'ft09', 'vc33', 'tr87', 'sp80', 'sb26', 'tu93', 'cn04', 'cd82', 'lp85']
MBPP_GAMES = ['mbpp_0']
GAMES = ARC_GAMES + MBPP_GAMES

N_DRAWS = 20
MAX_STEPS = 10_000
MAX_SECONDS = 300

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1300')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'

SOLVER_PRESCRIPTIONS = {
    'ls20':  ('ls20_fullchain.json',  'all_actions'),
    'ft09':  ('ft09_fullchain.json',  'all_actions'),
    'vc33':  ('vc33_fullchain.json',  'all_actions_encoded'),
    'tr87':  ('tr87_fullchain.json',  'all_actions'),
    'sp80':  ('sp80_fullchain.json',  'sequence'),
    'sb26':  ('sb26_fullchain.json',  'all_actions'),
    'tu93':  ('tu93_fullchain.json',  'all_actions'),
    'cn04':  ('cn04_fullchain.json',  'sequence'),
    'cd82':  ('cd82_fullchain.json',  'sequence'),
    'lp85':  ('lp85_fullchain.json',  'full_sequence'),
}


# --- Utilities (copied from step1298) ---

def make_game(game_name):
    if game_name.lower().startswith('mbpp'):
        from mbpp_game import make as mbpp_make
        return mbpp_make(game_name.lower())
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def load_prescription(game_name):
    if game_name.lower() not in SOLVER_PRESCRIPTIONS:
        return None
    fname, field = SOLVER_PRESCRIPTIONS[game_name.lower()]
    try:
        with open(os.path.join(PDIR, fname)) as f:
            d = json.load(f)
        return d.get(field)
    except Exception:
        return None


def compute_solver_level_steps(game_name, seed=1):
    if game_name.lower().startswith('mbpp'):
        try:
            from mbpp_game import compute_solver_steps
            idx = int(game_name.split('_', 1)[1]) if '_' in game_name else 0
            return compute_solver_steps(idx)
        except Exception:
            return {}
    prescription = load_prescription(game_name)
    if prescription is None:
        return {}
    env = make_game(game_name)
    obs = env.reset(seed=seed)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103
    ACTION_OFFSET = {'ls20': -1, 'vc33': 7}
    offset = ACTION_OFFSET.get(game_name.lower(), 0)
    level, level_first_step, step, fresh_episode = 0, {}, 0, True
    for action in prescription:
        action_int = (int(action) + offset) % n_actions
        obs_next, reward, done, info = env.step(action_int)
        step += 1
        if fresh_episode:
            fresh_episode = False
            obs = obs_next
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level_first_step[cl] = step
            level = cl
        if done:
            obs = env.reset(seed=seed)
            fresh_episode = True
        else:
            obs = obs_next
    return level_first_step


def compute_arc_score(level_first_steps, solver_level_steps):
    scores = []
    for lv, solver_step in solver_level_steps.items():
        if lv in level_first_steps and level_first_steps[lv] > 0:
            ratio = solver_step / level_first_steps[lv]
            scores.append(min(1.0, ratio * ratio))
    return float(np.mean(scores)) if scores else 0.0


def compute_rhae(level_first_steps, solver_level_steps, total_actions):
    scores = []
    for lv, solver_step in solver_level_steps.items():
        if lv in level_first_steps and level_first_steps[lv] > 0:
            ratio = solver_step / level_first_steps[lv]
            scores.append(min(1.0, ratio * ratio))
    return float(np.mean(scores)) if scores else 0.0


def compute_post_transition_kl(action_log, l1_step, n_actions, window=100):
    if l1_step is None or l1_step < window:
        return None
    pre = action_log[max(0, l1_step - window):l1_step]
    post = action_log[l1_step:min(len(action_log), l1_step + window)]
    if len(pre) < 10 or len(post) < 10:
        return None
    pre_c = np.zeros(n_actions, np.float32)
    post_c = np.zeros(n_actions, np.float32)
    for a in pre:
        if 0 <= a < n_actions:
            pre_c[a] += 1
    for a in post:
        if 0 <= a < n_actions:
            post_c[a] += 1
    if pre_c.sum() == 0 or post_c.sum() == 0:
        return None
    pre_p = (pre_c + 1e-8) / (pre_c.sum() + 1e-8 * n_actions)
    post_p = (post_c + 1e-8) / (post_c.sum() + 1e-8 * n_actions)
    return round(float(np.sum(post_p * np.log(post_p / pre_p))), 4)


# --- Episode runner ---

def run_episode(env, substrate, n_actions, solver_level_steps, seed, is_mbpp=False):
    obs = env.reset(seed=seed)

    action_log = []
    steps = 0
    level = 0
    max_level = 0
    level_first_step = {}
    t_start = time.time()
    fresh_episode = True

    while steps < MAX_STEPS:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr) % n_actions
        action_log.append(action)

        obs_next, reward, done, info = env.step(action)
        steps += 1

        if hasattr(substrate, 'update_after_step') and obs_next is not None:
            substrate.update_after_step(obs_next, action, reward)

        if fresh_episode:
            fresh_episode = False
            obs = obs_next
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl not in level_first_step:
                level_first_step[cl] = steps
            if cl > max_level:
                max_level = cl
            level = cl
            substrate.on_level_transition()

        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            level = 0
        else:
            obs = obs_next

    elapsed = time.time() - t_start
    l1_step = level_first_step.get(1)
    l2_step = level_first_step.get(2)
    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    rhae = compute_rhae(level_first_step, solver_level_steps, steps)
    post_kl = compute_post_transition_kl(action_log, l1_step, n_actions)

    return {
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'L1_solved': bool(l1_step is not None),
        'l1_step': l1_step,
        'l2_step': l2_step,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(rhae, 6),
        'post_transition_kl': post_kl,
    }


def run_draw(game_name, draw_idx, solver_level_steps):
    """Two episodes A+B on same substrate (no reset between)."""
    env = make_game(game_name)
    is_mbpp = game_name.lower().startswith('mbpp')
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    seed_a = draw_idx * 2
    seed_b = draw_idx * 2 + 1

    substrate = StochasticGooseSubstrate(n_actions=n_actions, seed=draw_idx)

    result_a = run_episode(env, substrate, n_actions, solver_level_steps, seed=seed_a, is_mbpp=is_mbpp)
    result_b = run_episode(env, substrate, n_actions, solver_level_steps, seed=seed_b, is_mbpp=is_mbpp)

    speedup = None
    if result_a['l1_step'] is not None and result_b['l1_step'] is not None:
        speedup = round(result_a['l1_step'] / result_b['l1_step'], 3)
    elif result_a['l1_step'] is None and result_b['l1_step'] is not None:
        speedup = float('inf')

    return {
        'game': game_name,
        'draw': draw_idx,
        'condition': 'stochastic_goose',
        'episode_A': result_a,
        'episode_B': result_b,
        'second_exposure_speedup': speedup,
        'n_actions': n_actions,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step 1300 — StochasticGoose PRISM Baseline")
    print(f"Games: {GAMES}")
    print(f"Draws: {N_DRAWS} per game")
    print(f"Total: {len(GAMES) * N_DRAWS} substrate pairs")
    print()

    # Pre-compute solver level steps (once per game)
    print("Computing solver baselines...")
    solver_steps_cache = {}
    for game in ARC_GAMES:
        try:
            solver_steps_cache[game] = compute_solver_level_steps(game)
            print(f"  {game}: {solver_steps_cache[game]}")
        except Exception as e:
            print(f"  {game}: ERROR {e}")
            solver_steps_cache[game] = {}
    for game in MBPP_GAMES:
        solver_steps_cache[game] = {}
    print()

    all_results = []
    summary_rows = []

    for game in GAMES:
        solver_level_steps = solver_steps_cache.get(game, {})
        game_results = []
        l1_count_a = 0
        l1_count_b = 0
        l2_count_a = 0
        arc_scores_a = []
        rhae_vals_a = []
        speedups = []
        runtimes = []

        print(f"--- {game.upper()} ---")
        for draw in range(N_DRAWS):
            t0 = time.time()
            try:
                result = run_draw(game, draw, solver_level_steps)
            except Exception as e:
                print(f"  draw {draw:2d} ERROR: {e}")
                continue

            game_results.append(result)
            all_results.append(result)

            ea = result['episode_A']
            eb = result['episode_B']
            if ea['L1_solved']:
                l1_count_a += 1
            if eb['L1_solved']:
                l1_count_b += 1
            if ea['l2_step'] is not None:
                l2_count_a += 1
            arc_scores_a.append(ea['arc_score'])
            rhae_vals_a.append(ea['RHAE'])
            if result['second_exposure_speedup'] is not None and result['second_exposure_speedup'] != float('inf'):
                speedups.append(result['second_exposure_speedup'])
            runtimes.append(ea['elapsed_seconds'] + eb['elapsed_seconds'])

            elapsed = time.time() - t0
            su = result['second_exposure_speedup']
            su_str = f"{su:.2f}x" if (su is not None and su != float('inf')) else ("∞" if su == float('inf') else "N/A")
            print(f"  draw {draw:2d}: A_L1={ea['L1_solved']}, A_arc={ea['arc_score']:.4f}, "
                  f"A_RHAE={ea['RHAE']:.2e}, B_L1={eb['L1_solved']}, speedup={su_str}, "
                  f"t={elapsed:.1f}s")

        # Save game results
        out_path = os.path.join(RESULTS_DIR, f"{game}_stochastic_goose.jsonl")
        with open(out_path, 'w') as f:
            for r in game_results:
                f.write(json.dumps(r) + '\n')

        # Summary row
        n = len(game_results)
        row = {
            'game': game,
            'condition': 'GOOSE',
            'n': n,
            'L1_rate_A': f"{l1_count_a}/{n}",
            'L1_rate_B': f"{l1_count_b}/{n}",
            'L2_rate_A': f"{l2_count_a}/{n}",
            'arc_mean': round(np.mean(arc_scores_a), 4) if arc_scores_a else 0.0,
            'rhae_mean': float(np.mean(rhae_vals_a)) if rhae_vals_a else 0.0,
            'speedup_mean': round(np.mean(speedups), 3) if speedups else None,
            'speedup_n_pairs': len(speedups),
            'runtime_mean': round(np.mean(runtimes), 1) if runtimes else 0.0,
        }
        summary_rows.append(row)
        print(f"  SUMMARY: L1={l1_count_a}/{n}, L2={l2_count_a}/{n}, "
              f"arc={row['arc_mean']:.4f}, RHAE={row['rhae_mean']:.2e}, "
              f"speedup={row['speedup_mean']}")
        print()

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'summary': summary_rows, 'n_draws': N_DRAWS}, f, indent=2)

    # Print full summary table
    print("\n" + "="*100)
    print("STEP 1300 — STOCHASTIC GOOSE PRISM BASELINE")
    print(f"{'Game':<10} {'L1_A':>8} {'L1_B':>8} {'L2_A':>8} {'ARC':>8} {'RHAE':>10} {'Speedup':>10} {'Runtime':>10}")
    print("-"*100)
    for row in summary_rows:
        print(f"{row['game']:<10} {row['L1_rate_A']:>8} {row['L1_rate_B']:>8} {row['L2_rate_A']:>8} "
              f"{row['arc_mean']:>8.4f} {row['rhae_mean']:>10.2e} "
              f"{str(row['speedup_mean']) if row['speedup_mean'] is not None else 'N/A':>10} "
              f"{row['runtime_mean']:>10.1f}s")
    print("="*100)
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
