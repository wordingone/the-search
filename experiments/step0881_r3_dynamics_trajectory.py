"""
step0881_r3_dynamics_trajectory.py -- R3 dynamics trajectory for step800b.

R3 hypothesis: does R3_dynamics (self-modification rate) change over time?
If delta_per_action converges → R3 decreases (substrate "settles"). If it stays
high → continuous adaptation.

Protocol: run 800b for 25K steps. Sample delta_per_action every 1K steps.
Compute:
1. L2 norm of delta_per_action (overall activation level)
2. Entropy over softmax(delta_per_action) (distribution uniformity)
3. Max delta - min delta (differentiation range)
4. L1 completions at each 5K checkpoint

Seeds: 6, 7, 8. Track one trajectory per seed.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0800b import EpsilonActionChange800b

TEST_SEEDS = [6, 7, 8]
TEST_STEPS = 25_000
N_ACTIONS = 4
SAMPLE_EVERY = 1_000
COMPLETION_CHECKPOINTS = {5_000, 10_000, 15_000, 20_000, 25_000}


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def softmax_entropy(x):
    x = x - x.max()
    e = np.exp(x)
    p = e / e.sum()
    return float(-np.sum(p * np.log(p + 1e-10)))


def run_trajectory(env_seed):
    sub = EpsilonActionChange800b(n_actions=N_ACTIONS, seed=0)
    sub.reset(0)
    env = make_game(); obs = env.reset(seed=env_seed)
    completions = 0; current_level = 0; step = 0

    trajectory = []  # (step, norm, entropy, range, cumulative_completions)
    comp_checkpoints = {}

    while step < TEST_STEPS:
        if obs is None:
            obs = env.reset(seed=env_seed); current_level = 0
            sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=env_seed); current_level = 0
            sub.on_level_transition()

        if step % SAMPLE_EVERY == 0 and hasattr(sub, 'delta_per_action'):
            d = sub.delta_per_action.copy()
            norm = float(np.linalg.norm(d))
            ent = softmax_entropy(d)
            rng = float(d.max() - d.min())
            trajectory.append((step, norm, ent, rng, completions))

        if step in COMPLETION_CHECKPOINTS:
            comp_checkpoints[step] = completions

    return trajectory, comp_checkpoints


print("=" * 70)
print("STEP 881 — R3 DYNAMICS TRAJECTORY (step800b, 25K steps)")
print("=" * 70)
print(f"Sampling delta_per_action every {SAMPLE_EVERY} steps. Seeds: {TEST_SEEDS}")

t0 = time.time()

for ts in TEST_SEEDS:
    print(f"\n--- seed={ts} ---")
    traj, comps = run_trajectory(ts * 1000)
    print(f"  {'step':>8}  {'norm':>8}  {'entropy':>8}  {'range':>8}  {'comps':>6}")
    for (step, norm, ent, rng, c) in traj[::5]:  # print every 5K steps
        print(f"  {step:>8}  {norm:>8.4f}  {ent:>8.4f}  {rng:>8.4f}  {c:>6}")
    print(f"  Final comps: {comps}")

print()
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 881 DONE")
