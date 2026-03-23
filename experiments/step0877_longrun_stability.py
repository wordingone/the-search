"""
step0877_longrun_stability.py -- Long-run stability: 800b at 100K steps.

R3 hypothesis: does step800b performance degrade at 100K steps, or does it
remain stable? If performance plateaus early → navigation is not improving from
continued experience. If degradation → delta_per_action drifts.

Jun-approved extended run (waiver for 5-min cap).

Protocol: run step800b cold for 100K steps on seed=6. Report L1 completions
at every 10K checkpoint. Also track delta_per_action at each checkpoint.

Metric: L1 completions per 10K window (not cumulative). Flat/stable = no degradation.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0800b import EpsilonActionChange800b

TEST_SEED = 6000
N_STEPS = 100_000
N_ACTIONS = 4
WINDOW = 10_000


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


print("=" * 70)
print("STEP 877 — LONG-RUN STABILITY (800b at 100K steps)")
print("=" * 70)
print(f"env_seed={TEST_SEED}, 100K steps, reporting every {WINDOW} steps.")
print("(Jun-approved extended run)")
print()

t0 = time.time()

sub = EpsilonActionChange800b(n_actions=N_ACTIONS, seed=0)
sub.reset(0)
env = make_game(); obs = env.reset(seed=TEST_SEED)
completions = 0; current_level = 0; step = 0
window_completions = 0

print(f"{'window_end':>12}  {'window_L1':>10}  {'cumul_L1':>10}  {'delta_per_action'}")

while step < N_STEPS:
    if obs is None:
        obs = env.reset(seed=TEST_SEED); current_level = 0
        sub.on_level_transition(); continue
    action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
    obs, _, done, info = env.step(action); step += 1
    cl = info.get('level', 0) if isinstance(info, dict) else 0
    if cl > current_level:
        completions += (cl - current_level)
        window_completions += (cl - current_level)
        current_level = cl
        sub.on_level_transition()
    if done:
        obs = env.reset(seed=TEST_SEED); current_level = 0
        sub.on_level_transition()

    if step % WINDOW == 0:
        d = sub.delta_per_action if hasattr(sub, 'delta_per_action') else None
        d_str = f"{d}" if d is not None else "N/A"
        print(f"{step:>12}  {window_completions:>10}  {completions:>10}  {d_str}")
        window_completions = 0

print()
print(f"Final cumulative L1: {completions}")
print(f"Mean per 10K window: {completions / (N_STEPS // WINDOW):.1f}")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 877 DONE")
