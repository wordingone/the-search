"""
step0863_action_consistency.py -- Action consistency: how often does 800b repeat the same action?

R3 hypothesis (characterization): does action consistency correlate with performance?
If 800b quickly converges to one dominant action (high consistency), performance is high.
If it keeps switching (low consistency), performance is low.

Measures:
1. Entropy of action distribution over 25K steps (low entropy = consistent)
2. Fraction of steps spent on dominant action
3. Time to convergence (steps until dominant action > 50% usage)
4. Correlation between action consistency and L1 completions across seeds

Tests seeds 1-20 with substrate_seeds 0-3 (to break the n_eff=1 artifact).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0800b import EpsilonActionChange800b

N_SEEDS = 20
TEST_STEPS = 25_000
N_ACTIONS = 4


def softmax_entropy(x):
    x = x - x.max()
    e = np.exp(x); p = e / e.sum()
    return float(-np.sum(p * np.log(p + 1e-10)))


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_with_tracking(env_seed, substrate_seed, n_steps):
    sub = EpsilonActionChange800b(n_actions=N_ACTIONS, seed=substrate_seed)
    sub.reset(substrate_seed)
    env = make_game(); obs = env.reset(seed=env_seed)
    completions = 0; current_level = 0; step = 0
    action_counts = np.zeros(N_ACTIONS, int)
    dominant_threshold_step = None

    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); current_level = 0
            sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        action_counts[action] += 1
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=env_seed); current_level = 0
            sub.on_level_transition()
        # Check convergence
        if dominant_threshold_step is None and step > 100:
            dom_frac = action_counts.max() / max(action_counts.sum(), 1)
            if dom_frac > 0.50:
                dominant_threshold_step = step

    action_frac = action_counts / max(action_counts.sum(), 1)
    action_entropy = softmax_entropy(action_frac)
    dominant_frac = action_frac.max()

    return {
        "completions": completions,
        "action_entropy": action_entropy,
        "dominant_frac": dominant_frac,
        "convergence_step": dominant_threshold_step,
        "action_counts": action_counts,
    }


print("=" * 70)
print("STEP 863 — ACTION CONSISTENCY CHARACTERIZATION")
print("=" * 70)
print(f"n_seeds={N_SEEDS}, steps={TEST_STEPS}. Substrate_seed = env_seed % 4.")

t0 = time.time()
results = []

for s in range(1, N_SEEDS + 1):
    env_seed = s * 1000
    substrate_seed = s % 4
    r = run_with_tracking(env_seed, substrate_seed, TEST_STEPS)
    results.append(r)

# Summary
completions = [r["completions"] for r in results]
entropies = [r["action_entropy"] for r in results]
dom_fracs = [r["dominant_frac"] for r in results]
conv_steps = [r["convergence_step"] for r in results if r["convergence_step"] is not None]

print(f"\nMean L1: {np.mean(completions):.1f}/seed  (random=36.4)")
print(f"Mean action entropy: {np.mean(entropies):.4f}  (max={np.log(N_ACTIONS):.4f})")
print(f"Mean dominant action fraction: {np.mean(dom_fracs):.3f}")
if conv_steps:
    print(f"Mean convergence step: {np.mean(conv_steps):.0f}  (fraction: {len(conv_steps)}/{N_SEEDS} converged)")
else:
    print(f"Convergence: 0/{N_SEEDS} reached >50% dominance")

# Correlation
if np.std(entropies) > 0 and np.std(completions) > 0:
    corr = np.corrcoef(entropies, completions)[0, 1]
    print(f"Correlation (entropy vs L1): {corr:.3f}")

print(f"\nPer-seed summary:")
for i, r in enumerate(results):
    s = i + 1
    print(f"  seed={s:3d}: L1={r['completions']:4d}  entropy={r['action_entropy']:.4f}  dom={r['dominant_frac']:.3f}  conv={r['convergence_step']}")

print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 863 DONE")
