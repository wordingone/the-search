"""
ExprSubstrate on a%b (modular arithmetic).
Tests whether evolution finds signal dims (0,1) vs noise dims (2-31).
5 seeds, post-hoc accuracy. 5-minute cap.
"""
import sys, time, random
import numpy as np
import torch
from collections import Counter, defaultdict

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from expr import ExprSubstrate, evaluate


def get_split_dims(tree):
    """Recursively collect all feature indices used in splits."""
    if isinstance(tree, (int, float)):
        return []
    _, cond, then_b, else_b = tree
    _, feat_idx, _ = cond
    return [int(feat_idx)] + get_split_dims(then_b) + get_split_dims(else_b)


def make_input(a, b, n_dims=32):
    x = torch.zeros(n_dims)
    x[0] = a / 20.0
    x[1] = b / 5.0
    x[2:] = torch.randn(n_dims - 2) * 0.01
    return x


def run_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_dims, n_actions = 32, 5
    s = ExprSubstrate(n_dims=n_dims, n_actions=n_actions)

    # Build all 100 (a,b) pairs
    pairs = [(a, b) for b in range(1, 6) for a in range(20)]  # 100 pairs

    # Present 5 rounds = 500 steps (fixed inputs to avoid noise variance)
    for _ in range(5):
        for a, b in pairs:
            x = make_input(a, b, n_dims)
            s.step(x, n_actions=n_actions)

    # Evaluate: for each pair, record action (no noise — use fixed encoding)
    results = {}
    for a, b in pairs:
        x = torch.zeros(n_dims)
        x[0] = a / 20.0
        x[1] = b / 5.0
        action = evaluate(s.pop[s.best], x) % n_actions
        true_label = a % b
        results[(a, b)] = (action, true_label)

    # Post-hoc accuracy: assign each action to its majority true label
    action_to_labels = defaultdict(list)
    for (a, b), (action, true_label) in results.items():
        action_to_labels[action].append(true_label)

    action_to_class = {}
    for action, labels in action_to_labels.items():
        action_to_class[action] = Counter(labels).most_common(1)[0][0]

    correct = sum(1 for (a, b), (action, true_label) in results.items()
                  if action_to_class.get(action, -1) == true_label)
    accuracy = correct / len(results)

    # Which dims does the best tree split on?
    split_dims = get_split_dims(s.pop[s.best])
    n_splits = len(split_dims)
    signal_splits = sum(1 for d in split_dims if d < 2)
    noise_splits = n_splits - signal_splits
    signal_frac = signal_splits / n_splits if n_splits > 0 else 0.0
    chance_signal_frac = 2 / 32  # 6.25%

    actions_used = len(set(action for action, _ in results.values()))

    return accuracy, signal_frac, n_splits, signal_splits, actions_used


def main():
    t0 = time.time()
    print("=" * 55)
    print("ExprSubstrate: a%b (modular arithmetic)")
    print("d=32: x[0]=a/20, x[1]=b/5, x[2:]=noise(0,0.01)")
    print("500 steps (100 pairs × 5 rounds), 5 seeds")
    print("Leo prediction: ~20% accuracy (chance), random dim splits")
    print("=" * 55)

    accs = []
    signal_fracs = []
    for seed in range(5):
        acc, sfrac, n_splits, sig_splits, acts = run_seed(seed)
        accs.append(acc)
        signal_fracs.append(sfrac)
        status = "SURPRISE" if acc > 0.25 else ("CHANCE" if acc < 0.22 else "MARGINAL")
        print(f"  seed={seed}: acc={acc*100:.1f}%  signal_splits={sig_splits}/{n_splits} ({sfrac*100:.1f}%)  actions={acts}/5  [{status}]")

    chance_signal = 2/32 * 100
    print(f"\nSummary:")
    print(f"  accuracy: min={min(accs)*100:.1f}%  max={max(accs)*100:.1f}%  mean={np.mean(accs)*100:.1f}%")
    print(f"  signal dim frac: min={min(signal_fracs)*100:.1f}%  max={max(signal_fracs)*100:.1f}%  mean={np.mean(signal_fracs)*100:.1f}%  (chance={chance_signal:.1f}%)")
    print(f"  Leo predicted: ~20% accuracy, ~6.3% signal dim frac")

    elapsed = time.time() - t0
    print(f"\nelapsed={elapsed:.1f}s")


if __name__ == '__main__':
    main()
