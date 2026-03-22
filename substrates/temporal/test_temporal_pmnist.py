"""P-MNIST 1-task 5K steps: TemporalPrediction(d=784, n_actions=10). Post-hoc accuracy."""
import sys, time
import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from temporal import TemporalPrediction


def run():
    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("SKIP: torchvision not available")
        return None

    t0 = time.time()
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='/tmp/mnist_data', train=True, download=True, transform=transform)

    s = TemporalPrediction(d=784, n_actions=10)

    # Feed images in STANDARD ORDER (no shuffle) for 5K steps
    MAX_STEPS = 5000
    log = []  # (true_label, action)
    pred_errors = []

    for i in range(MAX_STEPS):
        img, label = dataset[i]
        x = img.view(-1).float()
        action = s.step(x)
        log.append((label, action))
        pred_errors.append(s.pred_err)

        if (i + 1) % 1000 == 0:
            rank = torch.linalg.matrix_rank(s.W, atol=1e-4).item()
            norm = s.W.norm().item()
            recent_err = float(np.mean(pred_errors[-100:]))
            print(f"  step {i+1}: pred_err={recent_err:.4f}  W_rank={rank}  W_norm={norm:.3f}", flush=True)

    # Post-hoc accuracy: assign each action to most-common true class
    from collections import Counter, defaultdict
    action_to_labels = defaultdict(list)
    for label, action in log:
        action_to_labels[action].append(label)

    action_to_class = {}
    for action, labels in action_to_labels.items():
        action_to_class[action] = Counter(labels).most_common(1)[0][0]

    correct = sum(1 for label, action in log if action_to_class.get(action, -1) == label)
    accuracy = correct / len(log)

    # Count unique actions used
    actions_used = len(action_to_labels)

    elapsed = time.time() - t0
    print(f"P-MNIST 5K steps: accuracy={accuracy*100:.1f}%  (chance=10%)  actions_used={actions_used}/10  {elapsed:.1f}s")
    print(f"Predicted: ~10%. SURPRISE if >15%.")
    return accuracy


if __name__ == '__main__':
    run()
