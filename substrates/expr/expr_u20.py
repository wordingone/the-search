"""
ExprSubstrate with U20 scoring: temporal_smoothness x action_coverage.

U20: local continuity of input-action mapping.
High smoothness = stable policy for consecutive observations.
High coverage = uses all actions (no collapse).

Evolution drives splits toward coherent regions WITHOUT external reward:
- Random splits thrash every step (low smoothness).
- Signal-aligned splits are stable within regions (high smoothness).
"""

import copy
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from expr import ExprSubstrate, evaluate, mutate


class ExprU20(ExprSubstrate):
    """ExprSubstrate with temporal_smoothness * action_coverage scoring."""

    def _evolve(self):
        recent = self.history[-self.window:]

        for i, tree in enumerate(self.pop):
            actions = [evaluate(tree, obs) % self.n_actions for obs, _ in recent]

            # U20: temporal smoothness — consecutive same-action pairs
            same_pairs = sum(a == b for a, b in zip(actions[:-1], actions[1:]))
            smoothness = same_pairs / max(len(actions) - 1, 1)

            # diversity guard — avoid single-action collapse
            coverage = len(set(actions)) / self.n_actions

            self.scores[i] = smoothness * coverage

        self.best = max(range(len(self.pop)), key=lambda i: self.scores[i])
        worst = min(range(len(self.pop)), key=lambda i: self.scores[i])

        candidate = mutate(copy.deepcopy(self.pop[self.best]),
                           self.n_dims, self.n_actions)
        test_actions = [evaluate(candidate, obs) % self.n_actions
                        for obs, _ in self.history[-self.window:]]
        if len(set(test_actions)) > 1:
            self.pop[worst] = candidate
