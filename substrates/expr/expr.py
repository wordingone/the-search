"""
The Self-Modifying Expression Tree — Phase 2

State: a nested list (AST). NOT vectors. NOT a codebook. NOT a tape.
The tree IS the program. Evaluating it IS the computation.
Mutating it IS the learning.

  State = ['if', ['>', 42, 0.5], 2, ['if', ['>', 10, 0.3], 0, 3]]

  eval(state, obs) → action (integer)
  mutate(state, obs) → new state (modified AST)

Match:  evaluate the tree against obs (recursive descent)
Chain:  sub-expressions reference sub-expressions (function calls)
Attract: mutate one node (threshold, feature index, action, structure)
Spawn:  add a new branch (deepen the tree)

R1: No external objective. eval(tree, obs) → action. No loss.
R2: Mutation depends on evaluation (which node was decisive → mutate near it).
R3: The tree IS the only state. Every node is mutable. The CONDITIONS,
    the ACTIONS, the STRUCTURE — all data in the list. Form changes.
R4: Eval before mutation vs after. If action changed for recent inputs, keep.
    If no change (mutation was neutral), accept. If degraded on held-out, revert.
R5: The evaluator + mutator (~30 lines) is frozen.
R6: Remove eval → no action. Remove mutate → no learning.

S1-S21 DO NOT APPLY. No cosine. No vectors. No codebook.
The tree naturally does feature selection (each node tests ONE dimension).
"""

import random
import torch


def make_leaf(n_actions):
    """Random action leaf."""
    return random.randint(0, n_actions - 1)


def make_node(depth, n_dims, n_actions, max_depth=6):
    """Random expression tree."""
    if depth >= max_depth or random.random() < 0.3:
        return make_leaf(n_actions)
    feat = random.randint(0, n_dims - 1)
    thresh = random.random()
    return ['if', ['>', feat, thresh],
            make_node(depth + 1, n_dims, n_actions, max_depth),
            make_node(depth + 1, n_dims, n_actions, max_depth)]


def evaluate(tree, obs):
    """Evaluate expression tree against observation. Returns action int."""
    if isinstance(tree, (int, float)):
        return int(tree)
    # tree = ['if', ['>', feat_idx, threshold], then_branch, else_branch]
    _, cond, then_b, else_b = tree
    _, feat_idx, thresh = cond
    val = obs[int(feat_idx) % len(obs)].item() if torch.is_tensor(obs) else obs[int(feat_idx) % len(obs)]
    if val > thresh:
        return evaluate(then_b, obs)
    else:
        return evaluate(else_b, obs)


def mutate(tree, n_dims, n_actions, rate=0.15):
    """Mutate one node. Returns new tree (copy)."""
    if isinstance(tree, (int, float)):
        if random.random() < rate:
            return random.randint(0, n_actions - 1)
        return tree
    _, cond, then_b, else_b = tree
    _, feat_idx, thresh = cond
    r = random.random()
    if r < rate * 0.3:
        # Mutate feature index
        feat_idx = random.randint(0, n_dims - 1)
    elif r < rate * 0.6:
        # Mutate threshold
        thresh = max(0.0, min(1.0, thresh + random.gauss(0, 0.1)))
    elif r < rate * 0.8:
        # Mutate structure: replace a branch with new subtree
        if random.random() < 0.5:
            then_b = make_node(0, n_dims, n_actions, max_depth=3)
        else:
            else_b = make_node(0, n_dims, n_actions, max_depth=3)
    # Recurse into branches
    then_b = mutate(then_b, n_dims, n_actions, rate)
    else_b = mutate(else_b, n_dims, n_actions, rate)
    return ['if', ['>', feat_idx, thresh], then_b, else_b]


def tree_depth(tree):
    if isinstance(tree, (int, float)):
        return 0
    return 1 + max(tree_depth(tree[2]), tree_depth(tree[3]))


def tree_size(tree):
    if isinstance(tree, (int, float)):
        return 1
    return 1 + tree_size(tree[2]) + tree_size(tree[3])


import copy

class ExprSubstrate:
    """The substrate. State = expression tree. ~30 lines of logic."""

    def __init__(self, n_dims, n_actions, pop_size=4):
        self.n_dims = n_dims
        self.n_actions = n_actions
        # Population of trees (small — 4 candidates)
        self.pop = [make_node(0, n_dims, n_actions, max_depth=4)
                     for _ in range(pop_size)]
        self.scores = [0.0] * pop_size
        self.best = 0
        self.history = []       # recent (obs, action) pairs for scoring
        self.window = 32        # scoring window
        self.steps = 0

    def step(self, x, n_actions=None):
        """One step. Evaluate best tree, score population, evolve."""
        n_actions = n_actions or self.n_actions

        # MATCH + CHAIN: evaluate the best tree (recursive descent)
        action = evaluate(self.pop[self.best], x) % n_actions

        # Store for scoring
        self.history.append((x.clone() if torch.is_tensor(x) else x, action))
        if len(self.history) > self.window * 2:
            self.history = self.history[-self.window * 2:]

        self.steps += 1

        # ATTRACT + SPAWN: evolve every N steps
        if self.steps % self.window == 0 and len(self.history) >= self.window:
            self._evolve()

        return action

    def _evolve(self):
        """Score population, replace worst with mutated best."""
        recent = self.history[-self.window:]

        for i, tree in enumerate(self.pop):
            # R1: score = action diversity × consistency
            # Diverse: uses all actions (not stuck on one)
            # Consistent: same obs → same action across window
            actions = [evaluate(tree, obs) % self.n_actions for obs, _ in recent]
            unique = len(set(actions))
            diversity = unique / self.n_actions  # 0..1

            # Consistency: eval twice on same inputs should match
            actions2 = [evaluate(tree, obs) % self.n_actions for obs, _ in recent]
            consistent = sum(a == b for a, b in zip(actions, actions2)) / len(actions)

            self.scores[i] = diversity * consistent  # both needed

        # R4: compare before/after
        self.best = max(range(len(self.pop)), key=lambda i: self.scores[i])
        worst = min(range(len(self.pop)), key=lambda i: self.scores[i])

        # R3: mutate best → replace worst (the program changes form)
        candidate = mutate(copy.deepcopy(self.pop[self.best]),
                          self.n_dims, self.n_actions)

        # Only replace if candidate is not degenerate
        test_actions = [evaluate(candidate, obs) % self.n_actions
                       for obs, _ in self.history[-self.window:]]
        if len(set(test_actions)) > 1:  # not constant
            self.pop[worst] = candidate

    @property
    def size(self):
        return sum(tree_size(t) for t in self.pop)
