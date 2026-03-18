#!/usr/bin/env python3
"""
Step 452 — Online Adaptive Partition Tree + Graph (Architecture Family #5).
kd-tree splitting: axis-aligned splits on highest-variance dimension.
No cosine, no prototypes, no sphere. Adaptive density via splitting.
Tests U27: is adaptive density sufficient, or is cosine attract specifically required?

Ban check (all 4 pass):
1. No cosine matching — axis-aligned threshold comparisons
2. Not LVQ — no prototypes, no competitive learning
3. Not codebook+X — decision tree, not prototype set
4. No spatial engine on sphere — splits in raw space

T values: 50, 30, 100. 3 seeds each. LS20, 10K steps.
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


class TreeNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.split_dim = -1   # -1 = leaf
        self.split_val = 0.0
        self.left = None
        self.right = None
        # Welford stats (leaf only, kept after split but ignored)
        self.visit_count = 0
        self.n = 0
        self.mean = np.zeros(256, dtype=np.float32)
        self.M2 = np.zeros(256, dtype=np.float32)

    def update(self, x):
        """Welford online mean/variance update."""
        self.visit_count += 1
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        if self.n < 2:
            return np.zeros(256, dtype=np.float32)
        return self.M2 / (self.n - 1)


class PartitionTree:
    def __init__(self, n_actions=4, split_threshold=50, max_depth=15):
        self._next_id = 0
        self.root = self._new_node()
        self.n_actions = n_actions
        self.split_threshold = split_threshold
        self.max_depth = max_depth
        self.edges = {}          # {(leaf_id, action): {next_leaf_id: count}}
        self.prev_leaf_id = None
        self.prev_action = None
        self.step_count = 0
        self.leaf_ids_seen = set()
        self.split_count = 0

    def _new_node(self):
        n = TreeNode(self._next_id)
        self._next_id += 1
        return n

    def _find_leaf(self, obs):
        node = self.root
        d = 0
        while node.split_dim != -1 and d < self.max_depth:
            if obs[node.split_dim] <= node.split_val:
                node = node.left
            else:
                node = node.right
            d += 1
        return node

    def _split(self, leaf):
        """Split leaf on highest-variance dimension. Edges from leaf are lost."""
        var = leaf.variance()
        dim = int(np.argmax(var))
        val = float(leaf.mean[dim])
        leaf.split_dim = dim
        leaf.split_val = val
        leaf.left = self._new_node()
        leaf.right = self._new_node()
        # Remove outgoing edges (children rebuild from scratch)
        for a in range(self.n_actions):
            self.edges.pop((leaf.node_id, a), None)
        self.split_count += 1

    def step(self, obs):
        self.step_count += 1
        leaf = self._find_leaf(obs)
        leaf.update(obs)
        self.leaf_ids_seen.add(leaf.node_id)

        # Graph edge: prev → current
        if self.prev_leaf_id is not None and self.prev_action is not None:
            key = (self.prev_leaf_id, self.prev_action)
            d = self.edges.setdefault(key, {})
            d[leaf.node_id] = d.get(leaf.node_id, 0) + 1

        # Action: least-visited outgoing edge from current leaf
        visit_counts = [
            sum(self.edges.get((leaf.node_id, a), {}).values())
            for a in range(self.n_actions)
        ]
        min_count = min(visit_counts)
        candidates = [a for a, c in enumerate(visit_counts) if c == min_count]
        action = candidates[int(np.random.randint(len(candidates)))]

        self.prev_leaf_id = leaf.node_id
        self.prev_action = action

        # Growth: split if leaf (split_dim==-1) and visit_count > threshold.
        # Guard: _find_leaf can return internal node at max_depth — don't re-split.
        if leaf.split_dim == -1 and leaf.visit_count > self.split_threshold and leaf.n >= 2:
            self._split(leaf)
            # prev_leaf_id now points to an internal node; next step routes to child

        return action


def run_seed(arc, game_id, seed, split_threshold, max_steps=10000):
    from arcengine import GameState
    np.random.seed(seed)
    g = PartitionTree(n_actions=4, split_threshold=split_threshold, max_depth=15)
    env = arc.make(game_id)
    obs = env.reset()
    na = len(env.action_space)
    ts = go = lvls = 0
    action_counts = [0] * na
    level_step = None
    t0 = time.time()

    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue

        pooled = avgpool16(obs.frame)
        action_idx = g.step(pooled)

        action_counts[action_idx % na] += 1
        action = env.action_space[action_idx % na]
        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts

        if time.time() - t0 > 280: break

    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    ratio = len(g.leaf_ids_seen) / max(g.step_count, 1)
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_leaves': len(g.leaf_ids_seen), 'ratio': ratio,
        'dom': dom, 'splits': g.split_count,
        'elapsed': time.time() - t0,
    }


def main():
    import arc_agi
    print("Step 452: Online Adaptive Partition Tree + Graph. 3 T values x 3 seeds x 10K steps.", flush=True)
    print("Axis-aligned kd-tree splits. No cosine. Tests U27 (adaptive density).", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    thresholds = [50, 30, 100]
    all_results = {}

    for T in thresholds:
        print(f"\n--- T={T} ---", flush=True)
        results = []
        for seed in [0, 1, 2]:
            r = run_seed(arc, ls20.game_id, seed=seed, split_threshold=T, max_steps=10000)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  leaves={r['unique_leaves']:4d}"
                  f"  ratio={r['ratio']:.3f}  dom={r['dom']:.0f}%"
                  f"  splits={r['splits']}  {r['elapsed']:.0f}s", flush=True)
            results.append(r)
        wins = [r for r in results if r['levels'] > 0]
        avg_ratio = sum(r['ratio'] for r in results) / len(results)
        avg_leaves = sum(r['unique_leaves'] for r in results) / len(results)
        avg_splits = sum(r['splits'] for r in results) / len(results)
        print(f"  -> {len(wins)}/3  ratio={avg_ratio:.3f}  avg_leaves={avg_leaves:.0f}"
              f"  avg_splits={avg_splits:.0f}", flush=True)
        all_results[T] = {
            'wins': len(wins), 'avg_ratio': avg_ratio,
            'avg_leaves': avg_leaves, 'avg_splits': avg_splits,
            'level_steps': sorted([r['level_step'] for r in wins]),
        }

    print(f"\n{'='*60}", flush=True)
    print("STEP 452 RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'T':<6} {'Wins':<6} {'Ratio':<8} {'Leaves':<8} {'Splits':<8} {'Steps'}", flush=True)
    for T in thresholds:
        rr = all_results[T]
        print(f"  T={T:<3}  {rr['wins']}/3   {rr['avg_ratio']:.3f}   "
              f"{rr['avg_leaves']:<8.0f} {rr['avg_splits']:<8.0f} {rr['level_steps']}", flush=True)

    print(f"\nBaselines:", flush=True)
    print(f"  Random walk:  1/10 at 10K (step 1329)", flush=True)
    print(f"  Grid graph:   0/3  at 30K (ratio=0.19)", flush=True)
    print(f"  Cosine graph: 3/10 at 30K (step ~25K, ratio=0.07)", flush=True)

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)

    print(f"\nVERDICT:", flush=True)
    for T in thresholds:
        rr = all_results[T]
        if 0.02 <= rr['avg_ratio'] <= 0.15:
            dyn = "HEALTHY dynamics"
        elif rr['avg_ratio'] < 0.02:
            dyn = f"TOO COARSE (ratio={rr['avg_ratio']:.3f})"
        else:
            dyn = f"TOO SENSITIVE (ratio={rr['avg_ratio']:.3f})"
        nav = f"NAVIGATES ({rr['wins']}/3)" if rr['wins'] > 0 else "no navigation at 10K"
        print(f"  T={T}: {dyn}, {nav}", flush=True)


if __name__ == '__main__':
    main()
