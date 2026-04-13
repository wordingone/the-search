"""
Step 1397 — Dolphin v2: Persist graph across GAME_OVER.
Leo mail 4550 (Part B), 2026-04-13.

Single variable change from v1 (step1396):
  on_level_transition() no longer resets graph unconditionally.
  After the transition, hash the first new frame:
    - Hash matches existing node → GAME_OVER restart → keep graph, navigate from match
    - Hash is new → genuine level advance → reset graph (new state space)

This recovers cross-episode knowledge lost in v1 for SC25, SU15, DC22.

Constitutional audit (same as v1):
  R1: No external loss/reward.
  R2: Graph growth IS the computation.
  R3: Graph grows with experience. Persisting across GAME_OVER increases persistence.
  R4: Second-exposure testable (graph from ep1 helps ep2).
  R5: Game is ground truth.
  R6: Each component removable.
"""

import numpy as np
import hashlib
import sys
import os

sys.path.insert(0, 'B:/M/the-search/experiments/steps')
sys.path.insert(0, 'B:/M/the-search/experiments/environments')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search')

# Constants (identical to v1)
SALIENT_COLORS = frozenset({6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
NON_SALIENT_COLORS = frozenset({0, 1, 2, 3, 4, 5})
MEDIUM_MIN = 2
MEDIUM_MAX = 32
STATUS_BAR_COLOR = 16
INF_DISTANCE = 10_000_000

G0 = 0
G1 = 1
G2 = 2
G3 = 3
G4 = 4

N_KB = 7


def _flood_fill_segments(frame_int):
    """BFS flood-fill segmentation on 64x64 int frame (values 0-15)."""
    H, W = frame_int.shape
    visited = np.zeros((H, W), dtype=bool)
    segments = []

    for sy in range(H):
        for sx in range(W):
            if visited[sy, sx]:
                continue
            color = int(frame_int[sy, sx])
            pixels = []
            queue = [(sx, sy)]
            visited[sy, sx] = True
            while queue:
                x, y = queue.pop()
                pixels.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and not visited[ny, nx]:
                        if int(frame_int[ny, nx]) == color:
                            visited[ny, nx] = True
                            queue.append((nx, ny))

            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            area = len(pixels)
            is_rect = (area == w * h)

            segments.append({
                'pixels': pixels,
                'color': color,
                'area': area,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'is_rectangle': is_rect,
                'is_status_bar': False,
            })

    return segments


def _detect_status_bars(segments, frame_h=64, frame_w=64, edge_px=3, min_aspect=5.0):
    """Detect and mark status bar segments."""
    def touching_edges(seg):
        edges = set()
        if seg['y1'] <= edge_px:
            edges.add('top')
        if seg['y2'] >= frame_h - 1 - edge_px:
            edges.add('bottom')
        if seg['x1'] <= edge_px:
            edges.add('left')
        if seg['x2'] >= frame_w - 1 - edge_px:
            edges.add('right')
        return edges

    edge_segs = []
    for i, seg in enumerate(segments):
        e = touching_edges(seg)
        if e:
            w = seg['x2'] - seg['x1'] + 1
            h = seg['y2'] - seg['y1'] + 1
            aspect = max(w, h) / max(min(w, h), 1)
            edge_segs.append((i, seg, e, aspect))

    for i, seg, edges, aspect in edge_segs:
        if aspect >= min_aspect:
            seg['is_status_bar'] = True

    from collections import defaultdict
    groups = defaultdict(list)
    for i, seg, edges, aspect in edge_segs:
        if seg['is_status_bar']:
            continue
        for edge in edges:
            key = (edge, seg['color'], seg['area'], seg['is_rectangle'])
            groups[key].append(i)

    for key, idxs in groups.items():
        if len(idxs) >= 3:
            for i in idxs:
                segments[i]['is_status_bar'] = True

    return segments


def _build_masked_frame(frame_int, segments):
    """Replace status bar pixels with STATUS_BAR_COLOR (16)."""
    masked = frame_int.copy()
    for seg in segments:
        if seg['is_status_bar']:
            for x, y in seg['pixels']:
                masked[y, x] = STATUS_BAR_COLOR
    return masked


def _hash_frame(masked_frame):
    """Blake2B 128-bit hash of packed masked frame."""
    H, W = masked_frame.shape
    flat = masked_frame.flatten().astype(np.uint8)
    n = len(flat)
    if n % 2:
        flat = np.append(flat, 0)
    packed = flat[0::2] << 4 | flat[1::2]
    tag = f"{H}x{W}".encode()
    return hashlib.blake2b(packed.tobytes(), digest_size=16, person=tag[:16]).hexdigest()


def _priority_group(seg):
    """Compute priority group for a segment's click action."""
    if seg['is_status_bar']:
        return G4
    color = seg['color']
    w = seg['x2'] - seg['x1'] + 1
    h = seg['y2'] - seg['y1'] + 1
    is_salient = color in SALIENT_COLORS
    is_medium = (MEDIUM_MIN <= w <= MEDIUM_MAX) and (MEDIUM_MIN <= h <= MEDIUM_MAX)
    if is_salient and is_medium:
        return G0
    if is_medium and not is_salient:
        return G1
    if is_salient and not is_medium:
        return G2
    return G3


def _segment_to_click_action(seg):
    """Pick a random pixel within segment and encode as global action int."""
    pixels = seg['pixels']
    x, y = pixels[np.random.randint(len(pixels))]
    click_idx = y * 64 + x
    return N_KB + click_idx


def _compute_distances(nodes, edges, rev_edges, active_group):
    """Backward BFS from frontier."""
    frontier = set()
    for h, node in nodes.items():
        for i, g in enumerate(node['groups']):
            if g == active_group and i not in node['tested']:
                frontier.add(h)
                break

    distances = {}
    if not frontier:
        return distances

    queue = list(frontier)
    for h in frontier:
        distances[h] = 0

    head = 0
    while head < len(queue):
        h = queue[head]
        head += 1
        d = distances[h]
        for src_hash, local_idx in rev_edges.get(h, []):
            if src_hash not in distances:
                distances[src_hash] = d + 1
                queue.append(src_hash)

    return distances


CONFIG = {
    'step': 1397,
    'mechanism': 'dolphin_v2_persist_game_over',
    'description': 'v1 + persist graph across GAME_OVER (single variable)',
}


class DolphinV2:
    """Dolphin v2: graph persists across GAME_OVER restarts.

    v1 change: on_level_transition() sets a pending flag instead of
    resetting the graph. On the next process() call, the frame hash
    determines whether this is a GAME_OVER (hash seen before → keep graph)
    or a genuine level advance (new hash → reset graph).
    """

    def __init__(self):
        self._supports_click = False
        self._n_actions = N_KB
        self._pending_transition = False
        self._reset_graph()

    def _reset_graph(self):
        self._nodes = {}
        self._edges = {}
        self._rev_edges = {}
        self._active_group = 0
        self._current_hash = None
        self._last_local_action = None
        self._step = 0
        # Note: _pending_transition is NOT reset here — it's set by on_level_transition()
        # and consumed by process(). Resetting here would erase pending transition info.

    def set_game(self, n_actions):
        self._supports_click = (n_actions > N_KB)
        self._n_actions = n_actions

    def on_level_transition(self):
        """v2: Set pending flag instead of resetting graph.

        Called by harness on level increment AND on GAME_OVER reset.
        The actual decision (keep or reset) is deferred to process()
        when we can inspect the first frame of the new episode.
        """
        self._pending_transition = True
        # Clear navigation state — we don't know where we are yet
        self._current_hash = None
        self._last_local_action = None

    def process(self, obs):
        """Select action from graph exploration."""
        self._step += 1

        frame_int = self._extract_frame(obs)

        if frame_int is None:
            return np.random.randint(0, max(self._n_actions, 1))

        segments = _flood_fill_segments(frame_int)
        segments = _detect_status_bars(segments)
        masked_frame = _build_masked_frame(frame_int, segments)
        node_hash = _hash_frame(masked_frame)

        # v2: Resolve pending transition — GAME_OVER vs new level
        if self._pending_transition:
            self._pending_transition = False
            if node_hash in self._nodes:
                # Hash matches existing node → GAME_OVER restart
                # Keep graph, navigate from matched initial frame
                # (fall through — no edge recorded, _current_hash set below)
                pass
            else:
                # New hash → genuine level advance → reset graph
                self._reset_graph()
            # In both cases: don't record an edge from the pre-transition node
            # (_current_hash and _last_local_action are None, so edge recording below is skipped)

        # Record transition from previous node (skipped when _current_hash is None)
        if (self._current_hash is not None
                and self._last_local_action is not None
                and self._current_hash != node_hash):
            self._record_edge(self._current_hash, self._last_local_action, node_hash)

        # Add node if new
        if node_hash not in self._nodes:
            actions, groups = self._build_action_space(segments)
            self._nodes[node_hash] = {
                'actions': actions,
                'groups': groups,
                'tested': set(),
            }
            self._rev_edges.setdefault(node_hash, [])

        self._current_hash = node_hash

        # Group advancement
        distances = self._recompute_distances()
        if node_hash not in distances and self._active_group < G4:
            self._active_group += 1
            distances = self._recompute_distances()

        # Select action
        local_action = self._select_action(node_hash, distances)
        self._last_local_action = local_action

        node = self._nodes[node_hash]
        if local_action is not None and local_action < len(node['actions']):
            global_action = node['actions'][local_action]
            node['tested'].add(local_action)
            return global_action

        return np.random.randint(0, N_KB)

    # ------------------------------------------------------------------
    # Private helpers (identical to v1)
    # ------------------------------------------------------------------

    def _extract_frame(self, obs):
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-2:] == (64, 64):
            return arr[0].astype(np.int32)
        elif arr.ndim == 2 and arr.shape == (64, 64):
            return arr.astype(np.int32)
        return None

    def _build_action_space(self, segments):
        actions = list(range(N_KB))
        groups = [G0] * N_KB

        if self._supports_click:
            for seg in segments:
                g = _priority_group(seg)
                actions.append(_segment_to_click_action(seg))
                groups.append(g)

        return actions, groups

    def _record_edge(self, src_hash, local_idx, dst_hash):
        key = (src_hash, local_idx)
        if key not in self._edges:
            self._edges[key] = dst_hash
            self._rev_edges.setdefault(dst_hash, [])
            self._rev_edges[dst_hash].append((src_hash, local_idx))

    def _recompute_distances(self):
        return _compute_distances(
            self._nodes, self._edges, self._rev_edges, self._active_group
        )

    def _select_action(self, node_hash, distances):
        node = self._nodes[node_hash]
        actions = node['actions']
        groups = node['groups']
        tested = node['tested']

        untested_in_group = [
            i for i, g in enumerate(groups)
            if g == self._active_group and i not in tested
        ]

        if untested_in_group:
            return np.random.choice(untested_in_group)

        best_local = None
        best_dist = INF_DISTANCE

        for (src_hash, local_idx), dst_hash in self._edges.items():
            if src_hash != node_hash:
                continue
            d = distances.get(dst_hash, INF_DISTANCE)
            if d < best_dist:
                best_dist = d
                best_local = local_idx

        if best_local is not None:
            return best_local

        all_untested = [i for i in range(len(actions)) if i not in tested]
        if all_untested:
            return np.random.choice(all_untested)

        return np.random.randint(0, max(len(actions), 1))
