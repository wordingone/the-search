"""
Step 1396 — Full Graph Explorer (dolphin equivalent).
Leo mail 4079 (spec: docs/specs/dolphin_explorer_v1.md), 2026-03-31.
Re-implemented by Eli 2026-04-13 (prior run crashed, session loss).

Mechanism: CC segmentation + status bar masking + 5-tier priority + backward BFS.
Reference: github.com/dolphin-in-a-coma/arc-agi-3-just-explore (3rd place, 30/52 levels)

Kill: K ≤ 2 (must beat forward nav K=2/25 at 5K steps).
Tier 1: 3 random games, 10 seeds, 5K steps (smoke test).
Tier 2: 25 games, 10 seeds, 5K steps (competition eval).

Constitutional audit:
  R1: No external loss/reward. Segmentation uses color clustering (internal).
  R2: Graph growth IS the computation. No separate optimizer.
  R3: Graph grows with experience. Behavior changes as graph structure changes.
  R4: Second-exposure testable (graph from try1 helps try2).
  R5: Game is ground truth.
  R6: Each component removable (segmentation, masking, BFS).
"""

import numpy as np
import hashlib
import sys
import os

sys.path.insert(0, 'B:/M/the-search/experiments/steps')
sys.path.insert(0, 'B:/M/the-search/experiments/environments')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search')

# Constants from spec (exact dolphin thresholds)
SALIENT_COLORS = frozenset({6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
NON_SALIENT_COLORS = frozenset({0, 1, 2, 3, 4, 5})
MEDIUM_MIN = 2
MEDIUM_MAX = 32
STATUS_BAR_COLOR = 16  # masked segments
INF_DISTANCE = 10_000_000

# Priority groups
G0 = 0   # salient AND medium ("likely buttons") + all KB
G1 = 1   # medium but NOT salient
G2 = 2   # salient but NOT medium
G3 = 3   # everything else (not status bar)
G4 = 4   # status bar

N_KB = 7  # ACTION1-ACTION7


def _flood_fill_segments(frame_int):
    """BFS flood-fill segmentation on 64x64 int frame (values 0-15).

    Returns list of segment dicts:
      {pixels: list[(x,y)], color: int, area: int,
       x1: int, y1: int, x2: int, y2: int,
       is_rectangle: bool, is_status_bar: bool}
    """
    H, W = frame_int.shape
    visited = np.zeros((H, W), dtype=bool)
    segments = []

    for sy in range(H):
        for sx in range(W):
            if visited[sy, sx]:
                continue
            color = int(frame_int[sy, sx])
            # BFS from (sx, sy)
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
    """Detect and mark status bar segments (spec §1).

    For each segment touching an edge (bbox within edge_px of any border):
      - If aspect ratio >= 5:1 → status bar LINE → mark.
      - If aspect ratio < 5:1 → check for twins on same edge.
        If >= 3 twins (same area + color + is_rectangle) → mark all.
    """
    # Determine which edges a segment touches
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

    # Pass 1: mark line-like status bars (aspect >= 5:1)
    for i, seg, edges, aspect in edge_segs:
        if aspect >= min_aspect:
            seg['is_status_bar'] = True

    # Pass 2: mark dot-like status bars (>= 3 twins on same edge)
    # Group non-line edge segments by (edge, color, area, is_rectangle)
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
    """Blake2B 128-bit hash of packed masked frame.

    Pack two 4-bit pixels per byte (values 0-16 fit in 5 bits; status bar=16
    is stored as-is since the frame is hashed as bytes for comparison).
    Shape is embedded in the digest via person tag.
    """
    H, W = masked_frame.shape
    flat = masked_frame.flatten().astype(np.uint8)
    # Pack pairs of pixels into bytes (simple, consistent)
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
    """Backward BFS from frontier.

    Frontier = all node hashes with untested actions in active_group.
    Returns dict[hash → distance_to_nearest_frontier].
    """
    frontier = set()
    for h, node in nodes.items():
        for i, g in enumerate(node['groups']):
            if g == active_group and i not in node['tested']:
                frontier.add(h)
                break

    distances = {}
    if not frontier:
        return distances  # empty → all INF

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
    'step': 1396,
    'mechanism': 'dolphin_graph_explorer',
    'description': 'CC segmentation + status bar masking + 5-tier priority + backward BFS',
}


class DolphinExplorer:
    """Full graph explorer — dolphin-equivalent for ARC-AGI-3.

    Implements spec: docs/specs/dolphin_explorer_v1.md
    """

    def __init__(self):
        self._supports_click = False
        self._n_actions = N_KB
        self._reset_graph()

    def _reset_graph(self):
        # nodes[hash] = {'actions': list[int], 'groups': list[int], 'tested': set[int]}
        # actions[i] = global action index; groups[i] = priority group; tested = local indices used
        self._nodes = {}
        # edges[(src_hash, local_idx)] = dst_hash
        self._edges = {}
        # rev_edges[dst_hash] = [(src_hash, local_idx)]
        self._rev_edges = {}
        self._active_group = 0
        self._current_hash = None
        self._last_local_action = None
        self._step = 0

    def set_game(self, n_actions):
        self._supports_click = (n_actions > N_KB)
        self._n_actions = n_actions
        # Don't reset graph on set_game — called before first process()

    def on_level_transition(self):
        """Reset graph. Called on level increment AND on game_over reset."""
        self._reset_graph()

    def process(self, obs):
        """Select action from graph exploration.

        Args:
            obs: float32 array, shape (1, 64, 64) or (64, 64)
        Returns:
            int: global action index
        """
        self._step += 1

        # 1. Extract 64x64 int frame
        frame_int = self._extract_frame(obs)

        # Handle non-ARC observations (e.g. MBPP 1D array)
        if frame_int is None:
            return np.random.randint(0, max(self._n_actions, 1))

        # 2. Segment frame (flood-fill BFS)
        segments = _flood_fill_segments(frame_int)

        # 3. Detect and mark status bars
        segments = _detect_status_bars(segments)

        # 4. Build masked frame
        masked_frame = _build_masked_frame(frame_int, segments)

        # 5. Hash masked frame
        node_hash = _hash_frame(masked_frame)

        # 6. Record transition from previous node
        if (self._current_hash is not None
                and self._last_local_action is not None
                and self._current_hash != node_hash):
            self._record_edge(self._current_hash, self._last_local_action, node_hash)

        # 7. Add node if new
        if node_hash not in self._nodes:
            actions, groups = self._build_action_space(segments)
            self._nodes[node_hash] = {
                'actions': actions,
                'groups': groups,
                'tested': set(),
            }
            self._rev_edges.setdefault(node_hash, [])

        self._current_hash = node_hash

        # 8. Check if group advancement needed (current node unreachable from frontier)
        distances = self._recompute_distances()
        if node_hash not in distances and self._active_group < G4:
            self._active_group += 1
            distances = self._recompute_distances()

        # 9. Select action
        local_action = self._select_action(node_hash, distances)
        self._last_local_action = local_action

        node = self._nodes[node_hash]
        if local_action is not None and local_action < len(node['actions']):
            global_action = node['actions'][local_action]
            # Mark as tested
            node['tested'].add(local_action)
            return global_action

        # Fallback: random KB action
        return np.random.randint(0, N_KB)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_frame(self, obs):
        """Extract 64x64 int frame from obs array.

        Returns None if obs is not a 2D or 3D array with shape (..., 64, 64).
        (Non-ARC environments like MBPP pass 1D observations.)
        """
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-2:] == (64, 64):
            frame = arr[0]  # (1, 64, 64) → (64, 64)
        elif arr.ndim == 2 and arr.shape == (64, 64):
            frame = arr
        else:
            return None
        return frame.astype(np.int32)

    def _build_action_space(self, segments):
        """Build per-node action list with priority groups.

        KB actions (0..N_KB-1) are always G0.
        Click actions are derived from segment centers, assigned G0-G4.
        Click actions added only when game supports clicks.
        """
        actions = list(range(N_KB))  # KB first
        groups = [G0] * N_KB

        if self._supports_click:
            for seg in segments:
                g = _priority_group(seg)
                actions.append(_segment_to_click_action(seg))
                groups.append(g)

        return actions, groups

    def _record_edge(self, src_hash, local_idx, dst_hash):
        """Record a transition edge src→dst via local_idx."""
        key = (src_hash, local_idx)
        if key not in self._edges:
            self._edges[key] = dst_hash
            self._rev_edges.setdefault(dst_hash, [])
            self._rev_edges[dst_hash].append((src_hash, local_idx))

    def _recompute_distances(self):
        """Compute backward BFS distances from frontier for active group."""
        return _compute_distances(
            self._nodes, self._edges, self._rev_edges, self._active_group
        )

    def _select_action(self, node_hash, distances):
        """Select local action index for current node.

        If node is on frontier (has untested actions in active group):
            Pick random untested action from active group.
        Else:
            Navigate toward frontier via lowest-distance edge.
        """
        node = self._nodes[node_hash]
        actions = node['actions']
        groups = node['groups']
        tested = node['tested']

        # Find untested actions in active group
        untested_in_group = [
            i for i, g in enumerate(groups)
            if g == self._active_group and i not in tested
        ]

        if untested_in_group:
            # Frontier node: pick random untested action
            return np.random.choice(untested_in_group)

        # Not on frontier: navigate toward frontier
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

        # No known path: pick any untested action regardless of group
        all_untested = [i for i in range(len(actions)) if i not in tested]
        if all_untested:
            return np.random.choice(all_untested)

        # All tested — pick random
        return np.random.randint(0, max(len(actions), 1))
