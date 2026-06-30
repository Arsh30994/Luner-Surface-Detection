"""A* rover traverse planning over terrain safety constraints."""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np


def _validate_arrays(slope_deg: np.ndarray, hard_unsafe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope = np.asarray(slope_deg, dtype=np.float32)
    unsafe = np.asarray(hard_unsafe, dtype=bool)
    if slope.ndim != 2 or unsafe.ndim != 2:
        raise ValueError("slope_deg and hard_unsafe must be 2-D arrays.")
    if slope.shape != unsafe.shape:
        raise ValueError("slope_deg and hard_unsafe shapes must match.")
    if not np.isfinite(slope).all():
        raise ValueError("slope_deg contains NaN or infinite values.")
    return slope, unsafe


def _build_cost_map(
    slope_deg: np.ndarray,
    hard_unsafe: np.ndarray,
    slope_caution_deg: float = 5.0,
    slope_redline_deg: float = 15.0,
) -> np.ndarray:
    """Convert terrain constraints into positive traversal costs."""
    slope, unsafe = _validate_arrays(slope_deg, hard_unsafe)
    if not (0 <= slope_caution_deg < slope_redline_deg):
        raise ValueError("slope_caution_deg must be lower than slope_redline_deg.")

    caution_span = max(slope_redline_deg - slope_caution_deg, 1e-6)
    caution_penalty = np.clip((slope - slope_caution_deg) / caution_span, 0.0, 1.0)
    steep_penalty = np.clip(slope / max(slope_redline_deg, 1e-6), 0.0, 2.0)
    cost = 1.0 + 2.6 * caution_penalty ** 2 + 1.4 * steep_penalty
    cost = cost.astype(np.float32)
    cost[unsafe] = np.inf
    return cost


def _in_bounds(shape: tuple[int, int], row: int, col: int) -> bool:
    h, w = shape
    return 0 <= row < h and 0 <= col < w


def _nearest_traversable(cost: np.ndarray, row: int, col: int, max_radius: int | None = None) -> tuple[int, int] | None:
    """Breadth-first search for the closest finite-cost cell."""
    h, w = cost.shape
    if not _in_bounds(cost.shape, row, col):
        return None
    if np.isfinite(cost[row, col]):
        return int(row), int(col)

    if max_radius is None:
        max_radius = max(h, w) // 3

    visited = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int, int]] = deque([(int(row), int(col), 0)])
    visited[row, col] = True
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while q:
        r, c, d = q.popleft()
        if d > max_radius:
            continue
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not _in_bounds(cost.shape, nr, nc) or visited[nr, nc]:
                continue
            if np.isfinite(cost[nr, nc]):
                return int(nr), int(nc)
            visited[nr, nc] = True
            q.append((nr, nc, d + 1))
    return None


def _astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], pixel_size_m: float) -> list[tuple[int, int]] | None:
    h, w = cost.shape
    start_idx = start[0] * w + start[1]
    goal_idx = goal[0] * w + goal[1]

    dist = np.full(h * w, np.inf, dtype=np.float64)
    prev = np.full(h * w, -1, dtype=np.int64)
    closed = np.zeros(h * w, dtype=bool)

    def heuristic(r: int, c: int) -> float:
        return float(np.hypot(r - goal[0], c - goal[1]) * pixel_size_m)

    dist[start_idx] = 0.0
    heap: list[tuple[float, float, int, int]] = [(heuristic(*start), 0.0, start[0], start[1])]
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)), (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)), (1, 1, np.sqrt(2.0)),
    ]

    while heap:
        _, g, r, c = heapq.heappop(heap)
        idx = r * w + c
        if closed[idx]:
            continue
        closed[idx] = True
        if idx == goal_idx:
            path: list[tuple[int, int]] = []
            cur = goal_idx
            while cur != -1:
                path.append((int(cur // w), int(cur % w)))
                cur = int(prev[cur])
            path.reverse()
            return path

        for dr, dc, step_mult in neighbors:
            nr, nc = r + dr, c + dc
            if not _in_bounds(cost.shape, nr, nc):
                continue
            nidx = nr * w + nc
            if closed[nidx] or not np.isfinite(cost[nr, nc]):
                continue
            step = step_mult * pixel_size_m * 0.5 * (float(cost[r, c]) + float(cost[nr, nc]))
            tentative = g + step
            if tentative < dist[nidx]:
                dist[nidx] = tentative
                prev[nidx] = idx
                heapq.heappush(heap, (tentative + heuristic(nr, nc), tentative, nr, nc))
    return None


def _dijkstra_tree(cost: np.ndarray, start: tuple[int, int], pixel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Shortest-path tree from start to every reachable finite-cost cell."""
    h, w = cost.shape
    start_idx = start[0] * w + start[1]
    dist = np.full(h * w, np.inf, dtype=np.float64)
    prev = np.full(h * w, -1, dtype=np.int64)
    closed = np.zeros(h * w, dtype=bool)
    dist[start_idx] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)), (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)), (1, 1, np.sqrt(2.0)),
    ]

    while heap:
        g, r, c = heapq.heappop(heap)
        idx = r * w + c
        if closed[idx]:
            continue
        closed[idx] = True
        for dr, dc, step_mult in neighbors:
            nr, nc = r + dr, c + dc
            if not _in_bounds(cost.shape, nr, nc):
                continue
            nidx = nr * w + nc
            if closed[nidx] or not np.isfinite(cost[nr, nc]):
                continue
            step = step_mult * pixel_size_m * 0.5 * (float(cost[r, c]) + float(cost[nr, nc]))
            tentative = g + step
            if tentative < dist[nidx]:
                dist[nidx] = tentative
                prev[nidx] = idx
                heapq.heappush(heap, (tentative, nr, nc))
    return dist, prev


def _reconstruct_from_prev(prev: np.ndarray, end: tuple[int, int], width: int) -> list[tuple[int, int]]:
    cur = end[0] * width + end[1]
    path: list[tuple[int, int]] = []
    while cur != -1:
        path.append((int(cur // width), int(cur % width)))
        cur = int(prev[cur])
    path.reverse()
    return path


def _thin_waypoints(path: list[tuple[int, int]], max_points: int = 240) -> list[tuple[int, int]]:
    if len(path) <= max_points:
        return path
    stride = int(np.ceil(len(path) / max_points))
    sparse = path[::stride]
    if sparse[-1] != path[-1]:
        sparse.append(path[-1])
    return sparse


def _path_distance(path: list[tuple[int, int]], pixel_size_m: float) -> float:
    if len(path) < 2:
        return 0.0
    pts = np.asarray(path, dtype=np.float32)
    diffs = np.diff(pts, axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum() * pixel_size_m)


def _bfs_reachable(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    """
    Cheap unweighted BFS reachability check: is `goal` reachable from `start`
    at all, ignoring cost magnitude (only inf-cost cells block movement)?

    This lets `plan_rover_path` skip a full weighted A* search (which has to
    exhaustively expand the entire reachable region before it can prove a
    goal is unreachable) and jump straight to the Dijkstra-tree fallback
    when the literal goal cell is topologically isolated. BFS with an
    early-exit on the goal is far cheaper than that failed A* attempt.
    """
    h, w = cost.shape
    sr, sc = start
    gr, gc = goal
    if not np.isfinite(cost[sr, sc]) or not np.isfinite(cost[gr, gc]):
        return False

    visited = np.zeros((h, w), dtype=bool)
    visited[sr, sc] = True
    q: deque[tuple[int, int]] = deque([(sr, sc)])
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            return True
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not _in_bounds(cost.shape, nr, nc) or visited[nr, nc]:
                continue
            if not np.isfinite(cost[nr, nc]):
                continue
            visited[nr, nc] = True
            q.append((nr, nc))
    return False


def plan_rover_path(
    slope_deg: np.ndarray,
    hard_unsafe: np.ndarray,
    pixel_size_m: float,
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
    slope_caution_deg: float = 5.0,
    slope_redline_deg: float = 15.0,
) -> dict:
    """Plan a safe traverse from landing ellipse centre to target crater."""
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive.")

    slope, unsafe = _validate_arrays(slope_deg, hard_unsafe)
    cost = _build_cost_map(slope, unsafe, slope_caution_deg, slope_redline_deg)
    shape = cost.shape

    requested_start = (int(start_row), int(start_col))
    requested_goal = (int(goal_row), int(goal_col))
    if not _in_bounds(shape, *requested_start):
        return {
            "path_found": False,
            "diagnosis": {"reason": "Start pixel is outside the DEM bounds."},
            "waypoints": [],
        }
    if not _in_bounds(shape, *requested_goal):
        return {
            "path_found": False,
            "diagnosis": {"reason": "Goal pixel is outside the DEM bounds."},
            "waypoints": [],
        }

    start = _nearest_traversable(cost, *requested_start)
    goal = _nearest_traversable(cost, *requested_goal)
    if start is None:
        return {
            "path_found": False,
            "diagnosis": {"reason": "No traversable terrain near the selected landing site."},
            "waypoints": [],
        }
    if goal is None:
        return {
            "path_found": False,
            "diagnosis": {"reason": "No traversable terrain near the selected crater target."},
            "waypoints": [],
        }

    # Cheap unweighted reachability probe first: if the literal goal cell is
    # topologically isolated, a full weighted A* search would have to expand
    # almost the entire graph just to discover that — skip straight to the
    # Dijkstra-tree fallback instead, which we need to run anyway to find
    # the nearest reachable access point.
    goal_reachable = _bfs_reachable(cost, start, goal)
    path = _astar(cost, start, goal, pixel_size_m) if goal_reachable else None
    access_goal = goal
    goal_standoff_m = 0.0
    fallback_used = False
    if path is None:
        dist_tree, prev_tree = _dijkstra_tree(cost, start, pixel_size_m)
        reachable = np.isfinite(dist_tree.reshape(cost.shape)) & np.isfinite(cost)
        if not reachable.any():
            return {
                "path_found": False,
                "diagnosis": {"reason": "No traversable cells are reachable from the selected landing site."},
                "waypoints": [],
                "start": {"row": start[0], "col": start[1]},
                "goal": {"row": goal[0], "col": goal[1]},
            }

        rr, cc = np.where(reachable)
        standoff_px = np.hypot(rr - requested_goal[0], cc - requested_goal[1])
        max_access_px = max(4.0, min(cost.shape) * 0.45)
        best_i = int(np.argmin(standoff_px))
        if standoff_px[best_i] > max_access_px:
            return {
                "path_found": False,
                "diagnosis": {"reason": "No reachable access point is close enough to the crater target."},
                "waypoints": [],
                "start": {"row": start[0], "col": start[1]},
                "goal": {"row": goal[0], "col": goal[1]},
            }

        access_goal = (int(rr[best_i]), int(cc[best_i]))
        goal_standoff_m = float(standoff_px[best_i] * pixel_size_m)
        path = _reconstruct_from_prev(prev_tree, access_goal, cost.shape[1])
        fallback_used = True

    raw_distance_m = _path_distance(path, pixel_size_m)
    path_rows = np.asarray([p[0] for p in path], dtype=np.int32)
    path_cols = np.asarray([p[1] for p in path], dtype=np.int32)
    path_slopes = slope[path_rows, path_cols]
    waypoints = _thin_waypoints(path)

    adjusted = start != requested_start or goal != requested_goal
    reason = "Path found successfully."
    if fallback_used:
        reason = "Exact target is isolated; routed to nearest reachable crater access point."
    elif adjusted:
        reason = "Path found; start/goal snapped to nearest traversable cells."

    return {
        "path_found": True,
        "waypoints": [{"row": int(r), "col": int(c)} for r, c in waypoints],
        "n_waypoints": len(waypoints),
        "n_path_pixels": len(path),
        "total_distance_m": round(raw_distance_m, 1),
        "mean_slope_deg": round(float(path_slopes.mean()), 2),
        "max_slope_deg": round(float(path_slopes.max()), 2),
        "start": {"row": int(start[0]), "col": int(start[1])},
        "goal": {"row": int(access_goal[0]), "col": int(access_goal[1])},
        "requested_start": {"row": requested_start[0], "col": requested_start[1]},
        "requested_goal": {"row": requested_goal[0], "col": requested_goal[1]},
        "goal_standoff_m": round(goal_standoff_m, 1),
        "diagnosis": {"reason": reason},
    }
