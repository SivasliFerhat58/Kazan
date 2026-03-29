import heapq
import math
import numpy as np
from typing import Optional

RISK_WEIGHT: float = 10.0

OBSTACLE_RISK_THRESHOLD: float = 0.95

COST_ORTHO: float = 1.0
COST_DIAG:  float = math.sqrt(2)   

_NEIGHBOURS = [
    (-1,  0, COST_ORTHO),   
    ( 1,  0, COST_ORTHO),   
    ( 0, -1, COST_ORTHO),   
    ( 0,  1, COST_ORTHO),   
    (-1, -1, COST_DIAG),    
    (-1,  1, COST_DIAG),    
    ( 1, -1, COST_DIAG),    
    ( 1,  1, COST_DIAG),    
]

class _Node:
    __slots__ = ("f", "g", "row", "col")

    def __init__(self, f: float, g: float, row: int, col: int) -> None:
        self.f   = f
        self.g   = g
        self.row = row
        self.col = col

    def __lt__(self, other: "_Node") -> bool:
        return self.f < other.f

def plan_path(
    risk_map:     np.ndarray,
    obstacle_map: np.ndarray,
    start:        tuple[int, int],
    goal:         tuple[int, int],
    risk_weight:  float = RISK_WEIGHT,
) -> Optional[list[tuple[int, int]]]:
    H, W = risk_map.shape

    sx, sy = start
    gx, gy = goal

    s_row, s_col = sy, sx
    g_row, g_col = gy, gx

    for name, r, c in [("start", s_row, s_col), ("goal", g_row, g_col)]:
        if not (0 <= r < H and 0 <= c < W):
            raise ValueError(
                f"[path_planner] {name} point ({c},{r}) is outside "
                f"grid bounds {W}×{H}."
            )

    if obstacle_map[s_row, s_col] > 0.5:
        raise ValueError("[path_planner] Start cell is on an obstacle.")
    if obstacle_map[g_row, g_col] > 0.5:
        raise ValueError("[path_planner] Goal cell is on an obstacle.")

    def heuristic(row: int, col: int) -> float:
        return math.hypot(col - g_col, row - g_row)

    g_cost = np.full((H, W), np.inf, dtype=np.float64)
    g_cost[s_row, s_col] = 0.0

    parent: dict[tuple[int, int], tuple[int, int]] = {}

    closed = np.zeros((H, W), dtype=bool)

    h0 = heuristic(s_row, s_col)
    open_heap: list[_Node] = []
    heapq.heappush(open_heap, _Node(h0, 0.0, s_row, s_col))

    iterations = 0

    while open_heap:
        current = heapq.heappop(open_heap)
        cr, cc  = current.row, current.col

        if closed[cr, cc]:
            continue
        closed[cr, cc] = True
        iterations += 1

        if cr == g_row and cc == g_col:
            path = _reconstruct_path(parent, g_row, g_col, s_row, s_col)
            return path  

        for dr, dc, move_cost in _NEIGHBOURS:
            nr, nc = cr + dr, cc + dc

            if not (0 <= nr < H and 0 <= nc < W):
                continue

            if closed[nr, nc]:
                continue

            if obstacle_map[nr, nc] > 0.5:
                continue

            cell_risk = float(risk_map[nr, nc])
            if cell_risk >= OBSTACLE_RISK_THRESHOLD:
                continue

            risk_penalty = cell_risk * risk_weight
            tentative_g  = g_cost[cr, cc] + move_cost + risk_penalty

            if tentative_g < g_cost[nr, nc]:
                g_cost[nr, nc] = tentative_g
                parent[(nr, nc)] = (cr, cc)
                f = tentative_g + heuristic(nr, nc)
                heapq.heappush(open_heap, _Node(f, tentative_g, nr, nc))

    return None

def _reconstruct_path(
    parent: dict,
    g_row: int, g_col: int,
    s_row: int, s_col: int,
) -> list[tuple[int, int]]:
    path = []
    r, c = g_row, g_col
    while (r, c) != (s_row, s_col):
        path.append((c, r))   
        r, c = parent[(r, c)]
    path.append((s_col, s_row))
    path.reverse()
    return path
