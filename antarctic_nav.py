"""IceRoute Guardian: deterministic, uncertainty-aware A* routing prototype.

Research prototype only: not a certified navigation system.
Python 3.10+. Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import hypot, inf, isfinite
import os
import sys
from typing import Iterable, Literal, Mapping, Sequence
import pandas as pd

RouteProfile = Literal["fastest", "safest", "balanced"]


@dataclass(frozen=True, slots=True)
class Cell:
    """One grid cell. Values are normalized/validated by Grid.validate()."""
    row: int
    col: int
    ice_concentration: float = 0.0  # 0..1
    ice_thickness_m: float = 0.0
    wind_speed_ms: float = 0.0
    visibility_km: float = 20.0
    data_age_h: float = 0.0
    blocked: bool = False


@dataclass(frozen=True, slots=True)
class Vessel:
    # Standard maritime cruising speed in knots (1 knot = 1.852 km/h)
    max_speed_knots: float = 15.0
    ice_concentration_limit: float = 0.85
    ice_thickness_limit_m: float = 0.60
    min_visibility_km: float = 0.50
    max_wind_speed_ms: float = 25.0


@dataclass(frozen=True, slots=True)
class CostWeights:
    time: float = 1.0
    risk: float = 1.0
    fuel: float = 0.15
    stale_data: float = 0.50
    turn: float = 0.05


@dataclass(frozen=True, slots=True)
class StepCost:
    total: float
    time_h: float
    risk: float
    fuel: float
    stale_data: float
    distance_nm: float  # Stored in Nautical Miles


@dataclass(frozen=True, slots=True)
class Route:
    cells: tuple[tuple[int, int], ...]
    total_cost: float
    distance_nm: float  # Output in Nautical Miles
    travel_time_h: float
    risk: float
    fuel: float
    stale_data: float
    expanded_nodes: int
    profile: str


@dataclass
class Grid:
    cells: Mapping[tuple[int, int], Cell]
    rows: int
    cols: int
    cell_size_km: float = 1.0

    def validate(self) -> None:
        if self.rows <= 0 or self.cols <= 0 or self.cell_size_km <= 0:
            raise ValueError("rows, cols, and cell_size_km must be positive")
        expected = {(r, c) for r in range(self.rows) for c in range(self.cols)}
        if set(self.cells) != expected:
            missing = sorted(expected - set(self.cells))[:5]
            extra = sorted(set(self.cells) - expected)[:5]
            raise ValueError(f"Grid must contain every cell; missing={missing}, extra={extra}")
        for key, cell in self.cells.items():
            if (cell.row, cell.col) != key:
                raise ValueError(f"Cell key mismatch at {key}")
            numeric = (cell.ice_concentration, cell.ice_thickness_m,
                       cell.wind_speed_ms, cell.visibility_km, cell.data_age_h)
            if not all(isfinite(float(x)) for x in numeric):
                raise ValueError(f"Non-finite value in cell {key}")
            if not 0 <= cell.ice_concentration <= 1:
                raise ValueError(f"ice_concentration must be in [0,1] at {key}")
            if cell.ice_thickness_m < 0 or cell.wind_speed_ms < 0 or cell.data_age_h < 0:
                raise ValueError(f"Negative physical value at {key}")
            if cell.visibility_km < 0:
                raise ValueError(f"visibility_km cannot be negative at {key}")

    def neighbors(self, node: tuple[int, int], diagonals: bool = True) -> Iterable[tuple[int, int]]:
        r, c = node
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
        if diagonals:
            offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
        for dr, dc in offsets:
            nxt = (r + dr, c + dc)
            if nxt in self.cells:
                if dr and dc and (r + dr, c) in self.cells and (r, c + dc) in self.cells:
                    if self.cells[(r + dr, c)].blocked or self.cells[(r, c + dc)].blocked:
                        continue
                yield nxt


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _hazards(cell: Cell, vessel: Vessel) -> tuple[float, float, float, float]:
    ice = _clamp(0.65 * cell.ice_concentration +
                 0.35 * _clamp(cell.ice_thickness_m / max(vessel.ice_thickness_limit_m, 1e-9)))  
    wind = _clamp(cell.wind_speed_ms / max(vessel.max_wind_speed_ms, 1e-9))  
    visibility = _clamp(1.0 - cell.visibility_km / 20.0)   
    stale = _clamp(cell.data_age_h / 24.0)                 
    return ice, wind, visibility, stale


def _blocked(cell: Cell, vessel: Vessel) -> bool:
    return (cell.blocked or
            cell.ice_concentration > vessel.ice_concentration_limit or
            cell.ice_thickness_m > vessel.ice_thickness_limit_m or
            cell.visibility_km < vessel.min_visibility_km)


def _step_cost(grid: Grid, a: Cell, b: Cell, vessel: Vessel,
               weights: CostWeights, previous: tuple[int, int] | None) -> StepCost:
    diagonal = a.row != b.row and a.col != b.col
    distance_km = grid.cell_size_km * (2 ** 0.5 if diagonal else 1.0)    
    distance_nm = distance_km / 1.852  # Convert grid km distance to Nautical Miles (NM)
    
    ice, wind, visibility, stale = _hazards(b, vessel)
    risk = 0.45 * ice + 0.25 * wind + 0.15 * visibility + 0.15 * stale
    
    # Speed in Knots with environmental degradation
    speed_knots = vessel.max_speed_knots * max(0.10, 1.0 - 0.55 * ice - 0.25 * wind - 0.20 * visibility)
    
    time_h = distance_nm / speed_knots  # Time = Distance (NM) / Speed (Knots)
    fuel = distance_nm * (1.0 + 0.8 * ice + 0.3 * wind)
    
    turn = 0.0
    if previous is not None:
        old = (a.row - previous[0], a.col - previous[1])
        new = (b.row - a.row, b.col - a.col)
        turn = 1.0 if old != new else 0.0
        
    total = (weights.time * time_h + weights.risk * risk +
             weights.fuel * fuel + weights.stale_data * stale + weights.turn * turn)
    return StepCost(total, time_h, risk, fuel, stale, distance_nm)


def _heuristic(node: tuple[int, int], goal: tuple[int, int], grid: Grid,
               vessel: Vessel, weights: CostWeights) -> float:
    distance_km = grid.cell_size_km * hypot(goal[0] - node[0], goal[1] - node[1])
    distance_nm = distance_km / 1.852
    return weights.time * distance_nm / vessel.max_speed_knots


def astar(grid: Grid, start: tuple[int, int], goal: tuple[int, int],
          vessel: Vessel | None = None, weights: CostWeights | None = None,
          profile: RouteProfile = "balanced", diagonals: bool = True,
          buffer_cells: int = 300) -> Route:
    grid.validate()
    vessel = vessel or Vessel()
    if vessel.max_speed_knots <= 0 or vessel.ice_concentration_limit < 0 or vessel.ice_thickness_limit_m < 0:
        raise ValueError("Invalid vessel limits")
    base = weights or CostWeights()
    presets = {
        "fastest": CostWeights(1.0, 0.10, 0.05, 0.10, 0.02),
        "safest": CostWeights(0.25, 3.0, 0.10, 2.0, 0.10),
        "balanced": base,
    }
    weights = presets[profile]
    if start not in grid.cells or goal not in grid.cells:
        raise ValueError("Start and goal must be grid coordinates")
    if _blocked(grid.cells[start], vessel) or _blocked(grid.cells[goal], vessel):
        raise ValueError("Start and goal must be navigable")
    if start == goal:
        return Route((start,), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, profile) 

    min_r = max(0, min(start[0], goal[0]) - buffer_cells)
    max_r = min(grid.rows - 1, max(start[0], goal[0]) + buffer_cells)
    min_c = max(0, min(start[1], goal[1]) - buffer_cells)
    max_c = min(grid.cols - 1, max(start[1], goal[1]) + buffer_cells)

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heappush(open_heap, (_heuristic(start, goal, grid, vessel, weights), counter, start))
    g: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    step_info: dict[tuple[int, int], StepCost] = {}
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    expanded = 0

    while open_heap:
        f, _, current = heappop(open_heap)
        expected = g[current] + _heuristic(current, goal, grid, vessel, weights)
        if f > expected + 1e-12:
            continue
        expanded += 1
        if current == goal:
            path: list[tuple[int, int]] = []
            cur: tuple[int, int] | None = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            totals = [step_info[n] for n in path[1:]]
            return Route(tuple(path), g[goal], sum(x.distance_nm for x in totals),
                         sum(x.time_h for x in totals), sum(x.risk for x in totals),
                         sum(x.fuel for x in totals), sum(x.stale_data for x in totals),
                         expanded, profile)

        for nxt in grid.neighbors(current, diagonals):
            if not (min_r <= nxt[0] <= max_r and min_c <= nxt[1] <= max_c):
                continue
            cell = grid.cells[nxt]
            if _blocked(cell, vessel):
                continue
            cost = _step_cost(grid, grid.cells[current], cell, vessel, weights, previous[current])
            tentative = g[current] + cost.total
            if tentative + 1e-12 < g.get(nxt, inf):
                g[nxt] = tentative
                parent[nxt] = current
                previous[nxt] = current
                step_info[nxt] = cost
                counter += 1
                heappush(open_heap, (tentative + _heuristic(nxt, goal, grid, vessel, weights), counter, nxt))
    raise ValueError("No navigable route exists within bounding box corridor")


def load_grid_from_csv(
    grid_filepath: str,
    vectors_filepath: str = "ocean_vectors_antarctic.csv",
) -> Grid:
    """Loads both the high-res risk map and Soumya's ocean/wind vector dataset."""
    df_grid = pd.read_csv(grid_filepath, header=None)
    mat = df_grid.to_numpy()
    rows, cols = mat.shape

    vectors_df = None
    if os.path.exists(vectors_filepath):
        try:
            vectors_df = pd.read_csv(vectors_filepath)
            if {"grid_row", "grid_col"}.issubset(vectors_df.columns):
                vectors_df = vectors_df.set_index(["grid_row", "grid_col"])
        except Exception as e:
            print(f"⚠️ Could not parse vector file: {e}")

    cells: dict[tuple[int, int], Cell] = {}
    
    for r in range(rows):
        row_vals = mat[r]
        for c in range(cols):
            val = float(row_vals[c])
            blocked = val >= 7.0  
            ice_concentration = _clamp(val / 9.0)
            ice_thickness_m = ice_concentration * 0.7  
            
            wind_speed_ms = 5.0 + val * 1.2
            
            if vectors_df is not None and (r, c) in vectors_df.index:
                cell_vec = vectors_df.loc[(r, c)]
                if "wind_speed_knots" in cell_vec:
                    wind_speed_ms = float(cell_vec["wind_speed_knots"]) * 0.514444  
            
            cells[(r, c)] = Cell(
                row=r,
                col=c,
                ice_concentration=ice_concentration,
                ice_thickness_m=ice_thickness_m,
                wind_speed_ms=wind_speed_ms,
                visibility_km=15.0,
                data_age_h=2.0,
                blocked=blocked
            )
            
    return Grid(cells, rows, cols, cell_size_km=1.0)


if __name__ == "__main__":
    # 1. First priority: Check if a filepath was passed from the command line (e.g. by watcher.py)
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    # 2. Second priority: Fallback to reading whatever filename is recorded in latest.txt
    elif os.path.exists("latest.txt"):
        with open("latest.txt", "r") as f:
            filepath = f.read().strip()
    # 3. Third priority: Fallback default if you run it completely standalone for manual testing
    else:
        filepath = "risk_map_1000x1000.csv"

    print(f"Loading CSV from: {filepath}")

    grid = load_grid_from_csv(filepath, "ocean_vectors_antarctic.csv")
    grid.validate()
    print(f"✅ Successfully validated grid: {grid.rows}x{grid.cols} cells.")

    start_node = (50, 50)
    goal_node = (900, 900)

    for profile in ("fastest", "safest", "balanced"):
        try:
            route = astar(grid, start_node, goal_node, profile=profile, buffer_cells=300)
            print(f"[{profile.upper()}] Route found! Cost: {route.total_cost:.2f} | Distance: {route.distance_nm:.2f} NM | Time: {route.travel_time_h:.2f} hrs | Nodes Expanded: {route.expanded_nodes}")
        except ValueError as exc:
            print(profile, "NO ROUTE:", exc)