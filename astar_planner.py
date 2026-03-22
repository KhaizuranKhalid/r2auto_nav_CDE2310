import heapq
import math
from typing import List, Tuple, Optional


def map_to_grid(x: float, y: float, origin: Tuple[float, float], resolution: float) -> Tuple[int, int]:
    ox, oy = origin
    gx = int((x - ox) / resolution)
    gy = int((y - oy) / resolution)
    return gx, gy


def grid_to_map(gx: int, gy: int, origin: Tuple[float, float], resolution: float) -> Tuple[float, float]:
    ox, oy = origin
    x = (gx + 0.5) * resolution + ox
    y = (gy + 0.5) * resolution + oy
    return x, y


def neighbors(gx: int, gy: int, width: int, height: int, diagonal: bool = True):
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in steps:
        nx, ny = gx + dx, gy + dy
        if 0 <= nx < width and 0 <= ny < height:
            yield nx, ny


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    (x1, y1), (x2, y2) = a, b
    return math.hypot(x2 - x1, y2 - y1)


def astar_plan(
    grid: List[int],
    width: int,
    height: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    allow_diagonal: bool = True,
    occupancy_threshold: int = 1,
) -> Optional[List[Tuple[int, int]]]:
    """
    grid: flat list row-major of occupancy (0 free, 1 occupied, -1 unknown)
    start/goal: grid indices (gx, gy)
    returns list of grid indices from start to goal (inclusive) or None
    """
    def is_free(cell):
        x, y = cell
        v = grid[y * width + x]
        # treat unknown (-1) as obstacle for planning
        return v == 0

    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < width and 0 <= sy < height):
        return None
    if not (0 <= gx < width and 0 <= gy < height):
        return None
    if not is_free(goal):
        return None

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    gscore = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path

        for n in neighbors(current[0], current[1], width, height, diagonal=allow_diagonal):
            if not is_free(n):
                continue
            tentative_g = gscore[current] + heuristic(current, n)
            if n not in gscore or tentative_g < gscore[n]:
                gscore[n] = tentative_g
                priority = tentative_g + heuristic(n, goal)
                heapq.heappush(open_set, (priority, tentative_g, n))
                came_from[n] = current

    return None

def inflate_obstacles(grid, width, height, radius):
    """
    Inflate occupied cells by radius (in grid cells)
    grid: flat list
    """
    inflated = grid.copy()

    for y in range(height):
        for x in range(width):
            if grid[y * width + x] == 1:  # obstacle
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx = x + dx
                        ny = y + dy

                        if 0 <= nx < width and 0 <= ny < height:
                            inflated[ny * width + nx] = 1

    return inflated


def plan_path_from_map(
    occupancy: List[int],
    map_width: int,
    map_height: int,
    resolution: float,
    origin: Tuple[float, float],
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
) -> Optional[List[Tuple[float, float]]]:

    # Convert occupancy to obstacle map
    grid = []
    for v in occupancy:
        if v == 0:
            grid.append(0)   # free
        else:
            grid.append(1)   # obstacle or unknown

    # ---- OBSTACLE INFLATION ----
    # Compute inflation radius in grid cells
    robot_radius = 0.15  # meters
    safety_margin = 0.10  # meters
    total_radius_m = robot_radius + safety_margin
    inflation_radius = int(total_radius_m / resolution)
    
    grid = inflate_obstacles(grid, map_width, map_height, inflation_radius)

    # Convert world to grid
    sx, sy = map_to_grid(start_xy[0], start_xy[1], origin, resolution)
    gx, gy = map_to_grid(goal_xy[0], goal_xy[1], origin, resolution)

    # Run A*
    path_cells = astar_plan(grid, map_width, map_height, (sx, sy), (gx, gy))

    if path_cells is None:
        return None

    # Convert grid path back to world coordinates
    path_world = [grid_to_map(x, y, origin, resolution) for (x, y) in path_cells]

    return path_world
