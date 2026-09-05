"""
navigation/path_planner.py

A* global path planner for VisionNav UGV.

Grid convention:
    0 = free / traversable
    1 = blocked / obstacle

Coordinates:
    (row, column)
"""

import heapq
import math


class PathPlanner:
    """A* path planner operating on a 2D occupancy grid."""

    def __init__(self, grid):
        self.grid = grid

        if not grid or not grid[0]:
            raise ValueError("Grid must not be empty.")

        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, node):
        """Check whether a grid coordinate is inside the grid."""
        r, c = node
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_free(self, node):
        """Check whether a grid coordinate is traversable."""
        r, c = node
        return self.grid[r][c] == 0

    @staticmethod
    def _heuristic(a, b):
        """Euclidean distance heuristic."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _neighbors(self, node):
        """Return valid 8-connected neighboring cells."""
        r, c = node

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        for dr, dc in directions:
            neighbor = (r + dr, c + dc)

            if self._is_valid(neighbor) and self._is_free(neighbor):
                yield neighbor

    @staticmethod
    def _movement_cost(a, b):
        """Cost of moving between two neighboring cells."""
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])

        if dr == 1 and dc == 1:
            return math.sqrt(2)

        return 1.0

    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from goal back to start."""
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def plan(self, start, goal):
        """
        Find the shortest path from start to goal using A*.

        Args:
            start: (row, column)
            goal:  (row, column)

        Returns:
            List of coordinates if a path exists, otherwise None.
        """

        if not self._is_valid(start):
            raise ValueError(f"Start position {start} is outside the grid.")

        if not self._is_valid(goal):
            raise ValueError(f"Goal position {goal} is outside the grid.")

        if not self._is_free(start):
            raise ValueError(f"Start position {start} is blocked.")

        if not self._is_free(goal):
            raise ValueError(f"Goal position {goal} is blocked.")

        if start == goal:
            return [start]

        open_set = []

        # (f_cost, counter, node)
        counter = 0

        start_g = 0.0
        start_f = start_g + self._heuristic(start, goal)

        heapq.heappush(open_set, (start_f, counter, start))

        came_from = {}
        g_cost = {start: 0.0}

        closed_set = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            if current == goal:
                return self._reconstruct_path(came_from, current)

            closed_set.add(current)

            for neighbor in self._neighbors(current):

                if neighbor in closed_set:
                    continue

                tentative_g = (
                    g_cost[current]
                    + self._movement_cost(current, neighbor)
                )

                if tentative_g < g_cost.get(neighbor, float("inf")):

                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_g

                    f_cost = (
                        tentative_g
                        + self._heuristic(neighbor, goal)
                    )

                    counter += 1

                    heapq.heappush(
                        open_set,
                        (f_cost, counter, neighbor)
                    )

        return None


if __name__ == "__main__":

    print("=" * 50)
    print("VisionNav UGV - A* Path Planner Test")
    print("=" * 50)

    # 0 = free
    # 1 = obstacle
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (5, 7)

    planner = PathPlanner(grid)

    path = planner.plan(start, goal)

    print(f"\nStart: {start}")
    print(f"Goal : {goal}")

    if path:
        print("\nPath found!")
        print(f"Path length: {len(path)}")
        print("Path:")

        for position in path:
            print(position)

    else:
        print("\nNo path found.")

    print("\nA* test completed.")