"""
navigation/path_planner.py

VisionNav UGV - Cost-Aware A* Path Planner

Costmap values:

    0   = SAFE
    20  = UNCERTAIN
    50  = DIFFICULT
    100 = BLOCKED

The planner:

    1. Never enters blocked cells.
    2. Prefers safe cells.
    3. Penalizes uncertain and difficult cells.
    4. Supports 8-directional movement.
    5. Returns a collision-free path.
"""

import heapq
import math


class PathPlanner:
    """
    Cost-aware A* path planner.

    Grid coordinates use:

        (row, column)

    Costmap values:

        0   -> safe
        20  -> uncertain
        50  -> difficult
        100 -> blocked
    """

    def __init__(self, grid):

        self.grid = grid

        self.rows = len(grid)
        self.cols = len(grid[0])

    # ==============================================================
    # CELL CHECK
    # ==============================================================

    def is_walkable(self, row, col):
        """
        Return True if the cell is inside the grid and not blocked.
        """

        if not (
            0 <= row < self.rows
            and
            0 <= col < self.cols
        ):
            return False

        return self.grid[row][col] < 100

    # ==============================================================
    # CELL TRAVERSAL COST
    # ==============================================================

    def cell_cost(self, row, col):
        """
        Convert costmap value into actual A* traversal cost.

        SAFE:
            0 -> 1.0

        UNCERTAIN:
            20 -> 1.4

        DIFFICULT:
            50 -> 2.0

        BLOCKED:
            100 -> infinity
        """

        value = float(
            self.grid[row][col]
        )

        if value >= 100:
            return float("inf")

        if value == 0:
            return 1.0

        if value <= 20:
            return 1.4

        if value <= 50:
            return 2.0

        return 3.0

    # ==============================================================
    # HEURISTIC
    # ==============================================================

    def heuristic(self, current, goal):
        """
        Euclidean distance heuristic.

        Since the minimum traversal cost is 1.0,
        this remains a reasonable lower-bound estimate.
        """

        r1, c1 = current
        r2, c2 = goal

        return math.sqrt(
            (r1 - r2) ** 2
            +
            (c1 - c2) ** 2
        )

    # ==============================================================
    # NEIGHBOURS
    # ==============================================================

    def get_neighbors(self, row, col):
        """
        Return all valid 8-connected neighbouring cells.
        """

        directions = [
            (-1, 0),   # up
            (1, 0),    # down
            (0, -1),   # left
            (0, 1),    # right

            (-1, -1),  # up-left
            (-1, 1),   # up-right
            (1, -1),   # down-left
            (1, 1)     # down-right
        ]

        neighbors = []

        for dr, dc in directions:

            nr = row + dr
            nc = col + dc

            if self.is_walkable(nr, nc):
                neighbors.append(
                    (nr, nc)
                )

        return neighbors

    # ==============================================================
    # A* PLANNER
    # ==============================================================

    def plan(self, start, goal):
        """
        Find a cost-aware path from start to goal.

        Parameters
        ----------
        start : tuple
            (row, column)

        goal : tuple
            (row, column)

        Returns
        -------
        list
            Ordered list of grid cells.

        None
            If no path exists.
        """

        # ----------------------------------------------------------
        # Validate start and goal
        # ----------------------------------------------------------

        if not self.is_walkable(
            start[0],
            start[1]
        ):
            print(
                "[A*] Start cell is blocked."
            )
            return None

        if not self.is_walkable(
            goal[0],
            goal[1]
        ):
            print(
                "[A*] Goal cell is blocked."
            )
            return None

        # ----------------------------------------------------------
        # Priority queue
        # ----------------------------------------------------------

        open_set = []

        counter = 0

        start_f = self.heuristic(
            start,
            goal
        )

        heapq.heappush(
            open_set,
            (
                start_f,
                counter,
                start
            )
        )

        # ----------------------------------------------------------
        # Cost from start
        # ----------------------------------------------------------

        g_score = {
            start: 0.0
        }

        # ----------------------------------------------------------
        # Parent map
        # ----------------------------------------------------------

        came_from = {}

        # ----------------------------------------------------------
        # Closed set
        # ----------------------------------------------------------

        closed_set = set()

        # ----------------------------------------------------------
        # Main A* loop
        # ----------------------------------------------------------

        while open_set:

            _, _, current = heapq.heappop(
                open_set
            )

            # Skip already processed nodes
            if current in closed_set:
                continue

            # Goal reached
            if current == goal:

                return self.reconstruct_path(
                    came_from,
                    current
                )

            closed_set.add(current)

            current_row, current_col = current

            # ------------------------------------------------------
            # Explore neighbours
            # ------------------------------------------------------

            for neighbor in self.get_neighbors(
                current_row,
                current_col
            ):

                if neighbor in closed_set:
                    continue

                nr, nc = neighbor

                # --------------------------------------------------
                # Movement distance
                # --------------------------------------------------

                dr = abs(
                    nr - current_row
                )

                dc = abs(
                    nc - current_col
                )

                if dr == 1 and dc == 1:
                    movement_distance = math.sqrt(2)

                else:
                    movement_distance = 1.0

                # --------------------------------------------------
                # Traversability penalty
                # --------------------------------------------------

                traversal_cost = self.cell_cost(
                    nr,
                    nc
                )

                if math.isinf(
                    traversal_cost
                ):
                    continue

                # --------------------------------------------------
                # New cost
                # --------------------------------------------------

                tentative_g = (
                    g_score[current]
                    +
                    movement_distance
                    *
                    traversal_cost
                )

                old_g = g_score.get(
                    neighbor,
                    float("inf")
                )

                # --------------------------------------------------
                # Better route found
                # --------------------------------------------------

                if tentative_g < old_g:

                    came_from[
                        neighbor
                    ] = current

                    g_score[
                        neighbor
                    ] = tentative_g

                    f_score = (
                        tentative_g
                        +
                        self.heuristic(
                            neighbor,
                            goal
                        )
                    )

                    counter += 1

                    heapq.heappush(
                        open_set,
                        (
                            f_score,
                            counter,
                            neighbor
                        )
                    )

        # ----------------------------------------------------------
        # No path
        # ----------------------------------------------------------

        return None

    # ==============================================================
    # PATH RECONSTRUCTION
    # ==============================================================

    def reconstruct_path(
        self,
        came_from,
        current
    ):
        """
        Reconstruct path from goal back to start.
        """

        path = [
            current
        ]

        while current in came_from:

            current = came_from[
                current
            ]

            path.append(
                current
            )

        path.reverse()

        return path


# ======================================================================
# STANDALONE TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("VisionNav UGV - Cost-Aware A* Test")
    print("=" * 70)

    # --------------------------------------------------------------
    # Test costmap
    #
    # 0   = SAFE
    # 20  = UNCERTAIN
    # 50  = DIFFICULT
    # 100 = BLOCKED
    # --------------------------------------------------------------

    grid = [

        [0,   0,   0,   0,   0,   0,   0,   0],

        [0,   0,   0,   0,   0,   0,   0,   0],

        [0,   0, 100, 100, 100, 100,  0,   0],

        [0,   0, 100,  50,  50, 100,  0,   0],

        [0,   0, 100,  50,  50, 100,  0,   0],

        [0,   0, 100, 100, 100, 100,  0,   0],

        [0,   0,   0,   0,   0,   0,   0,   0],

        [0,   0,   0,   0,   0,   0,   0,   0]
    ]

    # --------------------------------------------------------------
    # Create planner
    # --------------------------------------------------------------

    planner = PathPlanner(
        grid
    )

    start = (
        7,
        3
    )

    goal = (
        0,
        4
    )

    print(
        f"\nStart: {start}"
    )

    print(
        f"Goal : {goal}"
    )

    # --------------------------------------------------------------
    # Run A*
    # --------------------------------------------------------------

    path = planner.plan(
        start,
        goal
    )

    # --------------------------------------------------------------
    # Display result
    # --------------------------------------------------------------

    if path:

        print(
            "\nSUCCESS - Cost-aware path found!"
        )

        print(
            "\nPath:"
        )

        for position in path:

            print(
                position
            )

        print(
            f"\nPath length: "
            f"{len(path)} cells"
        )

    else:

        print(
            "\nFAILED - No path found."
        )

    print(
        "\n" + "=" * 70
    )