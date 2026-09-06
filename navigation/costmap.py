"""
navigation/costmap.py

VisionNav UGV - Traversability-Aware Costmap V3

Combines:

    1. RGB traversability estimation
    2. YOLO obstacle detections
    3. Safety inflation

Cost values:

    0   = SAFE / highly traversable
    30  = UNCERTAIN
    60  = DIFFICULT / low confidence
    100 = BLOCKED / UNKNOWN

Important:

Unknown regions are treated conservatively as blocked.
This prevents A* from driving through areas where the camera
has insufficient evidence of traversability.
"""

import cv2
import numpy as np


class Costmap:

    def __init__(
        self,
        rows=12,
        cols=16,
        obstacle_inflation=1
    ):

        self.rows = rows
        self.cols = cols
        self.obstacle_inflation = obstacle_inflation

        # ----------------------------------------------------------
        # Cost grid
        #
        # 0   = safe
        # 30  = uncertain
        # 60  = difficult
        # 100 = blocked / unknown
        # ----------------------------------------------------------

        self.grid = np.full(
            (rows, cols),
            100,
            dtype=np.uint8
        )

    # ==============================================================
    # RESET
    # ==============================================================

    def reset(self):

        self.grid.fill(100)

    # ==============================================================
    # IMAGE -> GRID
    # ==============================================================

    def image_to_grid(
        self,
        x,
        y,
        image_width,
        image_height
    ):

        col = int(
            x * self.cols / image_width
        )

        row = int(
            y * self.rows / image_height
        )

        row = max(
            0,
            min(self.rows - 1, row)
        )

        col = max(
            0,
            min(self.cols - 1, col)
        )

        return row, col

    # ==============================================================
    # TRAVERSABILITY -> COST
    # ==============================================================

    def build_from_traversability(
        self,
        traversability_mask
    ):
        """
        Convert the binary traversability mask into a costmap.

        Mask:

            1 = estimated traversable
            0 = unknown / non-traversable

        Grid interpretation:

            >= 70% traversable
                -> SAFE (0)

            35% - 70% traversable
                -> UNCERTAIN (30)

            10% - 35% traversable
                -> DIFFICULT (60)

            < 10% traversable
                -> BLOCKED / UNKNOWN (100)

        This conservative interpretation prevents the planner
        from treating walls, sky, vegetation and unknown areas
        as normal drivable terrain.
        """

        if traversability_mask is None:

            raise ValueError(
                "Traversability mask cannot be None."
            )

        height, width = (
            traversability_mask.shape[:2]
        )

        for row in range(self.rows):

            for col in range(self.cols):

                # --------------------------------------------------
                # Pixel region represented by this grid cell
                # --------------------------------------------------

                x1 = int(
                    col
                    * width
                    / self.cols
                )

                x2 = int(
                    (col + 1)
                    * width
                    / self.cols
                )

                y1 = int(
                    row
                    * height
                    / self.rows
                )

                y2 = int(
                    (row + 1)
                    * height
                    / self.rows
                )

                cell = traversability_mask[
                    y1:y2,
                    x1:x2
                ]

                if cell.size == 0:

                    self.grid[row, col] = 100

                    continue

                # --------------------------------------------------
                # Percentage of pixels classified as traversable
                # --------------------------------------------------

                traversable_ratio = np.mean(
                    cell > 0
                )

                # --------------------------------------------------
                # Convert evidence into navigation cost
                # --------------------------------------------------

                if traversable_ratio >= 0.70:

                    # Strong evidence of ground
                    self.grid[row, col] = 0

                elif traversable_ratio >= 0.35:

                    # Some ground evidence
                    self.grid[row, col] = 30

                elif traversable_ratio >= 0.10:

                    # Weak evidence
                    self.grid[row, col] = 60

                else:

                    # Almost no evidence.
                    #
                    # Treat as blocked rather than allowing
                    # A* to blindly drive through it.
                    self.grid[row, col] = 100

    # ==============================================================
    # OBSTACLE INFLATION
    # ==============================================================

    def mark_obstacle(
        self,
        x1,
        y1,
        x2,
        y2,
        image_width,
        image_height
    ):
        """
        Mark YOLO detected obstacle as completely blocked.

        Inflates the obstacle to provide a safety margin around
        the detected object.
        """

        row1, col1 = self.image_to_grid(
            x1,
            y1,
            image_width,
            image_height
        )

        row2, col2 = self.image_to_grid(
            x2,
            y2,
            image_width,
            image_height
        )

        r_min = min(row1, row2)
        r_max = max(row1, row2)

        c_min = min(col1, col2)
        c_max = max(col1, col2)

        for row in range(
            r_min - self.obstacle_inflation,
            r_max + self.obstacle_inflation + 1
        ):

            for col in range(
                c_min - self.obstacle_inflation,
                c_max + self.obstacle_inflation + 1
            ):

                if (
                    0 <= row < self.rows
                    and
                    0 <= col < self.cols
                ):

                    self.grid[row, col] = 100

    # ==============================================================
    # BUILD FROM YOLO BOXES
    # ==============================================================

    def build_from_boxes(
        self,
        boxes,
        image_width,
        image_height
    ):
        """
        Add YOLO obstacles to the costmap.
        """

        for box in boxes:

            x1, y1, x2, y2 = box

            self.mark_obstacle(
                x1,
                y1,
                x2,
                y2,
                image_width,
                image_height
            )

    # ==============================================================
    # COMPLETE COSTMAP
    # ==============================================================

    def build(
        self,
        traversability_mask,
        boxes,
        image_width,
        image_height
    ):
        """
        Build complete navigation costmap.

        Pipeline:

            Traversability
                  ↓
            Base costmap
                  ↓
            YOLO obstacles
                  ↓
            Obstacle inflation
                  ↓
            Final costmap
        """

        self.reset()

        # ----------------------------------------------------------
        # 1. Traversability
        # ----------------------------------------------------------

        self.build_from_traversability(
            traversability_mask
        )

        # ----------------------------------------------------------
        # 2. YOLO obstacles
        # ----------------------------------------------------------

        self.build_from_boxes(
            boxes,
            image_width,
            image_height
        )

    # ==============================================================
    # CELL CHECKS
    # ==============================================================

    def is_free(
        self,
        row,
        col
    ):
        """
        Return True only for cells that are not completely blocked.

        Cost:

            0  -> free
            30 -> free but uncertain
            60 -> free but difficult
            100 -> blocked
        """

        if not (
            0 <= row < self.rows
            and
            0 <= col < self.cols
        ):

            return False

        return self.grid[row, col] < 100

    # ==============================================================
    # PRINT COSTMAP
    # ==============================================================

    def print_grid(self):

        print("\nCostmap:")

        print(
            "0=SAFE  "
            "30=UNCERTAIN  "
            "60=DIFFICULT  "
            "100=BLOCKED/UNKNOWN\n"
        )

        for row in self.grid:

            print(
                " ".join(
                    f"{int(value):3d}"
                    for value in row
                )
            )


# ======================================================================
# STANDALONE TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("VisionNav UGV - Costmap V3 Test")
    print("=" * 70)

    # --------------------------------------------------------------
    # Artificial test image
    # --------------------------------------------------------------

    height = 480
    width = 640

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # Simulated road corridor
    cv2.rectangle(
        mask,
        (180, 160),
        (460, 480),
        1,
        -1
    )

    # --------------------------------------------------------------
    # Create costmap
    # --------------------------------------------------------------

    costmap = Costmap(
        rows=12,
        cols=16,
        obstacle_inflation=1
    )

    # --------------------------------------------------------------
    # Example obstacle
    # --------------------------------------------------------------

    boxes = [
        (280, 260, 360, 360)
    ]

    # --------------------------------------------------------------
    # Build
    # --------------------------------------------------------------

    costmap.build(
        mask,
        boxes,
        width,
        height
    )

    # --------------------------------------------------------------
    # Display
    # --------------------------------------------------------------

    costmap.print_grid()

    print("\nCostmap V3 test completed.")
    print("=" * 70)