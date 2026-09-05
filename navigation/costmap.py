"""
navigation/costmap.py

Simple camera-derived traversability costmap for VisionNav UGV.

Grid convention:
    0 = free / traversable
    1 = obstacle

This is a prototype costmap. It converts obstacle regions
from image coordinates into a 2D navigation grid.
"""

import math


class Costmap:
    """Generate a 2D traversability grid from image-space obstacles."""

    def __init__(self, rows=12, cols=16, obstacle_inflation=1):
        self.rows = rows
        self.cols = cols
        self.obstacle_inflation = obstacle_inflation

        self.grid = [
            [0 for _ in range(cols)]
            for _ in range(rows)
        ]

    def reset(self):
        """Clear the costmap."""
        self.grid = [
            [0 for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

    def image_to_grid(self, x, y, image_width, image_height):
        """
        Convert image pixel coordinates to grid coordinates.

        Args:
            x, y: image coordinates
            image_width: image width in pixels
            image_height: image height in pixels

        Returns:
            (row, col)
        """

        col = int((x / image_width) * self.cols)
        row = int((y / image_height) * self.rows)

        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))

        return row, col

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
        Mark an image-space bounding box as an obstacle.

        The obstacle is expanded by obstacle_inflation cells
        to provide a safety margin.
        """

        r1, c1 = self.image_to_grid(
            x1, y1,
            image_width,
            image_height
        )

        r2, c2 = self.image_to_grid(
            x2, y2,
            image_width,
            image_height
        )

        # Make sure coordinates are ordered correctly.
        r_min = min(r1, r2)
        r_max = max(r1, r2)

        c_min = min(c1, c2)
        c_max = max(c1, c2)

        # Inflate obstacle for safety clearance.
        r_min = max(
            0,
            r_min - self.obstacle_inflation
        )

        r_max = min(
            self.rows - 1,
            r_max + self.obstacle_inflation
        )

        c_min = max(
            0,
            c_min - self.obstacle_inflation
        )

        c_max = min(
            self.cols - 1,
            c_max + self.obstacle_inflation
        )

        for row in range(r_min, r_max + 1):
            for col in range(c_min, c_max + 1):
                self.grid[row][col] = 1

    def build_from_boxes(
        self,
        boxes,
        image_width,
        image_height
    ):
        """
        Build a costmap from bounding boxes.

        Each box must contain:

            (x1, y1, x2, y2)

        Returns:
            2D grid
        """

        self.reset()

        for box in boxes:
            if len(box) != 4:
                continue

            x1, y1, x2, y2 = box

            self.mark_obstacle(
                x1,
                y1,
                x2,
                y2,
                image_width,
                image_height
            )

        return self.grid

    def is_free(self, row, col):
        """Return True if a grid cell is traversable."""

        if not (0 <= row < self.rows):
            return False

        if not (0 <= col < self.cols):
            return False

        return self.grid[row][col] == 0

    def print_grid(self):
        """Print the costmap in an easy-to-read format."""

        print("\nCostmap:")
        print("0 = free   1 = obstacle\n")

        for row in self.grid:
            print(" ".join(str(cell) for cell in row))


if __name__ == "__main__":

    print("=" * 55)
    print("VisionNav UGV - Traversability Costmap Test")
    print("=" * 55)

    # Simulated camera resolution.
    image_width = 640
    image_height = 480

    # Create a 12 x 16 navigation grid.
    costmap = Costmap(
        rows=12,
        cols=16,
        obstacle_inflation=1
    )

    # Simulated YOLO/obstacle bounding boxes.
    #
    # One obstacle in the center of the image.
    test_boxes = [
        (260, 220, 380, 400)
    ]

    grid = costmap.build_from_boxes(
        test_boxes,
        image_width,
        image_height
    )

    costmap.print_grid()

    print("\nGrid size:")
    print(f"Rows: {costmap.rows}")
    print(f"Cols: {costmap.cols}")

    print("\nTest obstacle:")
    print("(260, 220) -> (380, 400)")

    print("\nCostmap test completed.")