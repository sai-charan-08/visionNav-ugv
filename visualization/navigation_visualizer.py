"""
visualization/navigation_visualizer.py

Visualizes:
    Camera image
        ↓
    Detected obstacles
        ↓
    A* planned path

This module does not change perception, costmap, or A* logic.
"""

import cv2


class NavigationVisualizer:
    """Draw obstacle detections and planned navigation path."""

    def __init__(self):
        pass

    def draw_obstacles(self, frame, obstacles):
        """Draw detected obstacle bounding boxes."""

        output = frame.copy()

        for obstacle in obstacles:

            x1, y1, x2, y2 = obstacle.bbox

            # Draw bounding box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            # Label
            label = (
                f"{obstacle.class_name} "
                f"{obstacle.confidence:.2f}"
            )

            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return output

    def draw_path(
        self,
        frame,
        path,
        costmap_rows,
        costmap_cols
    ):
        """
        Draw A* path on the camera image.

        Path coordinates are grid coordinates:
            (row, column)

        They are converted back to image coordinates.
        """

        output = frame.copy()

        if not path:
            return output

        height, width = output.shape[:2]

        points = []

        for row, col in path:

            x = int(
                (col + 0.5)
                * width
                / costmap_cols
            )

            y = int(
                (row + 0.5)
                * height
                / costmap_rows
            )

            points.append((x, y))

        # Draw path
        for i in range(len(points) - 1):

            cv2.line(
                output,
                points[i],
                points[i + 1],
                (0, 255, 0),
                4
            )

        # Start
        cv2.circle(
            output,
            points[0],
            10,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            output,
            "START",
            (points[0][0] + 12, points[0][1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # Goal
        cv2.circle(
            output,
            points[-1],
            10,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            output,
            "GOAL",
            (points[-1][0] + 12, points[-1][1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        return output

    def visualize(
        self,
        frame,
        obstacles,
        path,
        costmap_rows,
        costmap_cols
    ):
        """Draw obstacles and A* path together."""

        output = self.draw_obstacles(
            frame,
            obstacles
        )

        output = self.draw_path(
            output,
            path,
            costmap_rows,
            costmap_cols
        )

        return output