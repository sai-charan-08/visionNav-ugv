"""
VisionNav UGV - Full Navigation Integration Test

Complete software navigation pipeline:

    Camera Image
         ↓
    YOLO Detection
         ↓
    Obstacle Detection
         ↓
    Traversability Detection
         ↓
    Costmap V3
         ↓
    Cost-Aware A*
         ↓
    Navigation Direction
         ↓
    Visualization
"""

import os
import sys

import cv2
import numpy as np


# ======================================================================
# PROJECT ROOT
# ======================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ======================================================================
# PROJECT MODULES
# ======================================================================

import config

from perception.yolo_detector import YOLODetector
from perception.obstacle_detector import ObstacleDetector
from perception.traversability import TraversabilityDetector

from navigation.costmap import Costmap
from navigation.path_planner import PathPlanner

from visualization.navigation_visualizer import NavigationVisualizer


# ======================================================================
# FIND BEST CELL
# ======================================================================

def find_best_cell(
    costmap,
    preferred_row,
    preferred_col,
    max_cost=30
):
    """
    Find the safest available cell near a preferred location.

    First searches for cells with cost <= max_cost.

    If no such cell exists, falls back to any non-blocked cell.
    """

    candidates = []

    # --------------------------------------------------------------
    # Search nearby rows
    # --------------------------------------------------------------

    for row_offset in range(-2, 3):

        row = preferred_row + row_offset

        if not (
            0 <= row < costmap.rows
        ):
            continue

        for col in range(costmap.cols):

            cost = int(
                costmap.grid[row, col]
            )

            if cost <= max_cost:

                distance = (
                    abs(row - preferred_row)
                    +
                    abs(col - preferred_col)
                )

                candidates.append(
                    (
                        cost,
                        distance,
                        row,
                        col
                    )
                )

    # --------------------------------------------------------------
    # Prefer lowest cost, then closest to preferred position
    # --------------------------------------------------------------

    if candidates:

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1]
            )
        )

        return (
            candidates[0][2],
            candidates[0][3]
        )

    # --------------------------------------------------------------
    # Fallback to any non-blocked cell
    # --------------------------------------------------------------

    fallback = []

    for row in range(costmap.rows):

        for col in range(costmap.cols):

            cost = int(
                costmap.grid[row, col]
            )

            if cost < 100:

                distance = (
                    abs(row - preferred_row)
                    +
                    abs(col - preferred_col)
                )

                fallback.append(
                    (
                        cost,
                        distance,
                        row,
                        col
                    )
                )

    if fallback:

        fallback.sort(
            key=lambda item: (
                item[0],
                item[1]
            )
        )

        return (
            fallback[0][2],
            fallback[0][3]
        )

    return None


# ======================================================================
# DETERMINE INITIAL NAVIGATION DIRECTION
# ======================================================================

def get_initial_direction(
    path,
    start_col
):
    """
    Determine the first steering direction from the A* path.

    Returns:

        LEFT
        RIGHT
        FORWARD
        STOP
    """

    if not path or len(path) < 2:

        return "STOP"

    next_col = path[1][1]

    if next_col < start_col:

        return "LEFT"

    if next_col > start_col:

        return "RIGHT"

    return "FORWARD"


# ======================================================================
# CREATE TRAVERSABILITY OVERLAY
# ======================================================================

def create_traversability_overlay(
    frame,
    traversability_mask
):
    """
    Create a simple visualization of the traversability mask.

    WHITE/BRIGHT GREEN:
        Estimated traversable ground.

    Original image:
        Preserved everywhere else.

    This function is intentionally implemented here rather than
    depending on a create_overlay() method inside TraversabilityDetector.
    """

    output = frame.copy()

    if traversability_mask is None:

        return output

    # Make sure mask is binary uint8
    mask = (
        traversability_mask > 0
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------------
    # Create green overlay
    # --------------------------------------------------------------

    overlay = np.zeros_like(
        frame
    )

    overlay[:, :, 1] = 255

    # --------------------------------------------------------------
    # Blend only where traversable
    # --------------------------------------------------------------

    alpha = 0.35

    traversable_pixels = (
        mask.astype(bool)
    )

    output[
        traversable_pixels
    ] = cv2.addWeighted(
        frame[
            traversable_pixels
        ],
        1.0 - alpha,
        overlay[
            traversable_pixels
        ],
        alpha,
        0
    )

    # --------------------------------------------------------------
    # Add label
    # --------------------------------------------------------------

    cv2.putText(
        output,
        "TRAVERSABLE GROUND",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    return output


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)

    print(
        "VisionNav UGV - Full Navigation Integration Test"
    )

    print("=" * 70)

    # ==================================================================
    # 1. LOAD CAMERA IMAGE
    # ==================================================================

    frame = cv2.imread(
        config.IMAGE_PATH
    )

    if frame is None:

        print(
            "\nERROR: Could not load image:"
        )

        print(
            config.IMAGE_PATH
        )

        return

    height, width = frame.shape[:2]

    print(
        f"\n[1] Camera image: "
        f"{width} x {height}"
    )

    # ==================================================================
    # 2. YOLO DETECTION
    # ==================================================================

    print(
        "\n[2] Running YOLO..."
    )

    detector = YOLODetector()

    yolo_result, inference_ms = (
        detector.detect(frame)
    )

    print(
        f"[2] YOLO inference: "
        f"{inference_ms:.1f} ms"
    )

    print(
        f"[2] Raw detections: "
        f"{len(yolo_result.boxes)}"
    )

    # ==================================================================
    # 3. OBSTACLE DETECTION
    # ==================================================================

    print(
        "\n[3] Running obstacle analysis..."
    )

    obstacle_detector = (
        ObstacleDetector()
    )

    obstacles = (
        obstacle_detector.analyze(
            yolo_result,
            width,
            height
        )
    )

    print(
        f"[3] Filtered obstacles: "
        f"{len(obstacles)}"
    )

    for index, obstacle in enumerate(
        obstacles,
        start=1
    ):

        print(
            f"    Obstacle {index}: "
            f"{obstacle.class_name}, "
            f"confidence={obstacle.confidence:.3f}, "
            f"zones={obstacle.zones}, "
            f"bbox={obstacle.bbox}"
        )

    # ==================================================================
    # 4. TRAVERSABILITY DETECTION
    # ==================================================================

    print(
        "\n[4] Estimating traversable ground..."
    )

    traversability_detector = (
        TraversabilityDetector()
    )

    traversability_mask = (
        traversability_detector.detect(
            frame
        )
    )

    if traversability_mask is None:

        print(
            "\nERROR: Traversability detector "
            "returned no mask."
        )

        return

    # ==================================================================
    # SAVE TRAVERSABILITY RESULTS
    # ==================================================================

    runs_dir = os.path.join(
        PROJECT_ROOT,
        "runs"
    )

    os.makedirs(
        runs_dir,
        exist_ok=True
    )

    # --------------------------------------------------------------
    # Binary mask
    # --------------------------------------------------------------

    mask_path = os.path.join(
        runs_dir,
        "navigation_traversability_mask.jpg"
    )

    binary_mask = (
        traversability_mask > 0
    ).astype(
        np.uint8
    ) * 255

    cv2.imwrite(
        mask_path,
        binary_mask
    )

    # --------------------------------------------------------------
    # Overlay
    # --------------------------------------------------------------

    overlay_path = os.path.join(
        runs_dir,
        "navigation_traversability_overlay.jpg"
    )

    traversability_overlay = (
        create_traversability_overlay(
            frame,
            traversability_mask
        )
    )

    cv2.imwrite(
        overlay_path,
        traversability_overlay
    )

    print(
        "[4] Traversability mask saved:"
    )

    print(
        f"    {mask_path}"
    )

    print(
        "[4] Traversability overlay saved:"
    )

    print(
        f"    {overlay_path}"
    )

    # ==================================================================
    # 5. BUILD COSTMAP V3
    # ==================================================================

    print(
        "\n[5] Building Costmap V3..."
    )

    costmap = Costmap(
        rows=12,
        cols=16,
        obstacle_inflation=1
    )

    # --------------------------------------------------------------
    # Extract YOLO bounding boxes
    # --------------------------------------------------------------

    boxes = [
        obstacle.bbox
        for obstacle in obstacles
    ]

    # --------------------------------------------------------------
    # Build complete costmap
    # --------------------------------------------------------------

    costmap.build(
        traversability_mask,
        boxes,
        width,
        height
    )

    costmap.print_grid()

    # ==================================================================
    # 6. SELECT START POSITION
    # ==================================================================

    preferred_start = (
        costmap.rows - 1,
        costmap.cols // 2
    )

    start = find_best_cell(
        costmap,
        preferred_start[0],
        preferred_start[1],
        max_cost=30
    )

    # ==================================================================
    # 7. SELECT FORWARD LOCAL GOAL
    # ==================================================================

    # The current traversability detector intentionally focuses
    # on the lower/middle ground region.
    #
    # Therefore, using row 0 as a goal would be incorrect.
    #
    # Around 35% of the costmap height gives us a forward
    # look-ahead region.

    preferred_goal = (
        int(costmap.rows * 0.35),
        costmap.cols // 2
    )

    goal = find_best_cell(
        costmap,
        preferred_goal[0],
        preferred_goal[1],
        max_cost=30
    )

    print(
        f"\n[6] Preferred start: "
        f"{preferred_start}"
    )

    print(
        f"[6] Selected start:   "
        f"{start}"
    )

    print(
        f"[6] Preferred goal:   "
        f"{preferred_goal}"
    )

    print(
        f"[6] Selected goal:     "
        f"{goal}"
    )

    if start is None:

        print(
            "\nERROR: No valid starting cell."
        )

        return

    if goal is None:

        print(
            "\nERROR: No valid forward goal cell."
        )

        return

    # ==================================================================
    # 8. COST-AWARE A* PLANNING
    # ==================================================================

    print(
        "\n[7] Running cost-aware A*..."
    )

    planner = PathPlanner(
        costmap.grid
    )

    path = planner.plan(
        start,
        goal
    )

    if not path:

        print(
            "\nA* FAILED - No path found."
        )

        print(
            "\nThe current perception system "
            "does not identify a traversable route "
            "between the selected start and goal."
        )

        return

    print(
        "\nSUCCESS - Navigation path found!"
    )

    # ==================================================================
    # PRINT PATH
    # ==================================================================

    print(
        "\nPlanned path:"
    )

    for cell in path:

        row, col = cell

        cell_cost = int(
            costmap.grid[
                row,
                col
            ]
        )

        print(
            f"  {cell} "
            f"cost={cell_cost}"
        )

    print(
        f"\nPath length: "
        f"{len(path)} cells"
    )

    # ==================================================================
    # 9. PATH RISK ANALYSIS
    # ==================================================================

    path_costs = [
        int(
            costmap.grid[
                row,
                col
            ]
        )
        for row, col in path
    ]

    maximum_path_cost = max(
        path_costs
    )

    # --------------------------------------------------------------
    # Count uncertain/difficult cells
    # --------------------------------------------------------------

    uncertain_cells = sum(
        1
        for cost in path_costs
        if cost == 30
    )

    difficult_cells = sum(
        1
        for cost in path_costs
        if cost == 60
    )

    # --------------------------------------------------------------
    # Determine path quality
    # --------------------------------------------------------------

    if maximum_path_cost == 0:

        path_quality = "SAFE"

    elif maximum_path_cost <= 30:

        path_quality = "LOW RISK / UNCERTAIN"

    elif maximum_path_cost < 100:

        path_quality = "DIFFICULT"

    else:

        path_quality = "BLOCKED"

    print(
        f"\nPath quality: "
        f"{path_quality}"
    )

    print(
        f"Maximum path cell cost: "
        f"{maximum_path_cost}"
    )

    print(
        f"Uncertain cells: "
        f"{uncertain_cells}"
    )

    print(
        f"Difficult cells: "
        f"{difficult_cells}"
    )

    # ==================================================================
    # 10. INITIAL NAVIGATION DIRECTION
    # ==================================================================

    direction = (
        get_initial_direction(
            path,
            start[1]
        )
    )

    print(
        f"\nInitial navigation decision: "
        f"{direction}"
    )

    # ==================================================================
    # 11. NAVIGATION VISUALIZATION
    # ==================================================================

    print(
        "\n[8] Creating navigation visualization..."
    )

    visualizer = (
        NavigationVisualizer()
    )

    result = visualizer.visualize(
        frame,
        obstacles,
        path,
        costmap.rows,
        costmap.cols
    )

    # ==================================================================
    # ADD NAVIGATION INFORMATION
    # ==================================================================

    # --------------------------------------------------------------
    # Direction
    # --------------------------------------------------------------

    cv2.putText(
        result,
        f"COMMAND: {direction}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3
    )

    # --------------------------------------------------------------
    # Path quality
    # --------------------------------------------------------------

    cv2.putText(
        result,
        f"PATH: {path_quality}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------------
    # Start / goal
    # --------------------------------------------------------------

    cv2.putText(
        result,
        f"START: {start}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"GOAL: {goal}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # ==================================================================
    # SAVE FINAL RESULT
    # ==================================================================

    result_path = os.path.join(
        runs_dir,
        "visionnav_full_navigation_result.jpg"
    )

    cv2.imwrite(
        result_path,
        result
    )

    print(
        "[8] Navigation result saved:"
    )

    print(
        f"    {result_path}"
    )

    # ==================================================================
    # 12. FINAL PIPELINE RESULT
    # ==================================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FULL PIPELINE TEST PASSED"
    )

    print(
        "=" * 70
    )

    print(
        """
Camera Image
     ↓
YOLO Obstacle Detection
     ↓
Obstacle Filtering
     ↓
Traversability Detection
     ↓
Costmap V3
     ↓
Cost-Aware A*
     ↓
Navigation Direction
     ↓
Visualization
"""
    )

    print(
        f"Initial command: {direction}"
    )

    print(
        f"Path quality: {path_quality}"
    )

    print(
        f"Maximum path cell cost: "
        f"{maximum_path_cost}"
    )

    print(
        "=" * 70
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()