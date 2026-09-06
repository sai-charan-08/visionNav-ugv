"""
navigation/multi_scenario_test.py

VisionNav UGV - Multi Scenario Navigation Test

Pipeline:
    Image
      ↓
    YOLO Object Detection
      ↓
    Obstacle Analysis
      ↓
    Traversability Detection
      ↓
    Costmap
      ↓
    Start / Goal Selection
      ↓
    A* Path Planning
      ↓
    Navigation Decision
      ↓
    Visualization

This is a software prototype using static camera images.
It does not represent physical motor control or real-time SLAM.
"""

import os
import sys
import time

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

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ======================================================================
# PROJECT IMPORTS
# ======================================================================

from perception.yolo_detector import YOLODetector
from perception.obstacle_detector import ObstacleDetector

from perception.traversability import (
    TraversabilityDetector,
    create_overlay
)

from navigation.costmap import Costmap
from navigation.path_planner import PathPlanner

from visualization.navigation_visualizer import (
    NavigationVisualizer
)


# ======================================================================
# CONFIGURATION
# ======================================================================

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "input"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "runs",
    "multi_scenario"
)

GRID_ROWS = 12
GRID_COLS = 16

OBSTACLE_INFLATION = 1


# ======================================================================
# CREATE OUTPUT DIRECTORY
# ======================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ======================================================================
# DISCOVER TEST IMAGES
# ======================================================================

def discover_test_images():
    """
    Automatically discover scenario images.

    test.jpeg is excluded because it is used for the standalone
    detector test.
    """

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    images = []

    if not os.path.isdir(INPUT_DIR):
        return images

    for filename in os.listdir(INPUT_DIR):

        if not filename.lower().endswith(
            valid_extensions
        ):
            continue

        if filename.lower() == "test.jpeg":
            continue

        images.append(
            filename
        )

    images.sort()

    return images


# ======================================================================
# NAVIGATION DIRECTION
# ======================================================================

def get_navigation_direction(path):
    """
    Determine the initial navigation direction from the first
    movement in the A* path.

    Grid convention:

        row decreases → forward
        col increases → right
        col decreases → left
    """

    if path is None or len(path) < 2:
        return "STOP"

    start = path[0]
    next_cell = path[1]

    dr = next_cell[0] - start[0]
    dc = next_cell[1] - start[1]

    if dr < 0 and dc == 0:
        return "FORWARD"

    if dr > 0 and dc == 0:
        return "BACKWARD"

    if dr == 0 and dc > 0:
        return "RIGHT"

    if dr == 0 and dc < 0:
        return "LEFT"

    if dr < 0 and dc > 0:
        return "FORWARD-RIGHT"

    if dr < 0 and dc < 0:
        return "FORWARD-LEFT"

    if dr > 0 and dc > 0:
        return "BACKWARD-RIGHT"

    if dr > 0 and dc < 0:
        return "BACKWARD-LEFT"

    return "STOP"


# ======================================================================
# PATH QUALITY
# ======================================================================

def get_path_quality(path, costmap):
    """
    Evaluate the maximum cost encountered along the path.
    """

    if path is None or len(path) == 0:
        return "BLOCKED"

    max_cost = 0

    for row, col in path:

        cost = int(
            costmap.grid[row, col]
        )

        max_cost = max(
            max_cost,
            cost
        )

    if max_cost == 0:
        return "SAFE"

    if max_cost <= 30:
        return "LOW RISK / UNCERTAIN"

    if max_cost < 100:
        return "DIFFICULT"

    return "BLOCKED"


# ======================================================================
# START SELECTION
# ======================================================================

def choose_start(costmap):
    """
    Choose a start cell near the bottom-center of the costmap.

    Priority:

        1. Safe cell
        2. Any non-blocked cell

    The bottom portion of the grid represents the area immediately
    around the UGV.
    """

    rows = costmap.rows
    cols = costmap.cols

    preferred_row = rows - 1
    preferred_col = cols // 2

    # --------------------------------------------------------------
    # First pass: SAFE cells
    # --------------------------------------------------------------

    safe_candidates = []

    for row in range(
        max(0, rows - 4),
        rows
    ):

        for col in range(cols):

            if costmap.grid[row, col] != 0:
                continue

            distance = (
                abs(row - preferred_row)
                + abs(col - preferred_col)
            )

            safe_candidates.append(
                (
                    distance,
                    row,
                    col
                )
            )

    if safe_candidates:

        safe_candidates.sort()

        return (
            safe_candidates[0][1],
            safe_candidates[0][2]
        )

    # --------------------------------------------------------------
    # Second pass: any traversable cell
    # --------------------------------------------------------------

    fallback_candidates = []

    for row in range(
        max(0, rows - 4),
        rows
    ):

        for col in range(cols):

            if costmap.grid[row, col] >= 100:
                continue

            distance = (
                abs(row - preferred_row)
                + abs(col - preferred_col)
            )

            fallback_candidates.append(
                (
                    distance,
                    row,
                    col
                )
            )

    if fallback_candidates:

        fallback_candidates.sort()

        return (
            fallback_candidates[0][1],
            fallback_candidates[0][2]
        )

    return None


# ======================================================================
# IMPROVED GOAL SELECTION
# ======================================================================

def choose_reachable_goal(
    costmap,
    planner,
    start
):
    """
    Select a meaningful forward goal.

    The previous goal-selection logic could choose a nearby safe
    cell because the traversability mask was sparse.

    This version:

        1. Searches for goals farther ahead.
        2. Requires safe cells.
        3. Checks reachability using A*.
        4. Prefers the farthest reachable goal.
        5. Prefers cells closer to the center.
        6. Falls back to the farthest reachable safe cell.

    This encourages a longer and more meaningful navigation path.
    """

    rows = costmap.rows
    cols = costmap.cols

    start_row, start_col = start

    # --------------------------------------------------------------
    # Minimum meaningful forward distance
    # --------------------------------------------------------------

    min_forward_distance = max(
        3,
        rows // 4
    )

    # --------------------------------------------------------------
    # Candidate goals
    # --------------------------------------------------------------

    candidates = []

    for row in range(rows):

        # Goal must be forward of the UGV.
        if row >= start_row:
            continue

        forward_distance = (
            start_row - row
        )

        # Ignore goals that are too close.
        if forward_distance < min_forward_distance:
            continue

        for col in range(cols):

            # Goal must be completely safe.
            if costmap.grid[row, col] != 0:
                continue

            center_distance = abs(
                col - cols // 2
            )

            candidates.append(
                (
                    row,
                    center_distance,
                    col
                )
            )

    # --------------------------------------------------------------
    # Sort:
    #
    # Smaller row = farther forward.
    # Smaller center distance = closer to center.
    # --------------------------------------------------------------

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1]
        )
    )

    # --------------------------------------------------------------
    # Find reachable goals
    # --------------------------------------------------------------

    reachable_candidates = []

    for row, center_distance, col in candidates:

        goal = (
            row,
            col
        )

        path = planner.plan(
            start,
            goal
        )

        if path is None:
            continue

        if len(path) < 2:
            continue

        forward_distance = (
            start_row - row
        )

        reachable_candidates.append(
            (
                forward_distance,
                center_distance,
                len(path),
                goal
            )
        )

    # --------------------------------------------------------------
    # Select best meaningful goal
    # --------------------------------------------------------------

    if reachable_candidates:

        reachable_candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2]
            )
        )

        return reachable_candidates[0][3]

    # --------------------------------------------------------------
    # FALLBACK
    #
    # If no goal satisfies the minimum forward distance, find the
    # farthest reachable safe goal.
    # --------------------------------------------------------------

    fallback_candidates = []

    for row in range(
        0,
        start_row
    ):

        for col in range(cols):

            if costmap.grid[row, col] != 0:
                continue

            goal = (
                row,
                col
            )

            path = planner.plan(
                start,
                goal
            )

            if path is None:
                continue

            if len(path) < 2:
                continue

            forward_distance = (
                start_row - row
            )

            center_distance = abs(
                col - cols // 2
            )

            fallback_candidates.append(
                (
                    forward_distance,
                    center_distance,
                    len(path),
                    goal
                )
            )

    if fallback_candidates:

        fallback_candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2]
            )
        )

        return fallback_candidates[0][3]

    return None


# ======================================================================
# COSTMAP DISPLAY
# ======================================================================

def print_costmap(costmap):
    """
    Print an ASCII representation of the costmap.
    """

    print("\nCostmap:")

    for row in range(
        costmap.rows
    ):

        row_text = []

        for col in range(
            costmap.cols
        ):

            value = int(
                costmap.grid[row, col]
            )

            if value >= 100:
                symbol = "#"

            elif value >= 50:
                symbol = "D"

            elif value >= 20:
                symbol = "?"

            else:
                symbol = "."

            row_text.append(
                symbol
            )

        print(
            " ".join(row_text)
        )

    print("\nLegend:")
    print(". = SAFE")
    print("? = UNCERTAIN")
    print("D = DIFFICULT")
    print("# = BLOCKED")


# ======================================================================
# PROCESS ONE IMAGE
# ======================================================================

def process_image(
    image_filename,
    yolo_detector,
    obstacle_detector,
    traversability_detector,
    visualizer
):
    """
    Process one scenario image through the complete navigation
    pipeline.
    """

    image_path = os.path.join(
        INPUT_DIR,
        image_filename
    )

    scenario_name = os.path.splitext(
        image_filename
    )[0]

    print("\n")
    print("=" * 70)
    print(
        f"SCENARIO: {image_filename}"
    )
    print("=" * 70)

    # ==================================================================
    # READ IMAGE
    # ==================================================================

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        print(
            "[ERROR] Could not read image."
        )

        return {
            "success": False,
            "scenario": scenario_name
        }

    height, width = frame.shape[:2]

    print(
        f"Camera image: {width} x {height}"
    )

    # ==================================================================
    # 1. YOLO OBJECT DETECTION
    # ==================================================================

    print(
        "\n[1] YOLO Object Detection"
    )

    yolo_result, inference_ms = (
        yolo_detector.detect(
            frame
        )
    )

    print(
        f"YOLO inference: "
        f"{inference_ms:.2f} ms"
    )

    # ==================================================================
    # 2. OBSTACLE ANALYSIS
    # ==================================================================

    print(
        "\n[2] Obstacle Analysis"
    )

    obstacles = obstacle_detector.analyze(
        yolo_result,
        width,
        height
    )

    print(
        "Raw/filtered obstacle detections: "
        f"{len(obstacles)}"
    )

    for obstacle in obstacles:

        print(
            f"  - {obstacle.class_name} "
            f"conf={obstacle.confidence:.3f} "
            f"zones={','.join(obstacle.zones)} "
            f"severity={obstacle.severity:.3f} "
            f"large={obstacle.is_large} "
            f"bbox={obstacle.bbox}"
        )

    # ==================================================================
    # 3. TRAVERSABILITY
    # ==================================================================

    print(
        "\n[3] Traversability Detection"
    )

    mask = traversability_detector.detect(
        frame
    )

    traversable_pixels = np.sum(
        mask == 1
    )

    total_pixels = mask.size

    traversability_percentage = (
        traversable_pixels
        / total_pixels
        * 100
    )

    print(
        f"Traversable area: "
        f"{traversability_percentage:.2f}%"
    )

    # ------------------------------------------------------------------
    # Save binary mask
    # ------------------------------------------------------------------

    mask_path = os.path.join(
        OUTPUT_DIR,
        f"{scenario_name}_traversability_mask.jpg"
    )

    cv2.imwrite(
        mask_path,
        mask * 255
    )

    # ------------------------------------------------------------------
    # Save overlay
    # ------------------------------------------------------------------

    overlay_path = os.path.join(
        OUTPUT_DIR,
        f"{scenario_name}_traversability_overlay.jpg"
    )

    overlay = create_overlay(
        frame,
        mask
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )

    # ==================================================================
    # 4. COSTMAP GENERATION
    # ==================================================================

    print(
        "\n[4] Costmap Generation"
    )

    costmap = Costmap(
        rows=GRID_ROWS,
        cols=GRID_COLS,
        obstacle_inflation=OBSTACLE_INFLATION
    )

    obstacle_boxes = []

    for obstacle in obstacles:

        obstacle_boxes.append(
            obstacle.bbox
        )

    costmap.build(
        traversability_mask=mask,
        boxes=obstacle_boxes,
        image_width=width,
        image_height=height
    )

    print_costmap(
        costmap
    )

    # ==================================================================
    # 5. START / GOAL SELECTION
    # ==================================================================

    print(
        "\n[5] Start / Goal Selection"
    )

    preferred_start = (
        GRID_ROWS - 1,
        GRID_COLS // 2
    )

    preferred_goal = (
        int(GRID_ROWS * 0.35),
        GRID_COLS // 2
    )

    print(
        f"Preferred start: {preferred_start}"
    )

    print(
        f"Preferred goal:   {preferred_goal}"
    )

    # ------------------------------------------------------------------
    # Choose start
    # ------------------------------------------------------------------

    start = choose_start(
        costmap
    )

    if start is None:

        print(
            "[ERROR] No valid start cell."
        )

        return {
            "success": False,
            "scenario": scenario_name,
            "obstacles": len(obstacles),
            "traversability": traversability_percentage
        }

    print(
        f"Selected start:   {start}"
    )

    # ------------------------------------------------------------------
    # Create A* planner
    # ------------------------------------------------------------------

    planner = PathPlanner(
        costmap.grid
    )

    # ------------------------------------------------------------------
    # Choose meaningful reachable goal
    # ------------------------------------------------------------------

    goal = choose_reachable_goal(
        costmap,
        planner,
        start
    )

    if goal is None:

        print(
            "[ERROR] No reachable goal found."
        )

        return {
            "success": False,
            "scenario": scenario_name,
            "obstacles": len(obstacles),
            "traversability": traversability_percentage,
            "start": start
        }

    print(
        f"Selected goal:    {goal}"
    )

    # ==================================================================
    # 6. A* PATH PLANNING
    # ==================================================================

    print(
        "\n[6] A* Path Planning"
    )

    planning_start = time.perf_counter()

    path = planner.plan(
        start,
        goal
    )

    planning_time_ms = (
        time.perf_counter()
        - planning_start
    ) * 1000

    if path is None:

        print(
            "A* RESULT: NO PATH"
        )

        return {
            "success": False,
            "scenario": scenario_name,
            "obstacles": len(obstacles),
            "traversability": traversability_percentage,
            "start": start,
            "goal": goal,
            "path_length": 0
        }

    print(
        "A* RESULT: PATH FOUND"
    )

    print(
        f"Path length: "
        f"{len(path)} cells"
    )

    print(
        f"Planning time: "
        f"{planning_time_ms:.3f} ms"
    )

    print(
        "\nPath:"
    )

    print(
        " -> ".join(
            str(cell)
            for cell in path
        )
    )

    # ==================================================================
    # 7. NAVIGATION DECISION
    # ==================================================================

    print(
        "\n[7] Navigation Decision"
    )

    path_quality = get_path_quality(
        path,
        costmap
    )

    direction = get_navigation_direction(
        path
    )

    print(
        f"Path quality: "
        f"{path_quality}"
    )

    print(
        f"Initial direction: "
        f"{direction}"
    )

    # ==================================================================
    # 8. VISUALIZATION
    # ==================================================================

    print(
        "\n[8] Visualization"
    )

    # IMPORTANT:
    #
    # Your existing NavigationVisualizer.visualize() accepts:
    #
    #     frame
    #     obstacles
    #     path
    #     rows
    #     cols
    #
    # Do NOT pass start and goal separately.

    result_image = visualizer.visualize(
        frame.copy(),
        obstacles,
        path,
        GRID_ROWS,
        GRID_COLS
    )

    result_path = os.path.join(
        OUTPUT_DIR,
        f"{scenario_name}_navigation_result.jpg"
    )

    cv2.imwrite(
        result_path,
        result_image
    )

    print(
        "\nResult saved to:"
    )

    print(
        f"  {result_path}"
    )

    print(
        "Mask saved to:"
    )

    print(
        f"  {mask_path}"
    )

    # ==================================================================
    # RETURN RESULT
    # ==================================================================

    return {
        "success": True,
        "scenario": scenario_name,
        "obstacles": len(obstacles),
        "traversability": traversability_percentage,
        "start": start,
        "goal": goal,
        "path_length": len(path),
        "path_quality": path_quality,
        "direction": direction,
        "yolo_ms": inference_ms,
        "astar_ms": planning_time_ms
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)

    print(
        "VISIONNAV UGV - MULTI SCENARIO TEST"
    )

    print("=" * 70)

    print(
        "\nInput directory:"
    )

    print(
        INPUT_DIR
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    # ==================================================================
    # DISCOVER TEST IMAGES
    # ==================================================================

    test_images = discover_test_images()

    print(
        f"\nFound {len(test_images)} test images:"
    )

    for filename in test_images:

        print(
            f"  - {filename}"
        )

    if not test_images:

        print(
            "\n[ERROR] No test images found."
        )

        return

    # ==================================================================
    # INITIALIZE MODULES
    # ==================================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "INITIALIZING VISION MODULES"
    )

    print(
        "=" * 70
    )

    yolo_detector = YOLODetector()

    obstacle_detector = (
        ObstacleDetector()
    )

    traversability_detector = (
        TraversabilityDetector()
    )

    visualizer = (
        NavigationVisualizer()
    )

    print(
        "\nAll modules initialized successfully."
    )

    # ==================================================================
    # PROCESS ALL SCENARIOS
    # ==================================================================

    results = []

    for index, image_filename in enumerate(
        test_images,
        start=1
    ):

        print(
            "\n\n"
            + "=" * 70
        )

        print(
            f"Processing scenario "
            f"{index}/{len(test_images)}"
        )

        print(
            "=" * 70
        )

        result = process_image(
            image_filename,
            yolo_detector,
            obstacle_detector,
            traversability_detector,
            visualizer
        )

        results.append(
            result
        )

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "FINAL TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    passed = 0

    for result in results:

        scenario = result.get(
            "scenario",
            "unknown"
        )

        success = result.get(
            "success",
            False
        )

        status = (
            "SUCCESS"
            if success
            else "FAILED"
        )

        if success:
            passed += 1

        print(
            f"\n{scenario:<30} {status}"
        )

        if not success:
            continue

        print(
            f"   Obstacles: "
            f"{result['obstacles']}"
        )

        print(
            f"   Traversability: "
            f"{result['traversability']:.1f}%"
        )

        print(
            f"   Start: "
            f"{result['start']}"
        )

        print(
            f"   Goal: "
            f"{result['goal']}"
        )

        print(
            f"   Path length: "
            f"{result['path_length']}"
        )

        print(
            f"   Path quality: "
            f"{result['path_quality']}"
        )

        print(
            f"   Direction: "
            f"{result['direction']}"
        )

        print(
            f"   YOLO: "
            f"{result['yolo_ms']:.1f} ms"
        )

        print(
            f"   A*: "
            f"{result['astar_ms']:.3f} ms"
        )

    print(
        f"\n\nScenarios passed: "
        f"{passed}/{len(results)}"
    )

    print(
        "\nAll output images are located in:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MULTI-SCENARIO TEST COMPLETE"
    )

    print(
        "=" * 70
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()