"""
VisionNav UGV - Dynamic Replanning Test

Milestone 2:
Dynamic obstacle detection + costmap update + A* replanning.

Pipeline:

    Frame
      ↓
    YOLO Detection
      ↓
    Traversability
      ↓
    Costmap
      ↓
    A* Path
      ↓
    Simulated UGV Movement
      ↓
    New Frame
      ↓
    Rebuild Costmap
      ↓
    Check Current Path
      ↓
    Replan if required
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
# IMPORT PROJECT MODULES
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
    "dynamic_replanning"
)

GRID_ROWS = 12
GRID_COLS = 16

OBSTACLE_INFLATION = 1

# Number of simulated UGV movement steps per frame.
MOVEMENT_STEPS = 2


# ======================================================================
# CREATE OUTPUT DIRECTORY
# ======================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ======================================================================
# DISCOVER SCENARIO IMAGES
# ======================================================================

def discover_images():

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
# START SELECTION
# ======================================================================

def choose_start(costmap):

    rows = costmap.rows
    cols = costmap.cols

    preferred_row = rows - 1
    preferred_col = cols // 2

    candidates = []

    # Prefer safe cells close to bottom center.
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

            candidates.append(
                (
                    distance,
                    row,
                    col
                )
            )

    if candidates:

        candidates.sort()

        return (
            candidates[0][1],
            candidates[0][2]
        )

    # Fallback to any traversable cell.
    candidates = []

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

            candidates.append(
                (
                    distance,
                    row,
                    col
                )
            )

    if candidates:

        candidates.sort()

        return (
            candidates[0][1],
            candidates[0][2]
        )

    return None


# ======================================================================
# GOAL SELECTION
# ======================================================================

def choose_goal(
    costmap,
    planner,
    start
):
    """
    Find the farthest reachable safe cell ahead.
    """

    rows = costmap.rows
    cols = costmap.cols

    start_row, start_col = start

    candidates = []

    for row in range(
        0,
        start_row
    ):

        for col in range(cols):

            if costmap.grid[row, col] != 0:
                continue

            path = planner.plan(
                start,
                (row, col)
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

            candidates.append(
                (
                    forward_distance,
                    center_distance,
                    len(path),
                    (row, col)
                )
            )

    if not candidates:
        return None

    # Farthest forward first.
    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2]
        )
    )

    return candidates[0][3]


# ======================================================================
# CHECK WHETHER CURRENT PATH IS STILL VALID
# ======================================================================

def path_is_valid(
    path,
    costmap,
    current_position
):
    """
    Check whether the remaining path is still traversable.

    Returns False if any future path cell is blocked.
    """

    if path is None:
        return False

    for cell in path:

        row, col = cell

        if not (
            0 <= row < costmap.rows
            and 0 <= col < costmap.cols
        ):
            return False

        if costmap.grid[row, col] >= 100:
            return False

    return True


# ======================================================================
# FIND PATH INDEX FOR CURRENT POSITION
# ======================================================================

def find_position_in_path(
    path,
    current_position
):

    if path is None:
        return None

    for index, cell in enumerate(path):

        if cell == current_position:
            return index

    return None


# ======================================================================
# GET NAVIGATION DIRECTION
# ======================================================================

def get_direction(
    current,
    next_cell
):

    if current is None or next_cell is None:
        return "STOP"

    dr = (
        next_cell[0]
        - current[0]
    )

    dc = (
        next_cell[1]
        - current[1]
    )

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
# BUILD COSTMAP FROM FRAME
# ======================================================================

def build_costmap(
    frame,
    yolo_detector,
    obstacle_detector,
    traversability_detector
):

    height, width = frame.shape[:2]

    # --------------------------------------------------------------
    # YOLO
    # --------------------------------------------------------------

    yolo_result, inference_ms = (
        yolo_detector.detect(
            frame
        )
    )

    # --------------------------------------------------------------
    # Obstacles
    # --------------------------------------------------------------

    obstacles = obstacle_detector.analyze(
        yolo_result,
        width,
        height
    )

    # --------------------------------------------------------------
    # Traversability
    # --------------------------------------------------------------

    mask = traversability_detector.detect(
        frame
    )

    # --------------------------------------------------------------
    # Costmap
    # --------------------------------------------------------------

    costmap = Costmap(
        rows=GRID_ROWS,
        cols=GRID_COLS,
        obstacle_inflation=OBSTACLE_INFLATION
    )

    obstacle_boxes = [
        obstacle.bbox
        for obstacle in obstacles
    ]

    costmap.build(
        traversability_mask=mask,
        boxes=obstacle_boxes,
        image_width=width,
        image_height=height
    )

    return (
        costmap,
        obstacles,
        mask,
        inference_ms
    )


# ======================================================================
# SAVE FRAME VISUALIZATION
# ======================================================================

def save_visualization(
    frame,
    obstacles,
    path,
    frame_number,
    visualizer
):

    result = visualizer.visualize(
        frame.copy(),
        obstacles,
        path,
        GRID_ROWS,
        GRID_COLS
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"frame_{frame_number:02d}_navigation.jpg"
    )

    cv2.imwrite(
        output_path,
        result
    )

    return output_path


# ======================================================================
# MAIN DYNAMIC REPLANNING TEST
# ======================================================================

def main():

    print("=" * 70)
    print("VISIONNAV UGV - DYNAMIC REPLANNING TEST")
    print("=" * 70)

    print(
        "\nThis test simulates a UGV receiving new camera frames."
    )

    print(
        "The system rebuilds the costmap and replans when necessary."
    )

    # ==================================================================
    # FIND IMAGES
    # ==================================================================

    images = discover_images()

    if not images:

        print(
            "\n[ERROR] No images found."
        )

        return

    print(
        f"\nFound {len(images)} frames:"
    )

    for filename in images:
        print(
            f"  - {filename}"
        )

    # ==================================================================
    # INITIALIZE
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INITIALIZING MODULES"
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
        "\nModules initialized."
    )

    # ==================================================================
    # STATE
    # ==================================================================

    current_position = None

    current_goal = None

    current_path = None

    total_replans = 0

    total_frames = 0

    # ==================================================================
    # PROCESS FRAMES
    # ==================================================================

    for frame_number, filename in enumerate(
        images,
        start=1
    ):

        print(
            "\n\n"
            + "=" * 70
        )

        print(
            f"FRAME {frame_number}: {filename}"
        )

        print(
            "=" * 70
        )

        image_path = os.path.join(
            INPUT_DIR,
            filename
        )

        frame = cv2.imread(
            image_path
        )

        if frame is None:

            print(
                "[ERROR] Could not read frame."
            )

            continue

        total_frames += 1

        # ==============================================================
        # PERCEPTION + COSTMAP
        # ==============================================================

        print(
            "\n[1] Perception + Costmap"
        )

        costmap_start = time.perf_counter()

        (
            costmap,
            obstacles,
            mask,
            inference_ms
        ) = build_costmap(
            frame,
            yolo_detector,
            obstacle_detector,
            traversability_detector
        )

        processing_ms = (
            time.perf_counter()
            - costmap_start
        ) * 1000

        print(
            f"YOLO inference: "
            f"{inference_ms:.2f} ms"
        )

        print(
            f"Total perception/costmap: "
            f"{processing_ms:.2f} ms"
        )

        print(
            f"Detected obstacles: "
            f"{len(obstacles)}"
        )

        for obstacle in obstacles:

            print(
                f"  - {obstacle.class_name} "
                f"conf={obstacle.confidence:.3f} "
                f"zones={','.join(obstacle.zones)} "
                f"bbox={obstacle.bbox}"
            )

        # ==============================================================
        # INITIAL POSITION
        # ==============================================================

        if current_position is None:

            current_position = choose_start(
                costmap
            )

            if current_position is None:

                print(
                    "[ERROR] Cannot find starting position."
                )

                continue

            print(
                f"\nInitial UGV position: "
                f"{current_position}"
            )

        else:

            print(
                f"\nCurrent UGV position: "
                f"{current_position}"
            )

        # ==============================================================
        # CHECK EXISTING PATH
        # ==============================================================

        need_replan = False

        if current_path is None:

            print(
                "\nNo existing path."
            )

            need_replan = True

        else:

            valid = path_is_valid(
                current_path,
                costmap,
                current_position
            )

            if valid:

                print(
                    "\nExisting path is still valid."
                )

            else:

                print(
                    "\n!!! EXISTING PATH BLOCKED !!!"
                )

                print(
                    "Dynamic obstacle detected."
                )

                print(
                    "A* replanning required."
                )

                need_replan = True

        # ==============================================================
        # REPLAN
        # ==============================================================

        if need_replan:

            print(
                "\n[2] A* REPLANNING"
            )

            planner = PathPlanner(
                costmap.grid
            )

            new_goal = choose_goal(
                costmap,
                planner,
                current_position
            )

            if new_goal is None:

                print(
                    "\n!!! NO REACHABLE GOAL !!!"
                )

                current_path = None
                current_goal = None

            else:

                planning_start = (
                    time.perf_counter()
                )

                new_path = planner.plan(
                    current_position,
                    new_goal
                )

                planning_ms = (
                    time.perf_counter()
                    - planning_start
                ) * 1000

                if new_path is None:

                    print(
                        "\nA* could not find a path."
                    )

                    current_path = None
                    current_goal = None

                else:

                    current_path = new_path

                    current_goal = new_goal

                    total_replans += 1

                    print(
                        "\n*** NEW PATH GENERATED ***"
                    )

                    print(
                        f"Goal: "
                        f"{current_goal}"
                    )

                    print(
                        f"Path length: "
                        f"{len(current_path)} cells"
                    )

                    print(
                        f"A* planning time: "
                        f"{planning_ms:.3f} ms"
                    )

                    print(
                        "\nNew path:"
                    )

                    print(
                        " -> ".join(
                            str(cell)
                            for cell in current_path
                        )
                    )

        else:

            print(
                "\n[2] A* REPLANNING"
            )

            print(
                "No replanning required."
            )

        # ==============================================================
        # NAVIGATION COMMAND
        # ==============================================================

        print(
            "\n[3] Navigation Command"
        )

        if current_path is None:

            print(
                "COMMAND: STOP"
            )

        else:

            path_index = find_position_in_path(
                current_path,
                current_position
            )

            if path_index is None:

                print(
                    "Current position is not on path."
                )

                print(
                    "Forcing replanning."
                )

                planner = PathPlanner(
                    costmap.grid
                )

                current_goal = choose_goal(
                    costmap,
                    planner,
                    current_position
                )

                if current_goal is not None:

                    current_path = planner.plan(
                        current_position,
                        current_goal
                    )

                    total_replans += 1

            if current_path:

                path_index = find_position_in_path(
                    current_path,
                    current_position
                )

                if (
                    path_index is not None
                    and path_index + 1 < len(current_path)
                ):

                    next_cell = current_path[
                        path_index + 1
                    ]

                    direction = get_direction(
                        current_position,
                        next_cell
                    )

                    print(
                        f"Current: "
                        f"{current_position}"
                    )

                    print(
                        f"Next: "
                        f"{next_cell}"
                    )

                    print(
                        f"Goal: "
                        f"{current_goal}"
                    )

                    print(
                        f"COMMAND: "
                        f"{direction}"
                    )

                else:

                    print(
                        "COMMAND: GOAL REACHED"
                    )

        # ==============================================================
        # VISUALIZATION
        # ==============================================================

        print(
            "\n[4] Visualization"
        )

        output_path = save_visualization(
            frame,
            obstacles,
            current_path,
            frame_number,
            visualizer
        )

        print(
            f"Saved: "
            f"{output_path}"
        )

        # ==============================================================
        # SIMULATED MOVEMENT
        # ==============================================================

        if current_path:

            path_index = find_position_in_path(
                current_path,
                current_position
            )

            if path_index is not None:

                movement_count = min(
                    MOVEMENT_STEPS,
                    len(current_path)
                    - path_index
                    - 1
                )

                if movement_count > 0:

                    new_position = current_path[
                        path_index
                        + movement_count
                    ]

                    print(
                        "\n[5] Simulated UGV Movement"
                    )

                    print(
                        f"Moving from "
                        f"{current_position} "
                        f"to "
                        f"{new_position}"
                    )

                    current_position = (
                        new_position
                    )

                else:

                    print(
                        "\nUGV has reached the current goal."
                    )

        # ==============================================================
        # FRAME SUMMARY
        # ==============================================================

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"FRAME {frame_number} SUMMARY"
        )

        print(
            "-" * 70
        )

        print(
            f"Position: "
            f"{current_position}"
        )

        print(
            f"Goal: "
            f"{current_goal}"
        )

        print(
            f"Path length: "
            f"{len(current_path) if current_path else 0}"
        )

        print(
            f"Total replans: "
            f"{total_replans}"
        )

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "DYNAMIC REPLANNING SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Frames processed: "
        f"{total_frames}"
    )

    print(
        f"Total A* replans: "
        f"{total_replans}"
    )

    print(
        f"Final UGV position: "
        f"{current_position}"
    )

    print(
        f"Final goal: "
        f"{current_goal}"
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DYNAMIC REPLANNING TEST COMPLETE"
    )

    print(
        "=" * 70
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()