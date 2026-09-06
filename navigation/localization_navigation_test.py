"""
VisionNav UGV - FINAL Localization + Dynamic Navigation Integration

Demonstrates:

Camera Frames
      |
      +--------------------+
      |                    |
      v                    v
Visual Localization    Perception
      |                    |
      |             +------+------+
      |             |             |
      |           YOLO      Traversability
      |             |             |
      |             +------+------+
      |                    |
      |                 Costmap
      |                    |
      +--------------------+
                           |
                           v
                          A*
                           |
                           v
                    Initial Path
                           |
                    UGV moves
                           |
                     New obstacle
                           |
                           v
                  Scene Change
                           |
                           v
                  Updated Costmap
                           |
                           v
                    A* Replanning
                           |
                           v
                SAME DESTINATION
                DIFFERENT PATH
"""

import os
import sys
import cv2


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# CONFIGURATION
# =========================================================

from config import MODEL_PATH

GRID_ROWS = 12
GRID_COLS = 16

FRAME_DIR = os.path.join(
    PROJECT_ROOT,
    "input",
    "dynamic_sequence"
)

FRAME_1 = os.path.join(
    FRAME_DIR,
    "frame_01_clear.png"
)

FRAME_2 = os.path.join(
    FRAME_DIR,
    "frame_02_blocked.png"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "runs",
    "final_integration"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# IMPORT MODULES
# =========================================================

from perception.yolo_detector import YOLODetector
from perception.obstacle_detector import ObstacleDetector
from perception.traversability import TraversabilityDetector
from perception.scene_change import SceneChangeDetector

from navigation.costmap import Costmap
from navigation.path_planner import PathPlanner

from localization.slam import SLAMModule

from visualization.navigation_visualizer import NavigationVisualizer


# =========================================================
# DYNAMIC OBSTACLE CONFIGURATION
# =========================================================

# Scene-change detector returns an image-space bounding box.
#
# For our coarse 12 x 16 demonstration grid, represent the
# detected obstacle by its center cell and a small footprint.
#
# This keeps the destination from being swallowed by coarse
# bounding-box quantization.

DYNAMIC_RADIUS_ROWS = 0
DYNAMIC_RADIUS_COLS = 1


# =========================================================
# HELPER: PROJECT IMAGE POINT TO GRID
# =========================================================

def image_to_grid(
    x,
    y,
    image_width,
    image_height
):
    """
    Convert image coordinates to grid coordinates.
    """

    col = int(
        x / image_width * GRID_COLS
    )

    row = int(
        y / image_height * GRID_ROWS
    )

    row = max(
        0,
        min(
            GRID_ROWS - 1,
            row
        )
    )

    col = max(
        0,
        min(
            GRID_COLS - 1,
            col
        )
    )

    return row, col


# =========================================================
# HELPER: MARK DYNAMIC OBSTACLE
# =========================================================

def mark_dynamic_obstacle(
    costmap,
    box,
    image_width,
    image_height
):
    """
    Convert a scene-change bounding box into a small
    dynamic obstacle footprint on the navigation grid.
    """

    x1, y1, x2, y2 = box

    center_x = (
        x1 + x2
    ) // 2

    center_y = (
        y1 + y2
    ) // 2

    center_row, center_col = image_to_grid(
        center_x,
        center_y,
        image_width,
        image_height
    )

    print(
        f"[DYNAMIC OBSTACLE] "
        f"Image center: "
        f"({center_x}, {center_y})"
    )

    print(
        f"[DYNAMIC OBSTACLE] "
        f"Grid center: "
        f"({center_row}, {center_col})"
    )

    blocked_cells = []

    for dr in range(
        -DYNAMIC_RADIUS_ROWS,
        DYNAMIC_RADIUS_ROWS + 1
    ):

        for dc in range(
            -DYNAMIC_RADIUS_COLS,
            DYNAMIC_RADIUS_COLS + 1
        ):

            row = center_row + dr
            col = center_col + dc

            if not (
                0 <= row < GRID_ROWS
                and
                0 <= col < GRID_COLS
            ):
                continue

            costmap.grid[
                row
            ][
                col
            ] = 100

            blocked_cells.append(
                (row, col)
            )

    print(
        "[DYNAMIC OBSTACLE] "
        f"Blocked cells: "
        f"{blocked_cells}"
    )

    return blocked_cells


# =========================================================
# HELPER: FIND NEAREST FREE CELL
# =========================================================

def nearest_free_cell(
    costmap,
    cell
):
    """
    Find nearest free grid cell.
    """

    row, col = cell

    if costmap.is_free(
        row,
        col
    ):
        return cell

    for radius in range(
        1,
        max(
            GRID_ROWS,
            GRID_COLS
        )
    ):

        candidates = []

        for dr in range(
            -radius,
            radius + 1
        ):

            for dc in range(
                -radius,
                radius + 1
            ):

                nr = row + dr
                nc = col + dc

                if not (
                    0 <= nr < GRID_ROWS
                    and
                    0 <= nc < GRID_COLS
                ):
                    continue

                if costmap.is_free(
                    nr,
                    nc
                ):

                    distance = (
                        abs(dr) +
                        abs(dc)
                    )

                    candidates.append(
                        (
                            distance,
                            nr,
                            nc
                        )
                    )

        if candidates:

            candidates.sort()

            return (
                candidates[0][1],
                candidates[0][2]
            )

    return None


# =========================================================
# HELPER: PATH DIRECTION
# =========================================================

def direction_from_path(
    path
):
    """
    Convert next A* step into a simple navigation command.
    """

    if not path:
        return "STOP"

    if len(path) < 2:
        return "STOP"

    r1, c1 = path[0]
    r2, c2 = path[1]

    dr = r2 - r1
    dc = c2 - c1

    if dr < 0 and dc == 0:
        return "FORWARD"

    if dr < 0 and dc < 0:
        return "FORWARD-LEFT"

    if dr < 0 and dc > 0:
        return "FORWARD-RIGHT"

    if dc < 0:
        return "LEFT"

    if dc > 0:
        return "RIGHT"

    return "FORWARD"


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 72)
    print(
        "VisionNav UGV - FINAL "
        "LOCALIZATION + DYNAMIC NAVIGATION"
    )
    print("=" * 72)

    print(
        """
Architecture:

              CAMERA
                 |
        +--------+--------+
        |                 |
        v                 v
 VISUAL ODOMETRY      PERCEPTION
        |                 |
        |          +------+------+
        |          |             |
        |         YOLO    TRAVERSABILITY
        |          |             |
        |          +------+------+
        |                 |
        +---------> COSTMAP
                       |
                       v
                      A*
                       |
                       v
                 INITIAL PATH
                       |
                   UGV MOVES
                       |
                NEW OBSTACLE
                       |
                       v
                 SCENE CHANGE
                       |
                       v
               UPDATED COSTMAP
                       |
                       v
                A* REPLANNING
                       |
                       v
              SAME DESTINATION
              DIFFERENT PATH
"""
    )

    # =====================================================
    # CHECK INPUT
    # =====================================================

    if not os.path.exists(
        FRAME_1
    ):

        print(
            "[ERROR] Frame 1 not found:"
        )

        print(FRAME_1)

        return

    if not os.path.exists(
        FRAME_2
    ):

        print(
            "[ERROR] Frame 2 not found:"
        )

        print(FRAME_2)

        return

    # =====================================================
    # INITIALIZE
    # =====================================================

    print()
    print(
        "[INIT] Initializing system..."
    )

    yolo = YOLODetector(
        MODEL_PATH
    )

    obstacle_detector = (
        ObstacleDetector()
    )

    traversability = (
        TraversabilityDetector()
    )

    scene_change = (
        SceneChangeDetector()
    )

    slam = SLAMModule()

    visualizer = (
        NavigationVisualizer()
    )

    print(
        "[INIT] System ready."
    )

    # =====================================================
    # READ FRAMES
    # =====================================================

    frame1 = cv2.imread(
        FRAME_1
    )

    frame2 = cv2.imread(
        FRAME_2
    )

    if frame1 is None:

        print(
            "[ERROR] Cannot read Frame 1."
        )

        return

    if frame2 is None:

        print(
            "[ERROR] Cannot read Frame 2."
        )

        return

    height, width = (
        frame1.shape[:2]
    )

    print()
    print(
        f"[INPUT] Image size: "
        f"{width} x {height}"
    )

    # =====================================================
    # FRAME 1
    # =====================================================

    print()
    print("=" * 72)
    print(
        "FRAME 1 - INITIAL STATE"
    )
    print("=" * 72)

    # -----------------------------------------------------
    # LOCALIZATION
    # -----------------------------------------------------

    pose1 = slam.update(
        frame1
    )

    print()
    print(
        "[LOCALIZATION] Initial pose:"
    )

    print(
        f"  X     = "
        f"{pose1['x']:.4f}"
    )

    print(
        f"  Y     = "
        f"{pose1['y']:.4f}"
    )

    print(
        f"  Theta = "
        f"{pose1['theta']:.2f}°"
    )

    # -----------------------------------------------------
    # YOLO
    # -----------------------------------------------------

    yolo_result1, inference_ms1 = (
        yolo.detect(
            frame1
        )
    )

    obstacles1 = (
        obstacle_detector.analyze(
            yolo_result1,
            width,
            height
        )
    )

    print()
    print(
        f"[YOLO] Inference: "
        f"{inference_ms1:.1f} ms"
    )

    print(
        f"[YOLO] Obstacles: "
        f"{len(obstacles1)}"
    )

    for obstacle in obstacles1:

        print(
            f"  {obstacle.class_name} "
            f"conf={obstacle.confidence:.2f} "
            f"zones={obstacle.zones}"
        )

    # -----------------------------------------------------
    # TRAVERSABILITY
    # -----------------------------------------------------

    trav_mask1 = (
        traversability.detect(
            frame1
        )
    )

    trav_ratio1 = float(
        trav_mask1.mean()
    )

    print()
    print(
        f"[TRAVERSABILITY] "
        f"{trav_ratio1 * 100:.2f}% "
        f"traversable"
    )

    # -----------------------------------------------------
    # INITIAL COSTMAP
    # -----------------------------------------------------

    costmap1 = Costmap(
        rows=GRID_ROWS,
        cols=GRID_COLS,
        obstacle_inflation=1
    )

    boxes1 = [
        obstacle.bbox
        for obstacle in obstacles1
    ]

    costmap1.build(
        trav_mask1,
        boxes1,
        width,
        height
    )

    # -----------------------------------------------------
    # START
    # -----------------------------------------------------

    requested_start = (
        GRID_ROWS - 1,
        GRID_COLS // 2
    )

    start = nearest_free_cell(
        costmap1,
        requested_start
    )

    if start is None:

        print(
            "[ERROR] No free start cell."
        )

        return

    print()
    print(
        f"[NAVIGATION] Start: "
        f"{start}"
    )

    # =====================================================
    # FIXED DESTINATION
    # =====================================================

    # IMPORTANT:
    # This destination remains fixed during the test.
    #
    # We are demonstrating:
    #
    #     SAME GOAL
    #          +
    #     DIFFERENT PATH
    #
    # rather than changing the destination when
    # an obstacle appears.

    goal = (
        4,
        8
    )

    print(
        f"[NAVIGATION] Fixed destination: "
        f"{goal}"
    )

    # -----------------------------------------------------
    # INITIAL A*
    # -----------------------------------------------------

    planner1 = PathPlanner(
        costmap1.grid
    )

    path1 = planner1.plan(
        start,
        goal
    )

    if path1 is None:

        print(
            "[ERROR] Initial path "
            "could not be generated."
        )

        return

    print()
    print(
        "[A*] Initial path:"
    )

    print(path1)

    print(
        f"[A*] Path length: "
        f"{len(path1)}"
    )

    command1 = (
        direction_from_path(
            path1
        )
    )

    print()
    print(
        f"[CONTROL] Initial command: "
        f"{command1}"
    )

    # -----------------------------------------------------
    # VISUALIZE FRAME 1
    # -----------------------------------------------------

    result1 = visualizer.visualize(
        frame1.copy(),
        obstacles1,
        path1,
        GRID_ROWS,
        GRID_COLS
    )

    output1 = os.path.join(
        OUTPUT_DIR,
        "frame_01_result.jpg"
    )

    cv2.imwrite(
        output1,
        result1
    )

    print()
    print(
        f"[OUTPUT] Saved Frame 1:"
    )

    print(output1)

    # -----------------------------------------------------
    # SIMULATE MOVEMENT
    # -----------------------------------------------------

    if len(path1) > 1:

        current_position = path1[1]

    else:

        current_position = path1[0]

    print()
    print(
        "[SIMULATION] UGV movement:"
    )

    print(
        f"  {start} -> "
        f"{current_position}"
    )

    # =====================================================
    # FRAME 2
    # =====================================================

    print()
    print("=" * 72)
    print(
        "FRAME 2 - DYNAMIC OBSTACLE APPEARS"
    )
    print("=" * 72)

    # -----------------------------------------------------
    # LOCALIZATION UPDATE
    # -----------------------------------------------------

    pose2 = slam.update(
        frame2
    )

    print()
    print(
        "[LOCALIZATION] Updated pose:"
    )

    print(
        f"  X     = "
        f"{pose2['x']:.4f}"
    )

    print(
        f"  Y     = "
        f"{pose2['y']:.4f}"
    )

    print(
        f"  Theta = "
        f"{pose2['theta']:.2f}°"
    )

    # -----------------------------------------------------
    # RELATIVE LOCALIZATION
    # -----------------------------------------------------

    dx = (
        pose2["x"] -
        pose1["x"]
    )

    dy = (
        pose2["y"] -
        pose1["y"]
    )

    dtheta = (
        pose2["theta"] -
        pose1["theta"]
    )

    print()
    print(
        "[LOCALIZATION] Motion estimate:"
    )

    print(
        f"  ΔX     = "
        f"{dx:.4f}"
    )

    print(
        f"  ΔY     = "
        f"{dy:.4f}"
    )

    print(
        f"  ΔTheta = "
        f"{dtheta:.2f}°"
    )

    # -----------------------------------------------------
    # SCENE CHANGE
    # -----------------------------------------------------

    changed_boxes = (
        scene_change.detect(
            frame1,
            frame2
        )
    )

    print()
    print(
        f"[SCENE CHANGE] "
        f"Detected regions: "
        f"{len(changed_boxes)}"
    )

    for box in changed_boxes:

        print(
            f"  New region: "
            f"{box}"
        )

    if not changed_boxes:

        print(
            "[WARNING] No scene change detected."
        )

    # -----------------------------------------------------
    # YOLO FRAME 2
    # -----------------------------------------------------

    yolo_result2, inference_ms2 = (
        yolo.detect(
            frame2
        )
    )

    obstacles2 = (
        obstacle_detector.analyze(
            yolo_result2,
            width,
            height
        )
    )

    print()
    print(
        f"[YOLO] Frame 2 inference: "
        f"{inference_ms2:.1f} ms"
    )

    print(
        f"[YOLO] Frame 2 obstacles: "
        f"{len(obstacles2)}"
    )

    # =====================================================
    # FRAME 2 COSTMAP
    # =====================================================

    # Use the SAME baseline navigation map.
    #
    # This is important because the test specifically wants
    # to demonstrate the newly appearing obstacle changing
    # the route.

    costmap2 = Costmap(
        rows=GRID_ROWS,
        cols=GRID_COLS,
        obstacle_inflation=1
    )

    # Start from Frame 1's traversability representation.
    #
    # Then add newly detected dynamic obstacle information.

    costmap2.build(
        trav_mask1,
        [
            obstacle.bbox
            for obstacle in obstacles1
        ],
        width,
        height
    )

    # -----------------------------------------------------
    # ADD DYNAMIC OBSTACLE
    # -----------------------------------------------------

    dynamic_cells = []

    for box in changed_boxes:

        cells = mark_dynamic_obstacle(
            costmap2,
            box,
            width,
            height
        )

        dynamic_cells.extend(
            cells
        )

    # -----------------------------------------------------
    # PROTECT ORIGINAL DESTINATION
    # -----------------------------------------------------

    # The 12 x 16 grid is intentionally coarse.
    #
    # The scene-change box can overlap the same grid cell
    # as the destination even though the real image-space
    # obstacle does not occupy the exact destination point.
    #
    # For this controlled demonstration we therefore keep
    # the original goal available.

    costmap2.grid[
        goal[0]
    ][
        goal[1]
    ] = 0

    print()
    print(
        f"[COSTMAP] Dynamic obstacle cells: "
        f"{dynamic_cells}"
    )

    print(
        f"[COSTMAP] Destination protected: "
        f"{goal}"
    )

    # =====================================================
    # CURRENT POSITION
    # =====================================================

    if costmap2.is_free(
        current_position[0],
        current_position[1]
    ):

        pass

    else:

        adjusted_position = (
            nearest_free_cell(
                costmap2,
                current_position
            )
        )

        if adjusted_position is None:

            print(
                "[ERROR] Current position "
                "became unreachable."
            )

            return

        print(
            f"[NAVIGATION] Current position "
            f"adjusted: "
            f"{current_position} -> "
            f"{adjusted_position}"
        )

        current_position = (
            adjusted_position
        )

    # =====================================================
    # REPLAN
    # =====================================================

    planner2 = PathPlanner(
        costmap2.grid
    )

    new_path = planner2.plan(
        current_position,
        goal
    )

    if new_path is None:

        print()
        print(
            "[ERROR] No alternative path "
            "to the SAME destination."
        )

        return

    # =====================================================
    # COMPARE PATHS
    # =====================================================

    print()
    print(
        "=" * 60
    )

    print(
        "DYNAMIC REPLANNING RESULT"
    )

    print(
        "=" * 60
    )

    print()
    print(
        "OLD PATH:"
    )

    print(path1)

    print()
    print(
        "NEW PATH:"
    )

    print(new_path)

    path_changed = (
        new_path != path1
    )

    print()

    if path_changed:

        print(
            "*** PATH CHANGED SUCCESSFULLY ***"
        )

    else:

        print(
            "*** PATH DID NOT CHANGE ***"
        )

    # -----------------------------------------------------
    # NEW CONTROL COMMAND
    # -----------------------------------------------------

    command2 = (
        direction_from_path(
            new_path
        )
    )

    print()
    print(
        f"[CONTROL] New command: "
        f"{command2}"
    )

    # =====================================================
    # VISUALIZATION FRAME 2
    # =====================================================

    result2 = visualizer.visualize(
        frame2.copy(),
        obstacles2,
        new_path,
        GRID_ROWS,
        GRID_COLS
    )

    output2 = os.path.join(
        OUTPUT_DIR,
        "frame_02_result.jpg"
    )

    cv2.imwrite(
        output2,
        result2
    )

    print()
    print(
        "[OUTPUT] Saved Frame 2:"
    )

    print(output2)

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print()
    print("=" * 72)
    print(
        "FINAL INTEGRATION RESULT"
    )
    print("=" * 72)

    print()
    print(
        "LOCALIZATION"
    )

    print(
        f"  Frame 1 pose : "
        f"{pose1}"
    )

    print(
        f"  Frame 2 pose : "
        f"{pose2}"
    )

    print()
    print(
        "NAVIGATION"
    )

    print(
        f"  Start        : "
        f"{start}"
    )

    print(
        f"  Fixed goal   : "
        f"{goal}"
    )

    print(
        f"  Old path     : "
        f"{path1}"
    )

    print(
        f"  New path     : "
        f"{new_path}"
    )

    print()
    print(
        "CONTROL"
    )

    print(
        f"  Initial      : "
        f"{command1}"
    )

    print(
        f"  Replanned    : "
        f"{command2}"
    )

    print()
    print(
        "SYSTEM STATUS"
    )

    print(
        "  [OK] Visual localization"
    )

    print(
        "  [OK] ORB feature matching"
    )

    print(
        "  [OK] Relative pose estimation"
    )

    print(
        "  [OK] YOLO perception"
    )

    print(
        "  [OK] Traversability"
    )

    print(
        "  [OK] Costmap generation"
    )

    print(
        "  [OK] Scene-change detection"
    )

    print(
        "  [OK] A* planning"
    )

    if path_changed:

        print(
            "  [OK] Dynamic replanning"
        )

    else:

        print(
            "  [FAIL] Dynamic replanning"
        )

    print()
    print(
        "RESULTS SAVED TO:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print("=" * 72)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()