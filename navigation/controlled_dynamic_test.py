import os
import cv2
import numpy as np

from config import MODEL_PATH

from perception.yolo_detector import YOLODetector
from perception.obstacle_detector import ObstacleDetector
from perception.traversability import TraversabilityDetector
from perception.scene_change import SceneChangeDetector

from navigation.costmap import Costmap
from navigation.path_planner import PathPlanner

from visualization.navigation_visualizer import NavigationVisualizer


# ============================================================
# CONFIGURATION
# ============================================================

GRID_ROWS = 12
GRID_COLS = 16

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_DIR = os.path.join(
    BASE_DIR,
    "input",
    "dynamic_sequence"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "runs",
    "controlled_dynamic"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DYNAMIC OBSTACLE SETTINGS
# ============================================================

# For the coarse 12x16 grid, we represent a newly detected
# obstacle using a small footprint around its center.
#
# This prevents one large image-space bounding box from
# swallowing an entire navigable region.

DYNAMIC_RADIUS_ROWS = 0
DYNAMIC_RADIUS_COLS = 1

# No extra inflation for the dynamic obstacle.
DYNAMIC_OBSTACLE_INFLATION = 0


# ============================================================
# FIND NEAREST FREE CELL
# ============================================================

def find_nearest_free_cell(
    costmap,
    position,
    max_radius=4
):

    row, col = position

    if (
        0 <= row < GRID_ROWS
        and
        0 <= col < GRID_COLS
        and
        costmap.is_free(row, col)
    ):
        return position

    candidates = []

    for radius in range(
        1,
        max_radius + 1
    ):

        for dr in range(
            -radius,
            radius + 1
        ):

            for dc in range(
                -radius,
                radius + 1
            ):

                r = row + dr
                c = col + dc

                if not (
                    0 <= r < GRID_ROWS
                    and
                    0 <= c < GRID_COLS
                ):
                    continue

                if costmap.is_free(
                    r,
                    c
                ):

                    distance = (
                        abs(dr) +
                        abs(dc)
                    )

                    candidates.append(
                        (
                            distance,
                            r,
                            c
                        )
                    )

    if not candidates:
        return None

    candidates.sort()

    return (
        candidates[0][1],
        candidates[0][2]
    )


# ============================================================
# PATH VALIDATION
# ============================================================

def path_is_valid(
    path,
    costmap
):

    if not path:
        return False

    for row, col in path:

        if not costmap.is_free(
            row,
            col
        ):
            return False

    return True


# ============================================================
# NAVIGATION COMMAND
# ============================================================

def get_command(path):

    if not path or len(path) < 2:
        return "STOP"

    current = path[0]
    next_cell = path[1]

    dr = next_cell[0] - current[0]
    dc = next_cell[1] - current[1]

    if dr < 0 and dc < 0:
        return "FORWARD-LEFT"

    if dr < 0 and dc > 0:
        return "FORWARD-RIGHT"

    if dr < 0:
        return "FORWARD"

    if dc < 0:
        return "LEFT"

    if dc > 0:
        return "RIGHT"

    return "STOP"


# ============================================================
# PRINT COSTMAP
# ============================================================

def print_costmap(
    costmap,
    start=None,
    goal=None
):

    print("\nCostmap:")

    for r in range(
        GRID_ROWS
    ):

        row_text = []

        for c in range(
            GRID_COLS
        ):

            if (
                start is not None
                and
                (r, c) == start
            ):

                row_text.append(" S ")

            elif (
                goal is not None
                and
                (r, c) == goal
            ):

                row_text.append(" G ")

            elif costmap.grid[r, c] >= 100:

                row_text.append("###")

            elif costmap.grid[r, c] >= 60:

                row_text.append(" 6 ")

            elif costmap.grid[r, c] >= 30:

                row_text.append(" 3 ")

            else:

                row_text.append(" . ")

        print(
            "".join(row_text)
        )


# ============================================================
# INITIAL GOAL
# ============================================================

def find_initial_goal(
    costmap,
    start
):

    planner = PathPlanner(
        costmap.grid
    )

    center_col = GRID_COLS // 2

    candidates = []

    preferred_cols = [
        center_col,
        center_col - 1,
        center_col + 1,
        center_col - 2,
        center_col + 2,
        center_col - 3,
        center_col + 3
    ]

    for row in range(
        1,
        start[0]
    ):

        for col in preferred_cols:

            if not (
                0 <= col < GRID_COLS
            ):
                continue

            if not costmap.is_free(
                row,
                col
            ):
                continue

            distance = (
                abs(start[0] - row)
                +
                abs(start[1] - col)
            )

            center_distance = abs(
                col - center_col
            )

            candidates.append(
                (
                    -distance,
                    center_distance,
                    row,
                    col
                )
            )

    candidates.sort()

    for (
        _,
        _,
        row,
        col
    ) in candidates:

        goal = (
            row,
            col
        )

        path = planner.plan(
            start,
            goal
        )

        if path is not None:

            return (
                goal,
                path
            )

    return (
        None,
        None
    )


# ============================================================
# CONVERT IMAGE BOX TO GRID CENTER
# ============================================================

def image_box_to_grid_center(
    box,
    image_width,
    image_height
):

    x1, y1, x2, y2 = box

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    col = int(
        center_x /
        image_width *
        GRID_COLS
    )

    row = int(
        center_y /
        image_height *
        GRID_ROWS
    )

    col = max(
        0,
        min(
            GRID_COLS - 1,
            col
        )
    )

    row = max(
        0,
        min(
            GRID_ROWS - 1,
            row
        )
    )

    return (
        row,
        col
    )


# ============================================================
# ADD SMALL DYNAMIC OBSTACLE
# ============================================================

def add_dynamic_obstacles(
    costmap,
    boxes,
    image_width,
    image_height
):

    processed_boxes = []

    if not boxes:
        return processed_boxes

    print(
        f"Adding {len(boxes)} "
        f"dynamic obstacle(s)."
    )

    for box in boxes:

        print(
            f"  Scene-change box: "
            f"{box}"
        )

        center_row, center_col = (
            image_box_to_grid_center(
                box,
                image_width,
                image_height
            )
        )

        print(
            f"  Projected grid center: "
            f"({center_row}, {center_col})"
        )

        # ----------------------------------------------------
        # SMALL GRID FOOTPRINT
        # ----------------------------------------------------

        for dr in range(
            -DYNAMIC_RADIUS_ROWS,
            DYNAMIC_RADIUS_ROWS + 1
        ):

            for dc in range(
                -DYNAMIC_RADIUS_COLS,
                DYNAMIC_RADIUS_COLS + 1
            ):

                row = (
                    center_row +
                    dr
                )

                col = (
                    center_col +
                    dc
                )

                if not (
                    0 <= row < GRID_ROWS
                    and
                    0 <= col < GRID_COLS
                ):
                    continue

                costmap.grid[
                    row,
                    col
                ] = 100

        processed_boxes.append(
            box
        )

        print(
            "  Dynamic grid cells: "
            f"rows "
            f"{max(0, center_row - DYNAMIC_RADIUS_ROWS)}"
            f"-"
            f"{min(GRID_ROWS - 1, center_row + DYNAMIC_RADIUS_ROWS)}, "
            f"cols "
            f"{max(0, center_col - DYNAMIC_RADIUS_COLS)}"
            f"-"
            f"{min(GRID_COLS - 1, center_col + DYNAMIC_RADIUS_COLS)}"
        )

    return processed_boxes


# ============================================================
# PROTECT GOAL
# ============================================================

def protect_goal_region(
    costmap,
    baseline_costmap,
    goal
):

    goal_row, goal_col = goal

    baseline_value = (
        baseline_costmap.grid[
            goal_row,
            goal_col
        ]
    )

    costmap.grid[
        goal_row,
        goal_col
    ] = baseline_value

    print(
        f"Protected destination "
        f"cell {goal} "
        f"(baseline cost="
        f"{baseline_value})."
    )


# ============================================================
# DRAW DYNAMIC BOX
# ============================================================

def draw_dynamic_boxes(
    image,
    original_boxes
):

    for box in original_boxes:

        x1, y1, x2, y2 = box

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            image,
            "NEW OBSTACLE",
            (
                x1,
                max(
                    25,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return image


# ============================================================
# FRAME 1
# ============================================================

def process_initial_frame(
    image_path,
    yolo,
    obstacle_detector,
    traversability_detector,
    visualizer,
    current_position
):

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        raise RuntimeError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    height, width = (
        frame.shape[:2]
    )

    print("\n" + "=" * 70)

    print(
        f"FRAME: "
        f"{os.path.basename(image_path)}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    yolo_result, inference_ms = (
        yolo.detect(frame)
    )

    obstacles = (
        obstacle_detector.analyze(
            yolo_result,
            width,
            height
        )
    )

    print(
        f"YOLO inference: "
        f"{inference_ms:.1f} ms"
    )

    print(
        f"YOLO obstacles detected: "
        f"{len(obstacles)}"
    )

    # --------------------------------------------------------
    # TRAVERSABILITY
    # --------------------------------------------------------

    traversability_mask = (
        traversability_detector.detect(
            frame
        )
    )

    traversable_ratio = (
        float(
            (
                traversability_mask > 0
            ).mean()
        )
        * 100
    )

    print(
        f"Traversable area: "
        f"{traversable_ratio:.2f}%"
    )

    # --------------------------------------------------------
    # COSTMAP
    # --------------------------------------------------------

    costmap = Costmap(
        rows=GRID_ROWS,
        cols=GRID_COLS,
        obstacle_inflation=1
    )

    boxes = [
        obstacle.bbox
        for obstacle in obstacles
    ]

    costmap.build(
        traversability_mask,
        boxes,
        width,
        height
    )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    requested_start = (
        current_position
    )

    start = find_nearest_free_cell(
        costmap,
        requested_start
    )

    if start is None:

        print(
            "ERROR: No free start cell."
        )

        return (
            None,
            None,
            None,
            None,
            current_position
        )

    if start != requested_start:

        print(
            f"Requested start "
            f"{requested_start} "
            f"is blocked."
        )

        print(
            f"Using free start: "
            f"{start}"
        )

        current_position = start

    else:

        print(
            f"Start: {start}"
        )

    # --------------------------------------------------------
    # GOAL
    # --------------------------------------------------------

    goal, path = (
        find_initial_goal(
            costmap,
            start
        )
    )

    if goal is None:

        print(
            "ERROR: Could not find "
            "reachable goal."
        )

        return (
            None,
            None,
            None,
            None,
            current_position
        )

    print(
        f"Fixed goal selected: "
        f"{goal}"
    )

    print(
        f"Initial A* path length: "
        f"{len(path)}"
    )

    print(
        f"Initial path:\n"
        f"{path}"
    )

    print_costmap(
        costmap,
        start,
        goal
    )

    # --------------------------------------------------------
    # COMMAND
    # --------------------------------------------------------

    command = get_command(
        path
    )

    print(
        f"Command: {command}"
    )

    # --------------------------------------------------------
    # SIMULATED MOVEMENT
    # --------------------------------------------------------

    if len(path) >= 2:

        next_position = path[1]

        print(
            f"Simulated movement: "
            f"{current_position}"
            f" -> "
            f"{next_position}"
        )

        current_position = (
            next_position
        )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    result_image = (
        visualizer.visualize(
            frame.copy(),
            obstacles,
            path,
            GRID_ROWS,
            GRID_COLS
        )
    )

    return (
        result_image,
        costmap,
        path,
        goal,
        current_position
    )


# ============================================================
# FRAME 2
# ============================================================

def process_dynamic_frame(
    image_path,
    previous_frame,
    baseline_costmap,
    existing_path,
    fixed_goal,
    current_position,
    yolo,
    obstacle_detector,
    traversability_detector,
    scene_change_detector,
    visualizer
):

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        raise RuntimeError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    height, width = (
        frame.shape[:2]
    )

    print("\n" + "=" * 70)

    print(
        f"FRAME: "
        f"{os.path.basename(image_path)}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    yolo_result, inference_ms = (
        yolo.detect(frame)
    )

    obstacles = (
        obstacle_detector.analyze(
            yolo_result,
            width,
            height
        )
    )

    print(
        f"YOLO inference: "
        f"{inference_ms:.1f} ms"
    )

    print(
        f"YOLO obstacles detected: "
        f"{len(obstacles)}"
    )

    # --------------------------------------------------------
    # TRAVERSABILITY
    # --------------------------------------------------------

    traversability_mask = (
        traversability_detector.detect(
            frame
        )
    )

    traversable_ratio = (
        float(
            (
                traversability_mask > 0
            ).mean()
        )
        * 100
    )

    print(
        f"Traversability "
        f"(reference only): "
        f"{traversable_ratio:.2f}%"
    )

    # --------------------------------------------------------
    # DYNAMIC COSTMAP
    # --------------------------------------------------------

    costmap = Costmap(
        rows=GRID_ROWS,
        cols=GRID_COLS,
        obstacle_inflation=(
            DYNAMIC_OBSTACLE_INFLATION
        )
    )

    costmap.grid = (
        baseline_costmap.grid.copy()
    )

    print(
        "\nUsing Frame 1 road map "
        "as dynamic baseline."
    )

    # --------------------------------------------------------
    # YOLO OBSTACLES
    # --------------------------------------------------------

    for obstacle in obstacles:

        x1, y1, x2, y2 = (
            obstacle.bbox
        )

        costmap.mark_obstacle(
            x1,
            y1,
            x2,
            y2,
            width,
            height
        )

    # --------------------------------------------------------
    # SCENE CHANGE
    # --------------------------------------------------------

    scene_boxes = (
        scene_change_detector.detect(
            previous_frame,
            frame
        )
    )

    print(
        f"New scene-change "
        f"obstacles: "
        f"{len(scene_boxes)}"
    )

    # --------------------------------------------------------
    # DYNAMIC OBSTACLE
    # --------------------------------------------------------

    dynamic_boxes = (
        add_dynamic_obstacles(
            costmap,
            scene_boxes,
            width,
            height
        )
    )

    # --------------------------------------------------------
    # PROTECT DESTINATION
    # --------------------------------------------------------

    protect_goal_region(
        costmap,
        baseline_costmap,
        fixed_goal
    )

    # --------------------------------------------------------
    # PRINT DYNAMIC COSTMAP
    # --------------------------------------------------------

    print_costmap(
        costmap,
        current_position,
        fixed_goal
    )

    # --------------------------------------------------------
    # CHECK OLD PATH
    # --------------------------------------------------------

    if path_is_valid(
        existing_path,
        costmap
    ):

        print(
            "\nExisting path: VALID"
        )

        print(
            "Action: CONTINUE"
        )

        path = existing_path

        command = get_command(
            path
        )

        print(
            f"Command: {command}"
        )

    else:

        print(
            "\nExisting path: BLOCKED"
        )

        print(
            "Action: DYNAMIC REPLANNING"
        )

        print(
            f"Old path:\n"
            f"{existing_path}"
        )

        print(
            f"Start: "
            f"{current_position}"
        )

        print(
            f"Same destination: "
            f"{fixed_goal}"
        )

        # ----------------------------------------------------
        # START
        # ----------------------------------------------------

        start = find_nearest_free_cell(
            costmap,
            current_position
        )

        if start is None:

            print(
                "ERROR: No free start cell."
            )

            return (
                None,
                None,
                current_position
            )

        if start != current_position:

            print(
                f"Current position "
                f"{current_position} "
                f"is blocked."
            )

            print(
                f"Using nearby free cell: "
                f"{start}"
            )

            current_position = start

        # ----------------------------------------------------
        # GOAL
        # ----------------------------------------------------

        if not costmap.is_free(
            fixed_goal[0],
            fixed_goal[1]
        ):

            print(
                "ERROR: Destination "
                "is blocked."
            )

            return (
                None,
                None,
                current_position
            )

        # ----------------------------------------------------
        # A*
        # ----------------------------------------------------

        planner = PathPlanner(
            costmap.grid
        )

        new_path = planner.plan(
            start,
            fixed_goal
        )

        if new_path is None:

            print(
                "ERROR: A* could not "
                "find a detour."
            )

            print(
                "\nThis means the Frame-1 "
                "road map does not contain "
                "a connected alternative "
                "around the detected obstacle."
            )

            print(
                "The costmap printed above "
                "shows exactly why."
            )

            return (
                None,
                None,
                current_position
            )

        path = new_path

        print(
            f"Replanned path length: "
            f"{len(path)}"
        )

        print(
            f"New path:\n"
            f"{path}"
        )

        # ----------------------------------------------------
        # PATH COMPARISON
        # ----------------------------------------------------

        if path != existing_path:

            print(
                "\n"
                "*** PATH CHANGED "
                "SUCCESSFULLY ***"
            )

        else:

            print(
                "\n"
                "WARNING: A* returned "
                "the same path."
            )

        # ----------------------------------------------------
        # COMMAND
        # ----------------------------------------------------

        command = get_command(
            path
        )

        print(
            f"New navigation command: "
            f"{command}"
        )

        # ----------------------------------------------------
        # SIMULATED MOVEMENT
        # ----------------------------------------------------

        if len(path) >= 2:

            next_position = path[1]

            print(
                f"Simulated movement: "
                f"{current_position}"
                f" -> "
                f"{next_position}"
            )

            current_position = (
                next_position
            )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    result_image = (
        visualizer.visualize(
            frame.copy(),
            obstacles,
            path,
            GRID_ROWS,
            GRID_COLS
        )
    )

    result_image = (
        draw_dynamic_boxes(
            result_image,
            scene_boxes
        )
    )

    return (
        result_image,
        path,
        current_position
    )


# ============================================================
# MAIN
# ============================================================

def main():

    frame_1_path = os.path.join(
        INPUT_DIR,
        "frame_01_clear.png"
    )

    frame_2_path = os.path.join(
        INPUT_DIR,
        "frame_02_blocked.png"
    )

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not os.path.exists(
        frame_1_path
    ):

        raise FileNotFoundError(
            f"Frame 1 not found:\n"
            f"{frame_1_path}"
        )

    if not os.path.exists(
        frame_2_path
    ):

        raise FileNotFoundError(
            f"Frame 2 not found:\n"
            f"{frame_2_path}"
        )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "CONTROLLED DYNAMIC "
        "OBSTACLE TEST"
    )

    print("=" * 70)

    print(
        "\nFrame 1 = Clear road"
    )

    print(
        "Frame 2 = Same road + new rock"
    )

    print(
        "\nThe destination remains unchanged."
    )

    print(
        "A* must find an alternative route "
        "if the old route becomes blocked."
    )

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    print(
        "\nInitializing VisionNav modules..."
    )

    yolo = YOLODetector(
        MODEL_PATH
    )

    obstacle_detector = (
        ObstacleDetector()
    )

    traversability_detector = (
        TraversabilityDetector()
    )

    scene_change_detector = (
        SceneChangeDetector()
    )

    visualizer = (
        NavigationVisualizer()
    )

    # --------------------------------------------------------
    # INITIAL POSITION
    # --------------------------------------------------------

    current_position = (
        GRID_ROWS - 1,
        GRID_COLS // 2
    )

    # --------------------------------------------------------
    # FRAME 1
    # --------------------------------------------------------

    print("\n\n")

    print("=" * 70)

    print(
        "PROCESSING FRAME 1 — "
        "CLEAR ROAD"
    )

    print("=" * 70)

    (
        result_1,
        baseline_costmap,
        current_path,
        fixed_goal,
        current_position
    ) = process_initial_frame(
        frame_1_path,
        yolo,
        obstacle_detector,
        traversability_detector,
        visualizer,
        current_position
    )

    if result_1 is None:

        print(
            "\nFrame 1 failed."
        )

        return

    output_1 = os.path.join(
        OUTPUT_DIR,
        "frame_01_result.jpg"
    )

    cv2.imwrite(
        output_1,
        result_1
    )

    print(
        "\nSaved visualization:"
    )

    print(
        output_1
    )

    # --------------------------------------------------------
    # PREVIOUS FRAME
    # --------------------------------------------------------

    previous_frame = cv2.imread(
        frame_1_path
    )

    # --------------------------------------------------------
    # FRAME 2
    # --------------------------------------------------------

    print("\n\n")

    print("=" * 70)

    print(
        "PROCESSING FRAME 2 — "
        "ROCK APPEARS"
    )

    print("=" * 70)

    (
        result_2,
        current_path,
        current_position
    ) = process_dynamic_frame(
        frame_2_path,
        previous_frame,
        baseline_costmap,
        current_path,
        fixed_goal,
        current_position,
        yolo,
        obstacle_detector,
        traversability_detector,
        scene_change_detector,
        visualizer
    )

    if result_2 is None:

        print(
            "\n"
            "Dynamic replanning test "
            "could not complete."
        )

        print(
            "Do NOT commit yet."
        )

        return

    output_2 = os.path.join(
        OUTPUT_DIR,
        "frame_02_result.jpg"
    )

    cv2.imwrite(
        output_2,
        result_2
    )

    print(
        "\nSaved visualization:"
    )

    print(
        output_2
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n\n")

    print("=" * 70)

    print(
        "CONTROLLED DYNAMIC "
        "TEST COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nFinal simulated position: "
        f"{current_position}"
    )

    print(
        f"Fixed destination: "
        f"{fixed_goal}"
    )

    print(
        "\nPipeline demonstrated:"
    )

    print(
        "1. Camera perception"
    )

    print(
        "2. Traversability estimation"
    )

    print(
        "3. Initial costmap creation"
    )

    print(
        "4. Initial A* path planning"
    )

    print(
        "5. Simulated UGV movement"
    )

    print(
        "6. Scene-change detection"
    )

    print(
        "7. Dynamic obstacle insertion"
    )

    print(
        "8. Existing-path validation"
    )

    print(
        "9. A* dynamic replanning"
    )

    print(
        "10. New navigation command"
    )


if __name__ == "__main__":
    main()