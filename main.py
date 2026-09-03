"""
main.py — VisionNav-UGV Entry Point
=====================================
Connects the full navigation pipeline:

    Camera / Image File
         ↓
    YOLODetector          (perception/yolo_detector.py)
         ↓
    ObstacleDetector      (perception/obstacle_detector.py)
         ↓
    FreeSpaceEstimator    (perception/free_space.py)
         ↓
    DecisionEngine        (navigation/decision_engine.py)
         ↓
    Dashboard             (visualization/dashboard.py)
         ↓
    CycleLogger           (utils/logger.py)
         ↓
    [STUBS]
    MotorController       (control/motor_controller.py)
    LocalPlanner          (navigation/local_planner.py)
    SLAM / PathPlanner    (localization/slam.py, navigation/path_planner.py)

INPUT_MODE in config.py controls whether the system runs on:
    "image"  — a single test image (default; no hardware needed)
    "camera" — a live webcam feed (press Q to quit)

Run:
    python main.py
"""

import sys
import os
import cv2

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# ------------------------------------------------------------------
# Perception
# ------------------------------------------------------------------
from perception.yolo_detector    import YOLODetector
from perception.obstacle_detector import ObstacleDetector
from perception.free_space        import FreeSpaceEstimator

# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------
from navigation.decision_engine import DecisionEngine
from navigation.local_planner   import LocalPlanner
from navigation.path_planner    import PathPlanner

# ------------------------------------------------------------------
# Localization (stub)
# ------------------------------------------------------------------
from localization.slam import SLAMModule

# ------------------------------------------------------------------
# Control (stub)
# ------------------------------------------------------------------
from control.motor_controller import MotorController

# ------------------------------------------------------------------
# Visualization + Logging
# ------------------------------------------------------------------
from visualization.dashboard import Dashboard
from utils.logger             import CycleLogger
from perception.free_space    import FreeSpaceEstimator


# ======================================================================
# PIPELINE INITIALISATION
# ======================================================================

def initialise_pipeline():
    """
    Load all modules and return them as a dict.
    Fails fast with a clear error if critical components are missing.
    """
    print("=" * 60)
    print("  VisionNav-UGV  —  Initialising Pipeline")
    print("=" * 60)
    print(f"  Input mode : {config.INPUT_MODE}")
    print(f"  Model      : {config.MODEL_PATH}")
    print()

    components = {}

    # --- Critical: YOLO detector ---
    try:
        components["detector"] = YOLODetector()
    except FileNotFoundError as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)

    # --- Perception ---
    components["obs_detector"] = ObstacleDetector()
    components["fs_estimator"] = FreeSpaceEstimator()

    # --- Navigation ---
    components["engine"] = DecisionEngine()

    # --- Stubs (non-critical — print warnings but don't abort) ---
    components["slam"]             = SLAMModule()
    components["local_planner"]    = LocalPlanner()
    components["path_planner"]     = PathPlanner()
    components["motor_controller"] = MotorController()

    # --- Visualization + Logger ---
    components["dashboard"] = Dashboard()
    components["logger"]    = CycleLogger()

    print()
    print("  Pipeline ready.")
    print("=" * 60)
    print()

    return components


# ======================================================================
# SINGLE FRAME PROCESSING
# ======================================================================

def process_frame(frame, components: dict) -> tuple:
    """
    Run one complete navigation cycle on a single BGR frame.

    Returns
    -------
    annotated    : annotated BGR frame
    decision     : navigation decision string
    inference_ms : YOLO inference time
    """
    detector     = components["detector"]
    obs_detector = components["obs_detector"]
    fs_estimator = components["fs_estimator"]
    engine       = components["engine"]
    motor_ctrl   = components["motor_controller"]
    dashboard    = components["dashboard"]
    logger       = components["logger"]

    h, w = frame.shape[:2]

    # 1. YOLO inference
    yolo_result, inference_ms = detector.detect(frame)

    # 2. Obstacle detection + filtering
    obstacles = obs_detector.analyze(yolo_result, w, h)

    # 3. Free space estimation
    free_space = fs_estimator.estimate(obstacles)

    # 4. Navigation decision
    decision = engine.decide(free_space, obstacles)

    # 5. [STUB] Motor command (logged only, no hardware)
    motor_ctrl.execute(decision)

    # 6. Visualization
    annotated = dashboard.render(frame, obstacles, free_space, decision, inference_ms)

    # 7. Logging
    logger.log(obstacles, free_space, decision, inference_ms)

    return annotated, decision, inference_ms


# ======================================================================
# IMAGE MODE
# ======================================================================

def run_image_mode(components: dict) -> None:
    """Process a single test image and save the annotated result."""
    print(f"[Image Mode] Loading: {config.IMAGE_PATH}")

    if not os.path.isfile(config.IMAGE_PATH):
        print(f"[ERROR] Image not found: {config.IMAGE_PATH}")
        print("        Place a JPEG/PNG image at that path and retry.")
        sys.exit(1)

    frame = cv2.imread(config.IMAGE_PATH)
    if frame is None:
        print(f"[ERROR] OpenCV could not read: {config.IMAGE_PATH}")
        sys.exit(1)

    annotated, decision, inference_ms = process_frame(frame, components)

    # Save result
    out_path = components["dashboard"].save(annotated)

    # Print summary
    print()
    print("=" * 60)
    print("  NAVIGATION RESULT")
    print("=" * 60)
    print(f"  Decision     : {decision}")
    print(f"  Inference    : {inference_ms:.1f} ms")
    print(f"  Output saved : {out_path}")
    print("=" * 60)
    print()

    # Display
    if config.SHOW_WINDOW:
        print("Press any key in the image window to close.")
        cv2.imshow(config.WINDOW_TITLE, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ======================================================================
# CAMERA MODE
# ======================================================================

def run_camera_mode(components: dict) -> None:
    """Run the full pipeline on a live webcam feed. Press Q to quit."""
    print(f"[Camera Mode] Opening webcam index {config.CAMERA_INDEX} ...")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (index {config.CAMERA_INDEX}).")
        print("        Check that a webcam is connected and not in use.")
        sys.exit(1)

    print("[Camera Mode] Running — press Q in the window to quit.")
    print()

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("[Camera Mode] Frame read failed. Stopping.")
                break

            frame_count += 1

            # Process
            annotated, decision, inference_ms = process_frame(frame, components)

            # Console update every 10 frames to reduce spam
            if frame_count % 10 == 0:
                print(f"  Frame {frame_count:>5}  |  {decision:<16}  |  {inference_ms:.0f}ms")

            # Display
            if config.SHOW_WINDOW:
                cv2.imshow(config.WINDOW_TITLE, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    print("\n[Camera Mode] Q pressed — stopping.")
                    break

                # Also detect window close button
                if cv2.getWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[Camera Mode] Window closed — stopping.")
                    break

    except KeyboardInterrupt:
        print("\n[Camera Mode] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[Camera Mode] Total frames processed: {frame_count}")


# ======================================================================
# ENTRY POINT
# ======================================================================

def main():
    # Initialise all pipeline components
    components = initialise_pipeline()

    # Dispatch to the correct mode
    mode = config.INPUT_MODE.strip().lower()

    if mode == "image":
        run_image_mode(components)

    elif mode == "camera":
        run_camera_mode(components)

    else:
        print(f"[ERROR] Unknown INPUT_MODE: '{config.INPUT_MODE}'")
        print("        Set INPUT_MODE to 'image' or 'camera' in config.py")
        sys.exit(1)


if __name__ == "__main__":
    main()