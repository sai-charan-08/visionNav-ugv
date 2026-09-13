import os
import time
import cv2

import config

from perception.yolo_detector import YOLODetector
from perception.obstacle_detector import ObstacleDetector
from perception.free_space import FreeSpaceEstimator

from navigation.decision_engine import DecisionEngine

from control.motor_controller import MotorController

from visualization.dashboard import Dashboard

from ros_bridge_client import ROSBridgeClient


def initialise_pipeline():

    print("=" * 60)
    print("VisionNav UGV - Initialising Pipeline")
    print("=" * 60)

    components = {}

    # ---------------------------------------------------------
    # PERCEPTION
    # ---------------------------------------------------------

    components["detector"] = YOLODetector(
        model_path=config.MODEL_PATH,
        confidence=config.YOLO_CONFIDENCE
    )

    components["obstacle_detector"] = ObstacleDetector()

    components["free_space"] = FreeSpaceEstimator()

    # ---------------------------------------------------------
    # DECISION ENGINE
    # ---------------------------------------------------------

    components["decision_engine"] = DecisionEngine()

    # ---------------------------------------------------------
    # MOTOR CONTROL
    # ---------------------------------------------------------

    components["motor_ctrl"] = MotorController()

    # ---------------------------------------------------------
    # ROS 2 / GAZEBO BRIDGE
    # ---------------------------------------------------------

    components["ros_client"] = ROSBridgeClient()

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------

    components["dashboard"] = Dashboard()

    print("[OK] Pipeline initialised")

    return components


def process_frame(frame, components):

    detector = components["detector"]
    obstacle_detector = components["obstacle_detector"]
    free_space_estimator = components["free_space"]
    decision_engine = components["decision_engine"]
    motor_ctrl = components["motor_ctrl"]
    ros_client = components["ros_client"]

    # ---------------------------------------------------------
    # 1. YOLO PERCEPTION
    # ---------------------------------------------------------

    yolo_result, inference_ms = detector.detect(frame)

    # ---------------------------------------------------------
    # 2. OBSTACLE ANALYSIS
    # ---------------------------------------------------------

    frame_height, frame_width = frame.shape[:2]

    obstacles = obstacle_detector.analyze(
        yolo_result,
        frame_width,
        frame_height
    )

    # ---------------------------------------------------------
    # 3. FREE SPACE ANALYSIS
    # ---------------------------------------------------------
    # IMPORTANT:
    # FreeSpaceEstimator.estimate() expects the obstacle list,
    # not the image frame.

    free_space = free_space_estimator.estimate(
        obstacles
    )

    # ---------------------------------------------------------
    # 4. NAVIGATION DECISION
    # ---------------------------------------------------------

    decision = decision_engine.decide(
        free_space,
        obstacles
    )

    # ---------------------------------------------------------
    # 5. LOCAL MOTOR COMMAND
    # ---------------------------------------------------------

    motor_ctrl.execute(
        decision
    )

    # ---------------------------------------------------------
    # 6. SEND DECISION TO ROS 2 / GAZEBO
    # ---------------------------------------------------------

    ros_client.send_decision(
        decision
    )

    # ---------------------------------------------------------
    # 7. PRINT RESULT
    # ---------------------------------------------------------

    print()
    print("-" * 60)
    print("VisionNav Decision")
    print("-" * 60)

    print(
        f"YOLO inference : "
        f"{inference_ms:.2f} ms"
    )

    print(
        f"Obstacles      : "
        f"{len(obstacles)}"
    )

    print(
        f"Free space     : "
        f"{FreeSpaceEstimator.summary(free_space)}"
    )

    print(
        f"Decision       : "
        f"{decision}"
    )

    print("-" * 60)

    return {
        "yolo_result": yolo_result,
        "obstacles": obstacles,
        "free_space": free_space,
        "decision": decision,
        "inference_ms": inference_ms
    }


def run_image_mode(components):

    print()
    print("=" * 60)
    print("VISIONNAV IMAGE MODE")
    print("=" * 60)

    image_path = config.IMAGE_PATH

    if not os.path.exists(image_path):

        print(
            f"[ERROR] Image not found: "
            f"{image_path}"
        )

        return

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        print(
            f"[ERROR] Could not read image: "
            f"{image_path}"
        )

        return

    try:

        # -----------------------------------------------------
        # PROCESS IMAGE
        # -----------------------------------------------------

        result = process_frame(
            frame,
            components
        )

        decision = result["decision"]

        # -----------------------------------------------------
        # GAZEBO DEMONSTRATION
        # -----------------------------------------------------

        if decision != config.NAV_STOP:

            print()
            print(
                "[Gazebo] Executing command "
                "for 1 second..."
            )

            time.sleep(
                1.0
            )

            print(
                "[Gazebo] Sending STOP"
            )

            components[
                "ros_client"
            ].stop()

            components[
                "motor_ctrl"
            ].stop()

        else:

            print(
                "[Gazebo] Decision is STOP"
            )

    except KeyboardInterrupt:

        print()
        print(
            "[INFO] Interrupted by user."
        )

    finally:

        # -----------------------------------------------------
        # SAFETY STOP
        # -----------------------------------------------------

        print(
            "[Gazebo] Final STOP"
        )

        try:

            components[
                "ros_client"
            ].stop()

        except Exception:
            pass

        try:

            components[
                "motor_ctrl"
            ].stop()

        except Exception:
            pass

    # ---------------------------------------------------------
    # SAVE OUTPUT
    # ---------------------------------------------------------

    output_path = os.path.join(
        config.OUTPUT_DIR,
        config.OUTPUT_FILENAME
    )

    os.makedirs(
        config.OUTPUT_DIR,
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        frame
    )

    print()
    print(
        "[OK] Result saved to:"
    )

    print(
        output_path
    )

    # ---------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------

    if config.SHOW_WINDOW:

        cv2.imshow(
            "VisionNav UGV",
            frame
        )

        cv2.waitKey(
            0
        )

        cv2.destroyAllWindows()


def run_camera_mode(components):

    print()
    print("=" * 60)
    print("VISIONNAV CAMERA MODE")
    print("=" * 60)

    camera = cv2.VideoCapture(
        config.CAMERA_INDEX
    )

    if not camera.isOpened():

        print(
            "[ERROR] Could not open camera."
        )

        return

    print(
        "[OK] Camera started"
    )

    print(
        "Press Q to stop."
    )

    try:

        while True:

            ret, frame = camera.read()

            if not ret:

                print(
                    "[ERROR] Failed to read camera frame."
                )

                break

            process_frame(
                frame,
                components
            )

            cv2.imshow(
                "VisionNav UGV",
                frame
            )

            key = cv2.waitKey(
                1
            ) & 0xFF

            if key == ord("q"):

                print(
                    "[INFO] Q pressed."
                )

                break

    except KeyboardInterrupt:

        print()
        print(
            "[INFO] Camera mode interrupted."
        )

    finally:

        # -----------------------------------------------------
        # ALWAYS STOP GAZEBO
        # -----------------------------------------------------

        print(
            "[Gazebo] Sending final STOP"
        )

        try:

            components[
                "ros_client"
            ].stop()

        except Exception:
            pass

        try:

            components[
                "motor_ctrl"
            ].stop()

        except Exception:
            pass

        camera.release()

        cv2.destroyAllWindows()


def main():

    print()
    print("=" * 60)
    print("VISIONNAV UGV")
    print("Perception → Decision → ROS 2 → Gazebo")
    print("=" * 60)
    print()

    components = initialise_pipeline()

    try:

        if config.INPUT_MODE == "image":

            run_image_mode(
                components
            )

        elif config.INPUT_MODE == "camera":

            run_camera_mode(
                components
            )

        else:

            print(
                f"[ERROR] Unknown INPUT_MODE: "
                f"{config.INPUT_MODE}"
            )

    except KeyboardInterrupt:

        print()
        print(
            "[INFO] Program interrupted."
        )

    finally:

        # -----------------------------------------------------
        # FINAL SAFETY STOP
        # -----------------------------------------------------

        print()
        print(
            "[Gazebo] Final safety STOP"
        )

        try:

            components[
                "ros_client"
            ].stop()

        except Exception:
            pass

        try:

            components[
                "motor_ctrl"
            ].stop()

        except Exception:
            pass

    print()
    print("=" * 60)
    print(
        "VisionNav execution completed"
    )
    print("=" * 60)


if __name__ == "__main__":

    main()