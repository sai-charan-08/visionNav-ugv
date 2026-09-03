"""
utils/logger.py — Navigation Cycle Logger
==========================================
Appends one CSV row per navigation cycle containing:

    timestamp, n_obstacles, left_status, center_status, right_status,
    decision, inference_ms, obstacle_summary

Usage:
    logger = CycleLogger()
    logger.log(obstacles, free_space, decision, inference_ms)
"""

import sys
import os
import csv
import datetime
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from perception.obstacle_detector import ObstacleInfo


# ======================================================================
# LOGGER CLASS
# ======================================================================

class CycleLogger:
    """
    Logs each navigation cycle to a CSV file.

    If ENABLE_LOGGING in config.py is False, all calls are no-ops.
    The log file is created (with header) on first write if it doesn't exist.
    """

    FIELDS = [
        "timestamp",
        "n_obstacles",
        "left_status",
        "center_status",
        "right_status",
        "decision",
        "inference_ms",
        "obstacle_summary",
    ]

    def __init__(self, log_dir: str = None, filename: str = None):
        self.enabled  = config.ENABLE_LOGGING
        self.log_dir  = log_dir  or config.LOG_DIR
        self.filename = filename or config.LOG_FILENAME
        self.log_path = os.path.join(self.log_dir, self.filename)

        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            self._ensure_header()

    # ------------------------------------------------------------------
    def _ensure_header(self):
        """Write CSV header row if the file doesn't already exist."""
        if not os.path.isfile(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()

    # ------------------------------------------------------------------
    def log(self,
            obstacles:    List[ObstacleInfo],
            free_space:   Dict[str, dict],
            decision:     str,
            inference_ms: float = 0.0) -> None:
        """
        Append one row to the CSV log file.

        Parameters
        ----------
        obstacles    : output of ObstacleDetector.analyze()
        free_space   : output of FreeSpaceEstimator.estimate()
        decision     : navigation decision string
        inference_ms : YOLO inference time in ms
        """
        if not self.enabled:
            return

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Build obstacle summary: "motorcycle(0.65,CENTER) bicycle(0.72,LEFT)"
        parts = []
        for obs in obstacles:
            zone_str = "+".join(obs.zones)
            parts.append(f"{obs.class_name}({obs.confidence:.2f},{zone_str})")
        obstacle_summary = " ".join(parts) if parts else "none"

        row = {
            "timestamp":        now,
            "n_obstacles":      len(obstacles),
            "left_status":      free_space.get("LEFT",   {}).get("status", "UNKNOWN"),
            "center_status":    free_space.get("CENTER", {}).get("status", "UNKNOWN"),
            "right_status":     free_space.get("RIGHT",  {}).get("status", "UNKNOWN"),
            "decision":         decision,
            "inference_ms":     f"{inference_ms:.1f}",
            "obstacle_summary": obstacle_summary,
        }

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)

    # ------------------------------------------------------------------
    def print_last_n(self, n: int = 5) -> None:
        """Print the last N log rows to stdout (useful for debugging)."""
        if not os.path.isfile(self.log_path):
            print("[Logger] Log file does not exist yet.")
            return

        with open(self.log_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            print("[Logger] Log file is empty.")
            return

        print(f"[Logger] Last {min(n, len(rows))} log entries:")
        for row in rows[-n:]:
            print(
                f"  {row['timestamp']}  |  "
                f"objs={row['n_obstacles']}  |  "
                f"L={row['left_status']}  C={row['center_status']}  R={row['right_status']}  |  "
                f"decision={row['decision']}  |  "
                f"{row['inference_ms']}ms  |  {row['obstacle_summary']}"
            )


# ======================================================================
# STANDALONE TEST
# Run: python -m utils.logger
# ======================================================================
if __name__ == "__main__":
    import cv2
    from perception.yolo_detector    import YOLODetector
    from perception.obstacle_detector import ObstacleDetector
    from perception.free_space        import FreeSpaceEstimator
    from navigation.decision_engine   import DecisionEngine

    print("=" * 60)
    print("  CycleLogger — standalone test")
    print("=" * 60)

    try:
        detector     = YOLODetector()
        obs_detector = ObstacleDetector()
        fs_estimator = FreeSpaceEstimator()
        engine       = DecisionEngine()
        logger       = CycleLogger()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not os.path.isfile(config.IMAGE_PATH):
        print(f"[ERROR] Test image not found: {config.IMAGE_PATH}")
        sys.exit(1)

    frame = cv2.imread(config.IMAGE_PATH)
    h, w  = frame.shape[:2]

    yolo_result, ms = detector.detect(frame)
    obstacles        = obs_detector.analyze(yolo_result, w, h)
    free_space       = fs_estimator.estimate(obstacles)
    decision         = engine.decide(free_space, obstacles)

    # Log 3 simulated cycles
    for i in range(3):
        logger.log(obstacles, free_space, decision, ms)

    print(f"Log file: {logger.log_path}")
    print()
    logger.print_last_n(3)
    print()
    print("CycleLogger test PASSED.")
    print("=" * 60)
