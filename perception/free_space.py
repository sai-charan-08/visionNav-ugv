"""
perception/free_space.py — Zone Free-Space Estimator
======================================================
Aggregates obstacle severities per zone and classifies each zone as:

    FREE             — no meaningful obstacle present
    PARTIALLY BLOCKED — low-severity obstacle present
    BLOCKED          — high-severity obstacle present

Thresholds are defined in config.py and are fully configurable.
"""

import sys
import os
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from perception.obstacle_detector import ObstacleInfo


# ======================================================================
# ZONE STATUS CONSTANTS
# ======================================================================
STATUS_FREE    = "FREE"
STATUS_PARTIAL = "PARTIALLY BLOCKED"
STATUS_BLOCKED = "BLOCKED"

ALL_ZONES = ["LEFT", "CENTER", "RIGHT"]


# ======================================================================
# MAIN CLASS
# ======================================================================

class FreeSpaceEstimator:
    """
    Computes the free-space status of each navigation zone based on
    the obstacles detected by ObstacleDetector.

    Severity is accumulated per zone (sum of obstacle severities).
    The resulting zone_severity is compared against two configurable
    thresholds to determine the zone status.

    Thresholds (from config.py):
        ZONE_BLOCKED_SEVERITY  — above this → BLOCKED
        ZONE_PARTIAL_SEVERITY  — above this → PARTIALLY BLOCKED
        below partial           → FREE
    """

    def __init__(self,
                 blocked_threshold: float = None,
                 partial_threshold: float = None):
        self.blocked_threshold = blocked_threshold if blocked_threshold is not None \
                                 else config.ZONE_BLOCKED_SEVERITY
        self.partial_threshold = partial_threshold if partial_threshold is not None \
                                 else config.ZONE_PARTIAL_SEVERITY

    # ------------------------------------------------------------------
    def estimate(self, obstacles: List[ObstacleInfo]) -> Dict[str, dict]:
        """
        Estimate free-space status for LEFT, CENTER, RIGHT zones.

        Parameters
        ----------
        obstacles : list of ObstacleInfo from ObstacleDetector

        Returns
        -------
        dict with structure:
            {
                "LEFT":   {"status": "FREE",    "severity": 0.0},
                "CENTER": {"status": "BLOCKED", "severity": 0.72},
                "RIGHT":  {"status": "FREE",    "severity": 0.0},
            }
        """
        # Accumulate severity per zone
        zone_severity: Dict[str, float] = {z: 0.0 for z in ALL_ZONES}

        for obs in obstacles:
            for zone in obs.zones:
                if zone in zone_severity:
                    # Use the zone-specific overlap contribution
                    overlap = obs.zone_overlaps.get(zone, 0.0)
                    zone_severity[zone] += obs.confidence * overlap

        # Classify each zone
        free_space: Dict[str, dict] = {}
        for zone in ALL_ZONES:
            sev = round(zone_severity[zone], 4)
            if sev >= self.blocked_threshold:
                status = STATUS_BLOCKED
            elif sev >= self.partial_threshold:
                status = STATUS_PARTIAL
            else:
                status = STATUS_FREE

            free_space[zone] = {
                "status":   status,
                "severity": sev,
            }

        return free_space

    # ------------------------------------------------------------------
    @staticmethod
    def summary(free_space: Dict[str, dict]) -> str:
        """Return a compact one-line summary string for logging/display."""
        parts = []
        for zone in ALL_ZONES:
            info   = free_space[zone]
            status = info["status"]
            sev    = info["severity"]
            parts.append(f"{zone}={status}({sev:.2f})")
        return "  |  ".join(parts)


# ======================================================================
# STANDALONE TEST
# Run: python -m perception.free_space
# ======================================================================
if __name__ == "__main__":
    import cv2
    from perception.yolo_detector   import YOLODetector
    from perception.obstacle_detector import ObstacleDetector

    print("=" * 60)
    print("  FreeSpaceEstimator — standalone test")
    print("=" * 60)

    # Load
    try:
        detector     = YOLODetector()
        obs_detector = ObstacleDetector()
        fs_estimator = FreeSpaceEstimator()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not os.path.isfile(config.IMAGE_PATH):
        print(f"[ERROR] Test image not found: {config.IMAGE_PATH}")
        sys.exit(1)

    frame        = cv2.imread(config.IMAGE_PATH)
    h, w         = frame.shape[:2]

    # Inference
    yolo_result, ms = detector.detect(frame)
    obstacles       = obs_detector.analyze(yolo_result, w, h)
    free_space      = fs_estimator.estimate(obstacles)

    print(f"\nInference time : {ms:.1f} ms")
    print(f"Obstacles found: {len(obstacles)}")
    print()

    # Display zone statuses
    print(f"  {'Zone':<8}  {'Status':<18}  {'Severity':>8}")
    print(f"  {'-'*8}  {'-'*18}  {'-'*8}")
    for zone in ALL_ZONES:
        info = free_space[zone]
        print(f"  {zone:<8}  {info['status']:<18}  {info['severity']:>8.4f}")

    print()
    print("Summary:", FreeSpaceEstimator.summary(free_space))
    print()
    print("FreeSpaceEstimator test PASSED.")
    print("=" * 60)
