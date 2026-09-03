"""
navigation/decision_engine.py — Navigation Decision Engine
===========================================================
Converts free-space zone statuses + obstacle data into a single
navigation command: MOVE FORWARD / TURN LEFT / TURN RIGHT / STOP.

Design principles (replacing the original rigid if/elif):
  1. Zone scoring — each zone is given a weighted "passability" score
  2. Direction preference — prefer the direction with the highest score
  3. Severity-aware — uses obstacle severity, not just binary blocked flags
  4. Large-obstacle penalty — large obstacles increase the cost of a zone
  5. All-blocked fallback — STOP only when truly necessary
"""

import sys
import os
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from perception.obstacle_detector import ObstacleInfo
from perception.free_space import (
    FreeSpaceEstimator, STATUS_FREE, STATUS_PARTIAL, STATUS_BLOCKED, ALL_ZONES
)


# ======================================================================
# DECISION ENGINE
# ======================================================================

class DecisionEngine:
    """
    Produces a navigation decision based on free-space zone statuses
    and the list of detected obstacles.

    Algorithm
    ---------
    1.  Compute a passability score for each zone (higher = safer to go).
    2.  If CENTER is FREE or only PARTIALLY BLOCKED → MOVE FORWARD.
    3.  Otherwise compare LEFT and RIGHT scores and pick the better one.
    4.  If all zones are BLOCKED → STOP.

    Passability score per zone:
        base = 1.0  (fully free)
        - severity (accumulated from obstacles)
        - large_obstacle_penalty for every large obstacle in the zone
        Clamped to [0.0, 1.0]
    """

    # Penalty deducted for each large obstacle in a zone
    LARGE_OBSTACLE_PENALTY = 0.20

    # Severity below which CENTER is treated as "good enough to go forward"
    CENTER_PARTIAL_FORWARD_THRESHOLD = 0.25

    def __init__(self):
        pass  # all config comes from config.py constants

    # ------------------------------------------------------------------
    def _zone_passability(self,
                          zone: str,
                          free_space: Dict[str, dict],
                          obstacles: List[ObstacleInfo]) -> float:
        """
        Compute a [0, 1] passability score for a zone.
        Higher = safer / more passable.
        """
        info     = free_space[zone]
        severity = info["severity"]

        # Count large obstacles in this zone
        large_count = sum(
            1 for obs in obstacles
            if zone in obs.zones and obs.is_large
        )

        score = 1.0 - severity - (large_count * self.LARGE_OBSTACLE_PENALTY)
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    def decide(self,
               free_space: Dict[str, dict],
               obstacles: List[ObstacleInfo]) -> str:
        """
        Make a navigation decision.

        Parameters
        ----------
        free_space : output of FreeSpaceEstimator.estimate()
        obstacles  : output of ObstacleDetector.analyze()

        Returns
        -------
        str — one of config.NAV_FORWARD / NAV_LEFT / NAV_RIGHT / NAV_STOP
        """
        # Compute passability scores for all zones
        scores = {
            zone: self._zone_passability(zone, free_space, obstacles)
            for zone in ALL_ZONES
        }

        center_status   = free_space["CENTER"]["status"]
        center_severity = free_space["CENTER"]["severity"]

        # ----------------------------------------------------------------
        # Rule 1: If CENTER is free, go forward.
        # ----------------------------------------------------------------
        if center_status == STATUS_FREE:
            return config.NAV_FORWARD

        # ----------------------------------------------------------------
        # Rule 2: CENTER is only PARTIALLY BLOCKED with low severity
        # → still prefer going forward (obstacle is small / distant).
        # ----------------------------------------------------------------
        if (center_status == STATUS_PARTIAL
                and center_severity <= self.CENTER_PARTIAL_FORWARD_THRESHOLD):
            return config.NAV_FORWARD

        # ----------------------------------------------------------------
        # Rule 3: CENTER is significantly blocked — pick best side.
        # ----------------------------------------------------------------
        left_score  = scores["LEFT"]
        right_score = scores["RIGHT"]

        left_status  = free_space["LEFT"]["status"]
        right_status = free_space["RIGHT"]["status"]

        # Both sides are blocked → STOP
        if left_status == STATUS_BLOCKED and right_status == STATUS_BLOCKED:
            return config.NAV_STOP

        # Both sides are passable → pick the one with the higher score
        if left_score > right_score:
            return config.NAV_LEFT
        elif right_score > left_score:
            return config.NAV_RIGHT
        else:
            # Tie → prefer left (conventional for most UGV protocols)
            return config.NAV_LEFT

    # ------------------------------------------------------------------
    @staticmethod
    def decision_color(decision: str) -> tuple:
        """Return a BGR color for the given navigation decision (for overlay)."""
        if decision == config.NAV_FORWARD:
            return config.NAV_COLOR_FORWARD
        elif decision == config.NAV_STOP:
            return config.NAV_COLOR_STOP
        else:
            return config.NAV_COLOR_TURN


# ======================================================================
# STANDALONE TEST
# Run: python -m navigation.decision_engine
# ======================================================================
if __name__ == "__main__":
    import cv2
    from perception.yolo_detector    import YOLODetector
    from perception.obstacle_detector import ObstacleDetector
    from perception.free_space        import FreeSpaceEstimator

    print("=" * 60)
    print("  DecisionEngine — standalone test")
    print("=" * 60)

    try:
        detector     = YOLODetector()
        obs_detector = ObstacleDetector()
        fs_estimator = FreeSpaceEstimator()
        engine       = DecisionEngine()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not os.path.isfile(config.IMAGE_PATH):
        print(f"[ERROR] Test image not found: {config.IMAGE_PATH}")
        sys.exit(1)

    frame = cv2.imread(config.IMAGE_PATH)
    h, w  = frame.shape[:2]

    yolo_result, ms = detector.detect(frame)
    obstacles       = obs_detector.analyze(yolo_result, w, h)
    free_space      = fs_estimator.estimate(obstacles)
    decision        = engine.decide(free_space, obstacles)

    print(f"\nInference time : {ms:.1f} ms")
    print(f"Obstacles      : {len(obstacles)}")
    print()
    print(f"  {FreeSpaceEstimator.summary(free_space)}")
    print()

    for zone in ALL_ZONES:
        score = engine._zone_passability(zone, free_space, obstacles)
        print(f"  Passability [{zone:<6}] : {score:.4f}")

    print()
    print(f">>> NAVIGATION DECISION: {decision}")
    print()
    print("DecisionEngine test PASSED.")
    print("=" * 60)
