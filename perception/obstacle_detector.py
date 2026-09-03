"""
perception/obstacle_detector.py — Obstacle Filtering and Zone Analysis
=======================================================================
Takes raw YOLO detections and produces structured ObstacleInfo objects.

Key improvements over the original yolo_test.py logic:
  1. Class allowlist — only obstacle-relevant classes are processed
  2. Zone assignment by bounding-box OVERLAP (not just center point)
     so a large object spanning two zones is recorded in both
  3. Severity score = confidence × zone_overlap_fraction
     giving a continuous measure instead of a binary blocked flag
  4. Configurable thresholds from config.py
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ======================================================================
# DATA STRUCTURES
# ======================================================================

@dataclass
class ObstacleInfo:
    """
    All information about a single detected obstacle.

    Fields
    ------
    class_name      : YOLO class label (e.g. "motorcycle")
    confidence      : raw YOLO detection confidence [0, 1]
    bbox            : (x1, y1, x2, y2) pixel coordinates
    center_x        : horizontal center of bounding box (pixels)
    center_y        : vertical center of bounding box (pixels)
    bbox_area_frac  : bbox area as fraction of total frame area
    zones           : list of zone names this obstacle overlaps
                      e.g. ["LEFT", "CENTER"]
    zone_overlaps   : dict zone_name → overlap fraction [0, 1]
                      fraction of the zone's width covered by the bbox
    severity        : max(conf × zone_overlap) across all zones
                      used downstream for zone status scoring
    is_large        : True if bbox_area_frac >= LARGE_OBSTACLE_AREA_FRACTION
    """
    class_name:      str
    confidence:      float
    bbox:            tuple          # (x1, y1, x2, y2)
    center_x:        int
    center_y:        int
    bbox_area_frac:  float
    zones:           List[str]      = field(default_factory=list)
    zone_overlaps:   dict           = field(default_factory=dict)
    severity:        float          = 0.0
    is_large:        bool           = False


# ======================================================================
# MAIN CLASS
# ======================================================================

class ObstacleDetector:
    """
    Filters raw YOLO boxes and produces a list of ObstacleInfo.

    The frame is divided into three equal vertical zones:
        LEFT    x in [0,           frame_width/3)
        CENTER  x in [frame_width/3, 2*frame_width/3)
        RIGHT   x in [2*frame_width/3, frame_width)

    A bounding box is associated with a zone only if the horizontal
    overlap between the box and the zone exceeds ZONE_OVERLAP_THRESHOLD.
    """

    ZONE_NAMES = ["LEFT", "CENTER", "RIGHT"]

    def __init__(self,
                 obstacle_classes: set = None,
                 confidence_threshold: float = None,
                 zone_overlap_threshold: float = None):
        """
        Parameters
        ----------
        obstacle_classes       : set of class name strings to consider
        confidence_threshold   : minimum confidence to keep a detection
        zone_overlap_threshold : minimum zone overlap fraction to assign a zone
        """
        self.obstacle_classes      = obstacle_classes      or config.OBSTACLE_CLASSES
        self.confidence_threshold  = confidence_threshold  if confidence_threshold is not None \
                                     else config.OBSTACLE_CONFIDENCE_THRESHOLD
        self.zone_overlap_threshold = zone_overlap_threshold if zone_overlap_threshold is not None \
                                      else config.ZONE_OVERLAP_THRESHOLD

    # ------------------------------------------------------------------
    def _compute_zone_boundaries(self, frame_width: int) -> List[tuple]:
        """
        Return list of (zone_name, x_start, x_end) for the three zones.
        """
        w = frame_width
        return [
            ("LEFT",   0,       w // 3),
            ("CENTER", w // 3,  (2 * w) // 3),
            ("RIGHT",  (2 * w) // 3, w),
        ]

    # ------------------------------------------------------------------
    def _compute_zone_overlap(self, x1: int, x2: int,
                               zone_x_start: int, zone_x_end: int,
                               zone_width: int) -> float:
        """
        Compute the fraction of a zone's width that is covered by [x1, x2].

        Returns a value in [0.0, 1.0].
        """
        # Intersection of bbox and zone along x-axis
        overlap_start = max(x1, zone_x_start)
        overlap_end   = min(x2, zone_x_end)
        overlap_px    = max(0, overlap_end - overlap_start)

        if zone_width == 0:
            return 0.0
        return overlap_px / zone_width

    # ------------------------------------------------------------------
    def analyze(self, yolo_result, frame_width: int, frame_height: int) -> List[ObstacleInfo]:
        """
        Process raw YOLO result boxes into a filtered list of ObstacleInfo.

        Parameters
        ----------
        yolo_result   : ultralytics Results object (result[0])
        frame_width   : image width in pixels
        frame_height  : image height in pixels

        Returns
        -------
        List[ObstacleInfo] — only obstacle-class detections above threshold
        """
        obstacles: List[ObstacleInfo] = []
        frame_area = frame_width * frame_height
        zone_boundaries = self._compute_zone_boundaries(frame_width)

        for box in yolo_result.boxes:

            confidence = float(box.conf[0])

            # --- Filter 1: confidence threshold ---
            if confidence < self.confidence_threshold:
                continue

            class_id   = int(box.cls[0])
            class_name = yolo_result.names[class_id]

            # --- Filter 2: obstacle class allowlist ---
            if class_name not in self.obstacle_classes:
                continue

            # --- Bounding box ---
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            bbox_area      = max(0, (x2 - x1) * (y2 - y1))
            bbox_area_frac = bbox_area / frame_area if frame_area > 0 else 0.0
            is_large       = bbox_area_frac >= config.LARGE_OBSTACLE_AREA_FRACTION

            # --- Zone overlap analysis ---
            zones:         List[str] = []
            zone_overlaps: dict      = {}

            for zone_name, zx_start, zx_end in zone_boundaries:
                zone_width   = zx_end - zx_start
                overlap_frac = self._compute_zone_overlap(
                    x1, x2, zx_start, zx_end, zone_width
                )

                if overlap_frac >= self.zone_overlap_threshold:
                    zones.append(zone_name)
                    zone_overlaps[zone_name] = round(overlap_frac, 3)

            # Severity: max(confidence × zone_overlap) — higher means more dangerous
            if zone_overlaps:
                severity = max(
                    confidence * ov for ov in zone_overlaps.values()
                )
            else:
                # Detected but doesn't substantially overlap any zone
                # Still record it with its center zone, low severity
                center_zone = self._center_zone(center_x, zone_boundaries)
                zones = [center_zone]
                zone_overlaps = {center_zone: 0.0}
                severity = 0.0

            obstacles.append(ObstacleInfo(
                class_name      = class_name,
                confidence      = round(confidence, 3),
                bbox            = (x1, y1, x2, y2),
                center_x        = center_x,
                center_y        = center_y,
                bbox_area_frac  = round(bbox_area_frac, 4),
                zones           = zones,
                zone_overlaps   = zone_overlaps,
                severity        = round(severity, 4),
                is_large        = is_large,
            ))

        return obstacles

    # ------------------------------------------------------------------
    @staticmethod
    def _center_zone(center_x: int, zone_boundaries: list) -> str:
        """Return the zone name that contains the given x coordinate."""
        for zone_name, zx_start, zx_end in zone_boundaries:
            if zx_start <= center_x < zx_end:
                return zone_name
        return "CENTER"  # fallback


# ======================================================================
# STANDALONE TEST
# Run: python -m perception.obstacle_detector
# ======================================================================
if __name__ == "__main__":
    import cv2
    from perception.yolo_detector import YOLODetector

    print("=" * 60)
    print("  ObstacleDetector — standalone test")
    print("=" * 60)

    # Load detector and image
    try:
        detector = YOLODetector()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not os.path.isfile(config.IMAGE_PATH):
        print(f"[ERROR] Test image not found: {config.IMAGE_PATH}")
        sys.exit(1)

    frame = cv2.imread(config.IMAGE_PATH)
    h, w  = frame.shape[:2]
    print(f"Frame: {w}x{h}")

    # Run YOLO
    yolo_result, ms = detector.detect(frame)
    print(f"YOLO inference: {ms:.1f} ms")
    print(f"Raw YOLO boxes: {len(yolo_result.boxes)}")

    # Run obstacle detector
    obs_detector = ObstacleDetector()
    obstacles    = obs_detector.analyze(yolo_result, w, h)

    print(f"\nFiltered obstacles: {len(obstacles)}")
    print()

    if not obstacles:
        print("  No obstacles after filtering.")
    else:
        print(f"  {'Class':<18} {'Conf':>5}  {'Zones':<20} {'Severity':>8}  {'Large':>5}  {'Bbox'}")
        print(f"  {'-'*18} {'-'*5}  {'-'*20} {'-'*8}  {'-'*5}  {'-'*25}")
        for obs in obstacles:
            zone_str = ",".join(obs.zones)
            bbox_str = f"({obs.bbox[0]},{obs.bbox[1]})-({obs.bbox[2]},{obs.bbox[3]})"
            print(
                f"  {obs.class_name:<18} {obs.confidence:>5.3f}  "
                f"{zone_str:<20} {obs.severity:>8.4f}  {str(obs.is_large):>5}  {bbox_str}"
            )

    print()
    print("ObstacleDetector test PASSED.")
    print("=" * 60)
