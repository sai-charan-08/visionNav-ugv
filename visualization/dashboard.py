"""
visualization/dashboard.py — Navigation Dashboard Renderer
===========================================================
Draws a fully annotated overlay on the camera frame showing:

  - Left / Center / Right zone dividers with status colors
  - Bounding boxes with class, confidence, zone
  - Zone status badges (FREE / PARTIALLY BLOCKED / BLOCKED)
  - Navigation decision with large colored text
  - Obstacle count and inference time
  - System title banner

Saves the result to runs/visionnav_navigation_result.jpg
Optionally shows the result in an OpenCV window.
"""

import sys
import os
import cv2
import numpy as np
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from perception.obstacle_detector import ObstacleInfo
from perception.free_space        import STATUS_FREE, STATUS_PARTIAL, STATUS_BLOCKED, ALL_ZONES


# ======================================================================
# COLOR PALETTE  (BGR)
# ======================================================================

# Zone divider lines
_ZONE_LINE    = (200, 200, 200)

# Zone header background colors based on status
_ZONE_BG = {
    STATUS_FREE:    (0, 140, 0),     # dark green
    STATUS_PARTIAL: (0, 140, 200),   # amber
    STATUS_BLOCKED: (0, 0, 200),     # red
}

# Bbox colors per zone
_BBOX_COLOR = {
    "LEFT":   (0, 200, 255),   # amber
    "CENTER": (0, 60,  255),   # red-orange
    "RIGHT":  (0, 200, 255),   # amber
}

# Navigation decision colors
_NAV_COLOR = {
    config.NAV_FORWARD: (0, 220, 60),
    config.NAV_LEFT:    (0, 200, 255),
    config.NAV_RIGHT:   (0, 200, 255),
    config.NAV_STOP:    (0, 0,   255),
}

# Text rendering defaults
_FONT         = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SMALL   = 0.55
_FONT_MEDIUM  = 0.75
_FONT_LARGE   = 1.15
_THICKNESS    = 2
_WHITE        = (255, 255, 255)
_BLACK        = (0, 0, 0)
_DARK_OVERLAY = (20, 20, 20)


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def _put_text_with_bg(img, text: str, org: tuple, font_scale: float,
                      color: tuple, thickness: int = 1, padding: int = 5):
    """Draw text with a dark semi-transparent background rectangle."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    x, y = org
    # Background rect
    cv2.rectangle(
        img,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        _DARK_OVERLAY,
        cv2.FILLED,
    )
    cv2.putText(img, text, (x, y), _FONT, font_scale, color, thickness, cv2.LINE_AA)


def _zone_x_ranges(frame_width: int):
    """Return (left_x1, left_x2, center_x1, center_x2, right_x1, right_x2)."""
    w = frame_width
    return 0, w // 3, w // 3, (2 * w) // 3, (2 * w) // 3, w


# ======================================================================
# DASHBOARD CLASS
# ======================================================================

class Dashboard:
    """
    Renders the full annotated navigation frame.

    Usage:
        dash = Dashboard()
        annotated = dash.render(frame, obstacles, free_space, decision, inference_ms)
        dash.save(annotated)
        dash.show(annotated)   # optional
    """

    def __init__(self):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    def render(self,
               frame,
               obstacles: List[ObstacleInfo],
               free_space: Dict[str, dict],
               decision: str,
               inference_ms: float = 0.0):
        """
        Build and return the annotated frame (does not display or save).

        Parameters
        ----------
        frame        : original BGR numpy array
        obstacles    : list of ObstacleInfo
        free_space   : dict from FreeSpaceEstimator.estimate()
        decision     : navigation decision string
        inference_ms : YOLO inference time in ms

        Returns
        -------
        annotated : numpy.ndarray (BGR)
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Zone x boundaries
        lx1, lx2, cx1, cx2, rx1, rx2 = _zone_x_ranges(w)

        # 1. Draw zone background tint (semi-transparent)
        self._draw_zone_tints(annotated, free_space, w, h)

        # 2. Draw zone divider lines
        cv2.line(annotated, (lx2, 0), (lx2, h), _ZONE_LINE, 2)
        cv2.line(annotated, (rx1, 0), (rx1, h), _ZONE_LINE, 2)

        # 3. Draw zone status badges at the top
        self._draw_zone_badges(annotated, free_space, w)

        # 4. Draw bounding boxes and labels for each obstacle
        for obs in obstacles:
            self._draw_obstacle(annotated, obs)

        # 5. Draw bottom info bar
        self._draw_info_bar(annotated, obstacles, decision, inference_ms, h, w)

        return annotated

    # ------------------------------------------------------------------
    def _draw_zone_tints(self, img, free_space, w, h):
        """Draw a subtle color tint over each zone based on its status."""
        overlay = img.copy()
        lx1, lx2, cx1, cx2, rx1, rx2 = _zone_x_ranges(w)

        zone_ranges = {
            "LEFT":   (lx1, lx2),
            "CENTER": (cx1, cx2),
            "RIGHT":  (rx1, rx2),
        }

        for zone, (x_start, x_end) in zone_ranges.items():
            status = free_space[zone]["status"]
            color  = _ZONE_BG.get(status, _DARK_OVERLAY)
            # Very light tint (alpha = 0.12)
            cv2.rectangle(overlay, (x_start, 0), (x_end, h), color, cv2.FILLED)

        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    # ------------------------------------------------------------------
    def _draw_zone_badges(self, img, free_space, w):
        """Draw zone name + status label at the top of each zone."""
        lx1, lx2, cx1, cx2, rx1, rx2 = _zone_x_ranges(w)
        zone_x_center = {
            "LEFT":   lx1 + (lx2 - lx1) // 2,
            "CENTER": cx1 + (cx2 - cx1) // 2,
            "RIGHT":  rx1 + (rx2 - rx1) // 2,
        }

        for zone in ALL_ZONES:
            info   = free_space[zone]
            status = info["status"]
            sev    = info["severity"]
            xc     = zone_x_center[zone]
            color  = _ZONE_BG.get(status, _WHITE)

            # Zone name
            (tw, _), _ = cv2.getTextSize(zone, _FONT, _FONT_MEDIUM, _THICKNESS)
            _put_text_with_bg(img, zone, (xc - tw // 2, 38), _FONT_MEDIUM, _WHITE, _THICKNESS)

            # Status
            short_status = status.replace("PARTIALLY ", "PARTIAL\n")
            status_lines = status.split()  # e.g. ["PARTIALLY", "BLOCKED"]
            if len(status_lines) > 1:
                label = status_lines[1]     # "BLOCKED" etc
            else:
                label = status

            (tw2, _), _ = cv2.getTextSize(label, _FONT, _FONT_SMALL, 1)
            _put_text_with_bg(img, label, (xc - tw2 // 2, 65), _FONT_SMALL, color, 1)

            # Severity
            sev_label = f"sev:{sev:.2f}"
            (tw3, _), _ = cv2.getTextSize(sev_label, _FONT, _FONT_SMALL - 0.1, 1)
            _put_text_with_bg(img, sev_label, (xc - tw3 // 2, 85), _FONT_SMALL - 0.1, _WHITE, 1)

    # ------------------------------------------------------------------
    def _draw_obstacle(self, img, obs: ObstacleInfo):
        """Draw bounding box, center dot, and label for one obstacle."""
        x1, y1, x2, y2 = obs.bbox

        # Pick color based on primary zone
        primary_zone = obs.zones[0] if obs.zones else "CENTER"
        color = _BBOX_COLOR.get(primary_zone, _WHITE)

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Center dot
        cv2.circle(img, (obs.center_x, obs.center_y), 4, color, -1)

        # Label: class + confidence + zone(s)
        zone_str = ",".join(obs.zones)
        label    = f"{obs.class_name} {obs.confidence:.2f} [{zone_str}]"
        _put_text_with_bg(
            img, label,
            (x1, max(y1 - 8, 20)),
            _FONT_SMALL, color, 1,
            padding=4
        )

        # Mark "LARGE" if applicable
        if obs.is_large:
            _put_text_with_bg(
                img, "LARGE",
                (x1, max(y1 - 28, 20)),
                _FONT_SMALL - 0.1, (0, 120, 255), 1,
                padding=3
            )

    # ------------------------------------------------------------------
    def _draw_info_bar(self, img, obstacles, decision, inference_ms, h, w):
        """Draw the bottom info strip: title, object count, decision, FPS."""
        bar_height = 80
        # Semi-transparent dark bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Title
        _put_text_with_bg(img, "VisionNav  UGV", (12, h - 50), _FONT_MEDIUM, _WHITE, 2, padding=0)

        # Object count
        obj_text = f"Objects: {len(obstacles)}"
        _put_text_with_bg(img, obj_text, (12, h - 22), _FONT_SMALL, _WHITE, 1, padding=0)

        # Inference time
        if inference_ms > 0:
            fps_text = f"{inference_ms:.0f}ms"
            (tw, _), _ = cv2.getTextSize(fps_text, _FONT, _FONT_SMALL, 1)
            _put_text_with_bg(img, fps_text, (w - tw - 12, h - 22), _FONT_SMALL, (180, 180, 180), 1, padding=0)

        # Navigation decision (large, colored, centered)
        nav_color = _NAV_COLOR.get(decision, _WHITE)
        (tw, _), _ = cv2.getTextSize(decision, _FONT, _FONT_LARGE, 2)
        nav_x = w - tw - 14
        _put_text_with_bg(img, decision, (nav_x, h - 42), _FONT_LARGE, nav_color, 2, padding=4)

    # ------------------------------------------------------------------
    def save(self, annotated_frame, filename: str = None) -> str:
        """
        Save the annotated frame to the output directory.
        Returns the output file path.
        """
        filename = filename or config.OUTPUT_FILENAME
        output_path = os.path.join(config.OUTPUT_DIR, filename)
        ok = cv2.imwrite(output_path, annotated_frame)
        if ok:
            print(f"[Dashboard] Saved: {output_path}")
        else:
            print(f"[Dashboard] WARNING: Failed to save to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    def show(self, annotated_frame, title: str = None):
        """
        Display the annotated frame in an OpenCV window.
        Blocks until any key is pressed or the window is closed.
        """
        if not config.SHOW_WINDOW:
            return
        title = title or config.WINDOW_TITLE
        cv2.imshow(title, annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ======================================================================
# STANDALONE TEST
# Run: python -m visualization.dashboard
# ======================================================================
if __name__ == "__main__":
    from perception.yolo_detector    import YOLODetector
    from perception.obstacle_detector import ObstacleDetector
    from perception.free_space        import FreeSpaceEstimator
    from navigation.decision_engine   import DecisionEngine

    print("=" * 60)
    print("  Dashboard — standalone test")
    print("=" * 60)

    try:
        detector     = YOLODetector()
        obs_detector = ObstacleDetector()
        fs_estimator = FreeSpaceEstimator()
        engine       = DecisionEngine()
        dash         = Dashboard()
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

    annotated = dash.render(frame, obstacles, free_space, decision, ms)
    out_path  = dash.save(annotated)

    print(f"Decision : {decision}")
    print(f"Saved    : {out_path}")
    print()
    print("Dashboard test PASSED.")
    print("=" * 60)
    print()
    print("Opening image window — press any key to close.")
    dash.show(annotated)
