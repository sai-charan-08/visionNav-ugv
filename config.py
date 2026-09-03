"""
config.py — VisionNav-UGV Central Configuration
================================================
All tunable constants live here.
No thresholds or paths should be hard-coded inside modules.
"""

import os

# ============================================================
# INPUT MODE
# ============================================================
# "image"  — run on a single static image (no hardware needed)
# "camera" — run on a live webcam feed
INPUT_MODE = "image"

# Webcam device index (used only when INPUT_MODE = "camera")
CAMERA_INDEX = 0

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n-seg.pt")
IMAGE_PATH = os.path.join(BASE_DIR, "input", "test.jpeg")
OUTPUT_DIR = os.path.join(BASE_DIR, "runs")
OUTPUT_FILENAME = "visionnav_navigation_result.jpg"

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILENAME = "visionnav_log.csv"

# ============================================================
# YOLO INFERENCE
# ============================================================
# Confidence passed to YOLO — pre-filter to avoid processing junk
YOLO_CONFIDENCE = 0.50

# ============================================================
# OBSTACLE DETECTION
# ============================================================
# Hard minimum confidence; detections below this are always discarded
OBSTACLE_CONFIDENCE_THRESHOLD = 0.55

# Classes treated as potential obstacles for UGV navigation.
# Anything NOT in this list is ignored (road surface, sky, etc.)
OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "suitcase",
}

# Minimum fraction of a zone's width that a bounding box must overlap
# before that zone is considered blocked by this object.
# (0.0 = any overlap counts; 1.0 = must fully cover the zone)
ZONE_OVERLAP_THRESHOLD = 0.15

# Bounding-box area as fraction of total frame area above which an obstacle
# is considered "large" (contributes more weight to navigation scoring).
LARGE_OBSTACLE_AREA_FRACTION = 0.08

# ============================================================
# FREE SPACE ESTIMATION
# ============================================================
# Severity score thresholds for zone status.
# Severity = sum of (confidence * bbox_zone_overlap_fraction) per obstacle in zone.
ZONE_BLOCKED_SEVERITY    = 0.50   # above this → BLOCKED
ZONE_PARTIAL_SEVERITY    = 0.15   # above this → PARTIALLY BLOCKED; below → FREE

# ============================================================
# NAVIGATION DECISION
# ============================================================
# Navigation outputs
NAV_FORWARD    = "MOVE FORWARD"
NAV_LEFT       = "TURN LEFT"
NAV_RIGHT      = "TURN RIGHT"
NAV_STOP       = "STOP"

# ============================================================
# VISUALIZATION
# ============================================================
# Whether to open an OpenCV display window (set False for headless runs)
SHOW_WINDOW = True

# Window title
WINDOW_TITLE = "VisionNav — UGV Navigation"

# Zone line color (BGR)
ZONE_LINE_COLOR   = (200, 200, 200)
ZONE_LINE_THICKNESS = 2

# Bounding box colors per zone (BGR)
BBOX_COLOR_LEFT   = (0, 200, 255)   # amber
BBOX_COLOR_CENTER = (0, 60,  255)   # red-orange
BBOX_COLOR_RIGHT  = (0, 200, 255)   # amber

# Navigation decision text color
NAV_COLOR_FORWARD = (0, 220, 0)
NAV_COLOR_TURN    = (0, 165, 255)
NAV_COLOR_STOP    = (0, 0,   255)

# ============================================================
# LOGGING
# ============================================================
ENABLE_LOGGING = True


# ============================================================
# QUICK SELF-TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  VisionNav-UGV Configuration")
    print("=" * 50)
    print(f"  INPUT_MODE              : {INPUT_MODE}")
    print(f"  CAMERA_INDEX            : {CAMERA_INDEX}")
    print(f"  MODEL_PATH              : {MODEL_PATH}")
    print(f"  MODEL EXISTS            : {os.path.isfile(MODEL_PATH)}")
    print(f"  IMAGE_PATH              : {IMAGE_PATH}")
    print(f"  IMAGE EXISTS            : {os.path.isfile(IMAGE_PATH)}")
    print(f"  OUTPUT_DIR              : {OUTPUT_DIR}")
    print(f"  YOLO_CONFIDENCE         : {YOLO_CONFIDENCE}")
    print(f"  OBSTACLE_CONF_THRESHOLD : {OBSTACLE_CONFIDENCE_THRESHOLD}")
    print(f"  ZONE_OVERLAP_THRESHOLD  : {ZONE_OVERLAP_THRESHOLD}")
    print(f"  ZONE_BLOCKED_SEVERITY   : {ZONE_BLOCKED_SEVERITY}")
    print(f"  ZONE_PARTIAL_SEVERITY   : {ZONE_PARTIAL_SEVERITY}")
    print(f"  OBSTACLE_CLASSES ({len(OBSTACLE_CLASSES)})   : {sorted(OBSTACLE_CLASSES)}")
    print(f"  ENABLE_LOGGING          : {ENABLE_LOGGING}")
    print("=" * 50)
    print("config.py OK")
