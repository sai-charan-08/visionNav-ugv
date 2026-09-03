"""
perception/yolo_detector.py — YOLO Model Wrapper
=================================================
Encapsulates YOLO model loading and raw inference.
Returns raw detection results; all filtering happens downstream.

This module preserves the proven YOLO inference logic from yolo_test.py
and wraps it in a reusable, testable class.
"""

import sys
import os
import time
import cv2

# Allow running as a standalone module from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class YOLODetector:
    """
    Loads a YOLO segmentation model and runs inference on a single frame.

    Usage:
        detector = YOLODetector()
        raw_results, inference_ms = detector.detect(frame)
    """

    def __init__(self, model_path: str = None, confidence: float = None):
        """
        Parameters
        ----------
        model_path : str, optional
            Path to the .pt model file. Defaults to config.MODEL_PATH.
        confidence : float, optional
            Minimum confidence passed to YOLO. Defaults to config.YOLO_CONFIDENCE.
        """
        self.model_path = model_path or config.MODEL_PATH
        self.confidence = confidence if confidence is not None else config.YOLO_CONFIDENCE
        self.model = None
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load the YOLO model. Raises FileNotFoundError if model is missing."""
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}\n"
                f"Make sure the model file exists at that path."
            )

        # Import here so the module can be imported even without ultralytics
        # (useful for partial unit tests)
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from e

        print(f"[YOLODetector] Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("[YOLODetector] Model loaded successfully.")

    # ------------------------------------------------------------------
    def detect(self, frame) -> tuple:
        """
        Run YOLO inference on a single BGR frame (NumPy array).

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image as returned by cv2.imread or cv2.VideoCapture.

        Returns
        -------
        result : ultralytics Results object (index 0)
            Contains .boxes, .masks, .names, etc.
        inference_ms : float
            Wall-clock inference time in milliseconds.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call _load_model() first.")

        t_start = time.perf_counter()

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False,          # suppress per-frame console spam
        )

        inference_ms = (time.perf_counter() - t_start) * 1000.0

        return results[0], inference_ms

    # ------------------------------------------------------------------
    @property
    def class_names(self) -> dict:
        """
        Return the model's class-name dictionary {id: name}.
        """
        if self.model is None:
            return {}
        return self.model.names


# ======================================================================
# STANDALONE TEST
# Run: python -m perception.yolo_detector
# ======================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  YOLODetector — standalone test")
    print("=" * 55)

    # 1. Load detector
    try:
        detector = YOLODetector()
    except (FileNotFoundError, ImportError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # 2. Load test image
    if not os.path.isfile(config.IMAGE_PATH):
        print(f"[ERROR] Test image not found: {config.IMAGE_PATH}")
        sys.exit(1)

    frame = cv2.imread(config.IMAGE_PATH)
    if frame is None:
        print(f"[ERROR] cv2 could not read: {config.IMAGE_PATH}")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"[YOLODetector] Image size: {w} x {h}")

    # 3. Run inference
    result, ms = detector.detect(frame)

    # 4. Report raw detections
    boxes = result.boxes
    print(f"\n[YOLODetector] Inference time  : {ms:.1f} ms")
    print(f"[YOLODetector] Raw detections  : {len(boxes)}")
    print()

    if len(boxes) == 0:
        print("  No objects detected at conf >= {:.2f}".format(config.YOLO_CONFIDENCE))
    else:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = detector.class_names.get(cls_id, "unknown")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"  [{i+1}] {cls_name:<18} conf={conf:.3f}  box=({x1},{y1})-({x2},{y2})")

    print()
    print("YOLODetector test PASSED.")
    print("=" * 55)
