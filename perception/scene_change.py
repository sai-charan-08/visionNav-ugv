import cv2
import numpy as np


class SceneChangeDetector:
    """
    Detect newly appearing obstacles between two
    consecutive camera frames.

    The previous frame is first aligned to the
    current frame so that small camera/image
    differences do not appear as obstacles.
    """

    def __init__(
        self,
        diff_threshold=25,
        min_area=500,
    ):
        self.diff_threshold = diff_threshold
        self.min_area = min_area

    # ========================================================
    # ALIGN PREVIOUS FRAME TO CURRENT FRAME
    # ========================================================

    def _align_frames(
        self,
        previous_frame,
        current_frame
    ):
        """
        Use ORB feature matching + homography to align
        the previous frame with the current frame.
        """

        previous_gray = cv2.cvtColor(
            previous_frame,
            cv2.COLOR_BGR2GRAY
        )

        current_gray = cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2GRAY
        )

        orb = cv2.ORB_create(
            nfeatures=3000
        )

        kp1, des1 = orb.detectAndCompute(
            previous_gray,
            None
        )

        kp2, des2 = orb.detectAndCompute(
            current_gray,
            None
        )

        if des1 is None or des2 is None:
            return previous_frame

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

        matches = matcher.knnMatch(
            des1,
            des2,
            k=2
        )

        good_matches = []

        for pair in matches:

            if len(pair) != 2:
                continue

            m, n = pair

            if m.distance < 0.70 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            return previous_frame

        src_points = np.float32(
            [
                kp1[m.queryIdx].pt
                for m in good_matches
            ]
        ).reshape(-1, 1, 2)

        dst_points = np.float32(
            [
                kp2[m.trainIdx].pt
                for m in good_matches
            ]
        ).reshape(-1, 1, 2)

        homography, inlier_mask = (
            cv2.findHomography(
                src_points,
                dst_points,
                cv2.RANSAC,
                5.0
            )
        )

        if homography is None:
            return previous_frame

        height, width = current_frame.shape[:2]

        aligned = cv2.warpPerspective(
            previous_frame,
            homography,
            (width, height)
        )

        return aligned

    # ========================================================
    # DETECT NEW OBSTACLES
    # ========================================================

    def detect(
        self,
        previous_frame,
        current_frame
    ):

        if previous_frame is None:
            return []

        if current_frame is None:
            return []

        # ----------------------------------------------------
        # Make dimensions identical
        # ----------------------------------------------------

        if (
            previous_frame.shape[:2]
            != current_frame.shape[:2]
        ):

            previous_frame = cv2.resize(
                previous_frame,
                (
                    current_frame.shape[1],
                    current_frame.shape[0]
                )
            )

        # ----------------------------------------------------
        # Align frames
        # ----------------------------------------------------

        aligned_previous = (
            self._align_frames(
                previous_frame,
                current_frame
            )
        )

        previous_gray = cv2.cvtColor(
            aligned_previous,
            cv2.COLOR_BGR2GRAY
        )

        current_gray = cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Calculate frame difference
        # ----------------------------------------------------

        difference = cv2.absdiff(
            previous_gray,
            current_gray
        )

        difference = cv2.GaussianBlur(
            difference,
            (9, 9),
            0
        )

        _, binary = cv2.threshold(
            difference,
            self.diff_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # ----------------------------------------------------
        # Restrict to forward driving region
        # ----------------------------------------------------

        height, width = binary.shape

        roi_mask = np.zeros_like(
            binary
        )

        polygon = np.array(
            [[
                int(width * 0.30),
                int(height * 0.18)
            ], [
                int(width * 0.70),
                int(height * 0.18)
            ], [
                int(width * 0.92),
                int(height * 0.96)
            ], [
                int(width * 0.08),
                int(height * 0.96)
            ]],
            dtype=np.int32
        )

        cv2.fillPoly(
            roi_mask,
            [polygon],
            255
        )

        binary = cv2.bitwise_and(
            binary,
            roi_mask
        )

        # ----------------------------------------------------
        # Remove small noise
        # ----------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )

        # ----------------------------------------------------
        # Find changed regions
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < self.min_area:
                continue

            x, y, w, h = (
                cv2.boundingRect(
                    contour
                )
            )

            if w < 20 or h < 20:
                continue

            boxes.append(
                (
                    x,
                    y,
                    x + w,
                    y + h
                )
            )

        return self._merge_boxes(
            boxes
        )

    # ========================================================
    # MERGE OVERLAPPING BOXES
    # ========================================================

    def _merge_boxes(
        self,
        boxes
    ):

        if not boxes:
            return []

        changed = True

        while changed:

            changed = False
            result = []

            used = [
                False
                for _ in boxes
            ]

            for i in range(
                len(boxes)
            ):

                if used[i]:
                    continue

                x1, y1, x2, y2 = (
                    boxes[i]
                )

                used[i] = True

                for j in range(
                    i + 1,
                    len(boxes)
                ):

                    if used[j]:
                        continue

                    a1, b1, a2, b2 = (
                        boxes[j]
                    )

                    if not (
                        x2 < a1 - 20
                        or a2 < x1 - 20
                        or y2 < b1 - 20
                        or b2 < y1 - 20
                    ):

                        x1 = min(
                            x1,
                            a1
                        )

                        y1 = min(
                            y1,
                            b1
                        )

                        x2 = max(
                            x2,
                            a2
                        )

                        y2 = max(
                            y2,
                            b2
                        )

                        used[j] = True
                        changed = True

                result.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

            boxes = result

        return boxes


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    import os

    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    frame1_path = os.path.join(
        base_dir,
        "input",
        "dynamic_sequence",
        "frame_01_clear.png"
    )

    frame2_path = os.path.join(
        base_dir,
        "input",
        "dynamic_sequence",
        "frame_02_blocked.png"
    )

    frame1 = cv2.imread(
        frame1_path
    )

    frame2 = cv2.imread(
        frame2_path
    )

    detector = (
        SceneChangeDetector()
    )

    boxes = detector.detect(
        frame1,
        frame2
    )

    print(
        "\nNew scene-change obstacles:"
    )

    for box in boxes:

        print(
            f"  {box}"
        )

    print(
        f"\nTotal detected changes: "
        f"{len(boxes)}"
    )