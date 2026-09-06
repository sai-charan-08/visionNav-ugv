"""
perception/traversability.py

VisionNav UGV - Lightweight Traversability Detector

Purpose
-------
Estimate which parts of a forward-facing camera image are likely
to be traversable ground.

The detector combines:

1. Bottom-center ground seed
2. LAB color similarity
3. Forward ground geometry
4. Green/saturated vertical-region rejection
5. Morphological cleanup
6. Connected-component filtering

Output
------
0 = non-traversable / unknown
1 = estimated traversable

Important
---------
This is a software prototype based on RGB camera information.

It is NOT a replacement for:
    - depth camera
    - stereo vision
    - LiDAR
    - camera calibration
    - trained semantic segmentation
"""

import cv2
import numpy as np


class TraversabilityDetector:
    """
    Lightweight RGB-based traversability estimator.

    The detector assumes the camera is mounted approximately facing
    forward from the UGV.
    """

    def __init__(
        self,
        seed_width_ratio=0.16,
        seed_height_ratio=0.10,
        color_threshold=32,
        min_component_area_ratio=0.002
    ):
        self.seed_width_ratio = seed_width_ratio
        self.seed_height_ratio = seed_height_ratio
        self.color_threshold = color_threshold
        self.min_component_area_ratio = min_component_area_ratio

    # ------------------------------------------------------------------
    # MAIN DETECTOR
    # ------------------------------------------------------------------

    def detect(self, frame):
        """
        Estimate traversable ground.

        Parameters
        ----------
        frame : numpy.ndarray
            OpenCV BGR image.

        Returns
        -------
        mask : numpy.ndarray
            uint8 mask:
                0 = non-traversable / unknown
                1 = traversable
        """

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        height, width = frame.shape[:2]

        # --------------------------------------------------------------
        # 1. Convert image to LAB and HSV
        # --------------------------------------------------------------

        lab = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2LAB
        )

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        # --------------------------------------------------------------
        # 2. Define bottom-center ground seed
        # --------------------------------------------------------------
        #
        # The region immediately in front of the UGV is our strongest
        # initial assumption for ground.
        #

        seed_width = max(
            20,
            int(width * self.seed_width_ratio)
        )

        seed_height = max(
            20,
            int(height * self.seed_height_ratio)
        )

        seed_x1 = max(
            0,
            width // 2 - seed_width // 2
        )

        seed_x2 = min(
            width,
            width // 2 + seed_width // 2
        )

        seed_y1 = max(
            0,
            height - seed_height
        )

        seed_y2 = height

        seed = lab[
            seed_y1:seed_y2,
            seed_x1:seed_x2
        ]

        # --------------------------------------------------------------
        # 3. Calculate robust ground appearance
        # --------------------------------------------------------------
        #
        # Median is less affected by small objects/reflections than
        # a simple mean.
        #

        seed_pixels = seed.reshape(-1, 3)

        ground_color = np.median(
            seed_pixels,
            axis=0
        ).astype(np.float32)

        # --------------------------------------------------------------
        # 4. Calculate LAB distance from estimated ground
        # --------------------------------------------------------------

        difference = (
            lab.astype(np.float32)
            - ground_color
        )

        color_distance = np.sqrt(
            np.sum(
                difference ** 2,
                axis=2
            )
        )

        color_candidate = (
            color_distance < self.color_threshold
        )

        # --------------------------------------------------------------
        # 5. Restrict to forward navigation region
        # --------------------------------------------------------------
        #
        # We allow the detector to look farther toward the horizon
        # than the previous version.
        #
        # Upper image:
        #     mostly sky / distant structures
        #
        # Middle:
        #     potential forward ground
        #
        # Bottom:
        #     immediate ground
        #

        geometry_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        # Horizon / upper cutoff.
        #
        # Previous value: 0.32
        # New value:      0.22
        #
        # This gives the planner more forward visibility.
        top_y = int(height * 0.22)

        # The ground corridor gets wider toward the camera.
        #
        # Far distance:
        #     narrow corridor
        #
        # Near UGV:
        #     wider corridor
        #

        for y in range(top_y, height):

            normalized = (
                (y - top_y)
                / max(1, height - top_y)
            )

            # Width grows from approximately 20% to 92%
            # of image width.
            half_width_ratio = (
                0.10
                + 0.36 * normalized
            )

            center_x = width // 2

            left = int(
                center_x
                - width * half_width_ratio
            )

            right = int(
                center_x
                + width * half_width_ratio
            )

            left = max(
                0,
                left
            )

            right = min(
                width - 1,
                right
            )

            geometry_mask[
                y,
                left:right + 1
            ] = 1

        # --------------------------------------------------------------
        # 6. Reject strongly green / saturated regions
        # --------------------------------------------------------------
        #
        # In outdoor scenes, strongly saturated green regions often
        # correspond to vegetation or painted walls rather than the
        # immediate road surface.
        #
        # This is a conservative heuristic, not a universal rule.
        #

        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]

        green_region = (
            (hue >= 35)
            & (hue <= 95)
            & (saturation >= 70)
        )

        # --------------------------------------------------------------
        # 7. Combine appearance + geometry
        # --------------------------------------------------------------

        candidate = (
            color_candidate
            & (geometry_mask == 1)
            & (~green_region)
        )

        candidate = (
            candidate.astype(np.uint8)
            * 255
        )

        # --------------------------------------------------------------
        # 8. Morphological cleanup
        # --------------------------------------------------------------

        kernel = np.ones(
            (7, 7),
            dtype=np.uint8
        )

        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            kernel
        )

        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            kernel
        )

        # --------------------------------------------------------------
        # 9. Connected component filtering
        # --------------------------------------------------------------
        #
        # Keep regions that are connected to the bottom-center ground
        # region.
        #

        connected = np.zeros_like(
            candidate
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                candidate,
                connectivity=8
            )
        )

        minimum_area = (
            height
            * width
            * self.min_component_area_ratio
        )

        # Search around bottom-center for ground components.
        bottom_x1 = max(
            0,
            width // 2 - seed_width
        )

        bottom_x2 = min(
            width,
            width // 2 + seed_width
        )

        bottom_y1 = max(
            0,
            height - 15
        )

        bottom_labels = set()

        for y in range(
            bottom_y1,
            height
        ):

            for x in range(
                bottom_x1,
                bottom_x2
            ):

                label = labels[y, x]

                if label == 0:
                    continue

                area = stats[
                    label,
                    cv2.CC_STAT_AREA
                ]

                if area >= minimum_area:
                    bottom_labels.add(label)

        # Keep bottom-connected components.
        for label in bottom_labels:

            connected[
                labels == label
            ] = 255

        # --------------------------------------------------------------
        # 10. Final binary mask
        # --------------------------------------------------------------

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        mask[
            connected > 0
        ] = 1

        return mask


# ======================================================================
# VISUALIZATION
# ======================================================================

def create_overlay(
    frame,
    mask
):
    """
    Create a visualization showing estimated traversable regions.

    Green:
        estimated traversable ground

    Original image:
        retained underneath the overlay
    """

    output = frame.copy()

    overlay = frame.copy()

    overlay[
        mask == 1
    ] = (0, 255, 0)

    output = cv2.addWeighted(
        output,
        0.65,
        overlay,
        0.35,
        0
    )

    return output


def save_visualization(
    frame,
    mask,
    output_path
):
    """
    Save traversability overlay.
    """

    output = create_overlay(
        frame,
        mask
    )

    cv2.imwrite(
        output_path,
        output
    )


# ======================================================================
# STANDALONE TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("VisionNav UGV - Improved Traversability Detector Test")
    print("=" * 70)

    image_path = "input/test.jpeg"

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        print(
            f"[ERROR] Could not read image: "
            f"{image_path}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------------
    # Create detector
    # --------------------------------------------------------------

    detector = TraversabilityDetector()

    # --------------------------------------------------------------
    # Detect traversability
    # --------------------------------------------------------------

    mask = detector.detect(
        frame
    )

    # --------------------------------------------------------------
    # Save binary mask
    # --------------------------------------------------------------

    mask_path = (
        "runs/traversability_mask.jpg"
    )

    cv2.imwrite(
        mask_path,
        mask * 255
    )

    # --------------------------------------------------------------
    # Save overlay
    # --------------------------------------------------------------

    overlay_path = (
        "runs/traversability_overlay.jpg"
    )

    save_visualization(
        frame,
        mask,
        overlay_path
    )

    # --------------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------------

    traversable_pixels = np.sum(
        mask == 1
    )

    total_pixels = mask.size

    percentage = (
        traversable_pixels
        / total_pixels
        * 100
    )

    # --------------------------------------------------------------
    # Print results
    # --------------------------------------------------------------

    print(
        f"\nInput image: "
        f"{frame.shape[1]} x {frame.shape[0]}"
    )

    print(
        f"\nEstimated traversable area: "
        f"{percentage:.2f}%"
    )

    print(
        "\nBinary mask saved to:"
    )

    print(
        f"    {mask_path}"
    )

    print(
        "\nOverlay visualization saved to:"
    )

    print(
        f"    {overlay_path}"
    )

    print("\nLegend:")

    print(
        "    WHITE = estimated traversable"
    )

    print(
        "    BLACK = non-traversable / unknown"
    )

    print(
        "\nTraversability detector test completed."
    )

    print("=" * 70)