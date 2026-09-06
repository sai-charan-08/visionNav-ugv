"""
localization/slam.py

VisionNav UGV - Lightweight Visual SLAM / Localization

Pipeline:
    Camera frame
        ↓
    ORB feature detection
        ↓
    Feature matching
        ↓
    RANSAC geometric filtering
        ↓
    Essential matrix
        ↓
    Relative camera motion
        ↓
    Pose integration
        ↓
    x, y, theta

Interface:
    slam = SLAMModule()
    pose = slam.update(frame)

Pose:
    {
        "x": float,
        "y": float,
        "theta": float
    }

Notes:
    - Monocular camera provides relative scale, not absolute metric scale.
    - This module is intended for VisionNav prototype localization.
    - IMU data can be incorporated later for scale/orientation fusion.
"""

import os
import sys
import math
import cv2
import numpy as np

# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


class SLAMModule:
    """
    Lightweight monocular visual odometry / SLAM front-end.

    Estimates the UGV's relative position and heading
    from consecutive camera frames.
    """

    def __init__(
        self,
        max_features=1200,
        match_ratio=0.75,
        min_matches=12
    ):

        print("[SLAM] Initializing visual localization...")

        self.max_features = max_features
        self.match_ratio = match_ratio
        self.min_matches = min_matches

        # --------------------------------------------------------------
        # ORB feature detector
        # --------------------------------------------------------------

        self.orb = cv2.ORB_create(
            nfeatures=self.max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            fastThreshold=12
        )

        # --------------------------------------------------------------
        # Feature matcher
        # --------------------------------------------------------------

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False
        )

        # --------------------------------------------------------------
        # Previous frame information
        # --------------------------------------------------------------

        self.previous_gray = None
        self.previous_keypoints = None
        self.previous_descriptors = None

        # --------------------------------------------------------------
        # Current pose
        # --------------------------------------------------------------

        self._pose = {
            "x": 0.0,
            "y": 0.0,
            "theta": 0.0
        }

        # --------------------------------------------------------------
        # Trajectory
        # --------------------------------------------------------------

        self.trajectory = [
            (0.0, 0.0)
        ]

        # --------------------------------------------------------------
        # Camera intrinsics
        #
        # Approximate values are used when a calibrated camera
        # matrix is not supplied.
        # --------------------------------------------------------------

        self.camera_matrix = None

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        self.last_match_count = 0
        self.last_inlier_count = 0
        self.last_translation_magnitude = 0.0
        self.last_rotation = 0.0

        self.frame_count = 0

        print("[SLAM] ORB initialized.")
        print("[SLAM] Ready for camera frames.")

    # ==================================================================
    # CAMERA MODEL
    # ==================================================================

    def _create_camera_matrix(self, width, height):

        """
        Create an approximate camera intrinsic matrix.

        For the prototype we assume focal length is approximately
        equal to image width.

        A real deployment should replace this with calibration values.
        """

        focal_length = float(width)

        cx = width / 2.0
        cy = height / 2.0

        self.camera_matrix = np.array(
            [
                [focal_length, 0.0, cx],
                [0.0, focal_length, cy],
                [0.0, 0.0, 1.0]
            ],
            dtype=np.float64
        )

    # ==================================================================
    # FEATURE DETECTION
    # ==================================================================

    def _detect_features(self, gray):

        keypoints, descriptors = self.orb.detectAndCompute(
            gray,
            None
        )

        if descriptors is None:
            return [], None

        return keypoints, descriptors

    # ==================================================================
    # FEATURE MATCHING
    # ==================================================================

    def _match_features(
        self,
        previous_descriptors,
        current_descriptors
    ):

        if (
            previous_descriptors is None
            or current_descriptors is None
        ):
            return []

        if (
            len(previous_descriptors) < 2
            or len(current_descriptors) < 2
        ):
            return []

        try:

            knn_matches = self.matcher.knnMatch(
                previous_descriptors,
                current_descriptors,
                k=2
            )

        except cv2.error:
            return []

        good_matches = []

        for pair in knn_matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < self.match_ratio * n.distance:
                good_matches.append(m)

        return good_matches

    # ==================================================================
    # MOTION ESTIMATION
    # ==================================================================

    def _estimate_motion(
        self,
        previous_keypoints,
        current_keypoints,
        matches
    ):

        if len(matches) < self.min_matches:

            return None

        previous_points = np.float32(
            [
                previous_keypoints[m.queryIdx].pt
                for m in matches
            ]
        )

        current_points = np.float32(
            [
                current_keypoints[m.trainIdx].pt
                for m in matches
            ]
        )

        # --------------------------------------------------------------
        # Essential matrix
        # --------------------------------------------------------------

        try:

            E, mask = cv2.findEssentialMat(
                current_points,
                previous_points,
                self.camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )

        except cv2.error:

            return None

        if E is None or mask is None:
            return None

        # --------------------------------------------------------------
        # Keep only inliers
        # --------------------------------------------------------------

        mask = mask.ravel().astype(bool)

        inlier_previous = previous_points[mask]
        inlier_current = current_points[mask]

        self.last_inlier_count = len(
            inlier_previous
        )

        if self.last_inlier_count < self.min_matches:

            return None

        # --------------------------------------------------------------
        # Recover relative camera pose
        # --------------------------------------------------------------

        try:

            _, R, t, pose_mask = cv2.recoverPose(
                E,
                inlier_current,
                inlier_previous,
                self.camera_matrix
            )

        except cv2.error:

            return None

        if R is None or t is None:
            return None

        return R, t

    # ==================================================================
    # POSE UPDATE
    # ==================================================================

    def _update_pose(self, R, t):

        """
        Integrate relative camera motion.

        Monocular translation has unknown absolute scale.

        We therefore use a normalized translation step for
        the prototype trajectory.
        """

        translation = t.reshape(3)

        magnitude = float(
            np.linalg.norm(translation)
        )

        self.last_translation_magnitude = magnitude

        if magnitude < 1e-6:
            return

        # --------------------------------------------------------------
        # Normalize translation
        # --------------------------------------------------------------

        direction = translation / magnitude

        # Small normalized movement step.
        step = 0.10

        dx_camera = float(
            direction[0] * step
        )

        dz_camera = float(
            direction[2] * step
        )

        # --------------------------------------------------------------
        # Estimate yaw rotation from rotation matrix
        # --------------------------------------------------------------

        yaw = math.atan2(
            R[0, 2],
            R[2, 2]
        )

        self.last_rotation = yaw

        # --------------------------------------------------------------
        # Update heading
        # --------------------------------------------------------------

        self._pose["theta"] += yaw

        # Keep theta in [-pi, pi]
        self._pose["theta"] = (
            self._pose["theta"] + math.pi
        ) % (
            2.0 * math.pi
        ) - math.pi

        # --------------------------------------------------------------
        # Transform camera movement into world coordinates
        # --------------------------------------------------------------

        theta = self._pose["theta"]

        world_dx = (
            math.cos(theta) * dx_camera
            -
            math.sin(theta) * dz_camera
        )

        world_dy = (
            math.sin(theta) * dx_camera
            +
            math.cos(theta) * dz_camera
        )

        self._pose["x"] += world_dx
        self._pose["y"] += world_dy

        self.trajectory.append(
            (
                self._pose["x"],
                self._pose["y"]
            )
        )

    # ==================================================================
    # MAIN UPDATE
    # ==================================================================

    def update(
        self,
        frame,
        imu_data=None
    ):

        """
        Process a new camera frame.

        Parameters
        ----------
        frame:
            OpenCV BGR image.

        imu_data:
            Optional IMU data.
            Currently reserved for future sensor fusion.

        Returns
        -------
        dict:
            {
                "x": float,
                "y": float,
                "theta": float
            }

        Returns the current pose even when a frame does not
        contain enough features to estimate new motion.
        """

        if frame is None:

            return self.get_pose()

        if len(frame.shape) == 2:

            gray = frame

        else:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

        height, width = gray.shape[:2]

        # --------------------------------------------------------------
        # Initialize camera matrix
        # --------------------------------------------------------------

        if self.camera_matrix is None:

            self._create_camera_matrix(
                width,
                height
            )

        # --------------------------------------------------------------
        # Detect current features
        # --------------------------------------------------------------

        keypoints, descriptors = (
            self._detect_features(gray)
        )

        self.frame_count += 1

        # --------------------------------------------------------------
        # First frame
        # --------------------------------------------------------------

        if self.previous_gray is None:

            self.previous_gray = gray

            self.previous_keypoints = (
                keypoints
            )

            self.previous_descriptors = (
                descriptors
            )

            print(
                f"[SLAM] Frame {self.frame_count}: "
                f"initialized with "
                f"{len(keypoints)} features"
            )

            return self.get_pose()

        # --------------------------------------------------------------
        # Match previous/current features
        # --------------------------------------------------------------

        matches = self._match_features(
            self.previous_descriptors,
            descriptors
        )

        self.last_match_count = len(matches)

        # --------------------------------------------------------------
        # Not enough matches
        # --------------------------------------------------------------

        if len(matches) < self.min_matches:

            print(
                f"[SLAM] Frame {self.frame_count}: "
                f"insufficient matches "
                f"({len(matches)}/{self.min_matches})"
            )

            self._store_current_frame(
                gray,
                keypoints,
                descriptors
            )

            return self.get_pose()

        # --------------------------------------------------------------
        # Estimate motion
        # --------------------------------------------------------------

        motion = self._estimate_motion(
            self.previous_keypoints,
            keypoints,
            matches
        )

        if motion is None:

            print(
                f"[SLAM] Frame {self.frame_count}: "
                f"motion estimation failed"
            )

        else:

            R, t = motion

            self._update_pose(
                R,
                t
            )

            print(
                f"[SLAM] Frame {self.frame_count}: "
                f"matches={self.last_match_count}, "
                f"inliers={self.last_inlier_count}, "
                f"x={self._pose['x']:.3f}, "
                f"y={self._pose['y']:.3f}, "
                f"theta={math.degrees(self._pose['theta']):.2f}°"
            )

        # --------------------------------------------------------------
        # Store current frame
        # --------------------------------------------------------------

        self._store_current_frame(
            gray,
            keypoints,
            descriptors
        )

        return self.get_pose()

    # ==================================================================
    # STORE FRAME
    # ==================================================================

    def _store_current_frame(
        self,
        gray,
        keypoints,
        descriptors
    ):

        self.previous_gray = gray

        self.previous_keypoints = keypoints

        self.previous_descriptors = descriptors

    # ==================================================================
    # GET POSE
    # ==================================================================

    def get_pose(self):

        return {
            "x": float(self._pose["x"]),
            "y": float(self._pose["y"]),
            "theta": float(self._pose["theta"])
        }

    # ==================================================================
    # GET TRAJECTORY
    # ==================================================================

    def get_trajectory(self):

        return list(self.trajectory)

    # ==================================================================
    # GET MAP
    # ==================================================================

    def get_map(self):

        """
        Return the current estimated trajectory.

        For this lightweight implementation the map is represented
        by the estimated camera/UGV trajectory.

        A full landmark/occupancy map can be added later.
        """

        return {
            "trajectory": self.get_trajectory(),
            "pose": self.get_pose()
        }

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self):

        self.previous_gray = None
        self.previous_keypoints = None
        self.previous_descriptors = None

        self._pose = {
            "x": 0.0,
            "y": 0.0,
            "theta": 0.0
        }

        self.trajectory = [
            (0.0, 0.0)
        ]

        self.last_match_count = 0
        self.last_inlier_count = 0
        self.last_translation_magnitude = 0.0
        self.last_rotation = 0.0

        self.frame_count = 0

        print("[SLAM] Reset complete.")


# ======================================================================
# STANDALONE TEST
# ======================================================================

def main():

    print("=" * 70)
    print("VisionNav UGV - Visual Localization / SLAM Test")
    print("=" * 70)

    print(
        """
Pipeline:

Camera Frames
      ↓
ORB Features
      ↓
Feature Matching
      ↓
RANSAC
      ↓
Essential Matrix
      ↓
Relative Camera Pose
      ↓
Trajectory
      ↓
UGV Localization
"""
    )

    slam = SLAMModule()

    # --------------------------------------------------------------
    # Test image sequence
    # --------------------------------------------------------------

    sequence_dir = os.path.join(
        PROJECT_ROOT,
        "input",
        "dynamic_sequence"
    )

    if not os.path.exists(sequence_dir):

        print(
            f"[ERROR] Sequence directory not found:\n"
            f"{sequence_dir}"
        )

        return

    image_files = []

    for filename in sorted(
        os.listdir(sequence_dir)
    ):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):

            image_files.append(
                os.path.join(
                    sequence_dir,
                    filename
                )
            )

    if len(image_files) < 2:

        print(
            "[ERROR] At least two frames are "
            "required for visual localization."
        )

        return

    print(
        f"\nFound {len(image_files)} frames."
    )

    # --------------------------------------------------------------
    # Process sequence
    # --------------------------------------------------------------

    for image_path in image_files:

        frame = cv2.imread(
            image_path
        )

        if frame is None:

            print(
                f"[WARNING] Could not read:\n"
                f"{image_path}"
            )

            continue

        print(
            f"\nProcessing: "
            f"{os.path.basename(image_path)}"
        )

        slam.update(frame)

    # --------------------------------------------------------------
    # Final pose
    # --------------------------------------------------------------

    pose = slam.get_pose()

    print("\n" + "=" * 70)
    print("FINAL LOCALIZATION")
    print("=" * 70)

    print(
        f"X       : {pose['x']:.4f}"
    )

    print(
        f"Y       : {pose['y']:.4f}"
    )

    print(
        f"Theta   : "
        f"{math.degrees(pose['theta']):.2f}°"
    )

    print(
        f"Frames  : {slam.frame_count}"
    )

    print(
        f"Matches : {slam.last_match_count}"
    )

    print(
        f"Inliers : {slam.last_inlier_count}"
    )

    print(
        f"Trajectory points: "
        f"{len(slam.trajectory)}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()