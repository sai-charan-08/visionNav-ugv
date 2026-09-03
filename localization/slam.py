"""
localization/slam.py — SLAM Module [STUB]
=========================================
Future: simultaneous localization and mapping using camera/IMU/LiDAR.

CURRENT STATUS: STUB — returns None. Not implemented.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SLAMModule:
    """
    [STUB] SLAM (Simultaneous Localization and Mapping) interface.

    Future interface:
        slam = SLAMModule()
        pose = slam.update(frame, imu_data)
        # Returns: {"x": float, "y": float, "theta": float}

    Candidate implementations for the future:
        - ORB-SLAM3 (camera-based)
        - OpenVSLAM
        - RTAB-Map
        - Custom visual odometry
    """

    def __init__(self):
        print("[SLAM] STUB — not implemented.")
        self._pose = {"x": 0.0, "y": 0.0, "theta": 0.0}

    def update(self, frame, imu_data=None) -> dict | None:
        """
        [STUB] Update pose estimate from new camera frame.

        Returns pose dict or None if not implemented.
        """
        # Not implemented — return None
        return None

    def get_map(self):
        """[STUB] Return the current occupancy map."""
        return None
