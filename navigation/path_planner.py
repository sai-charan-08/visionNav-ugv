"""
navigation/path_planner.py — Global Path Planner [STUB]
========================================================
Future: compute a global path from current pose to a goal pose
using a map and localization data.

CURRENT STATUS: STUB — returns None. Not implemented.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PathPlanner:
    """
    [STUB] Global path planner.

    Future interface:
        planner  = PathPlanner(map_data)
        waypoints = planner.plan(current_pose, goal_pose)
        # Returns: list of (x, y) waypoints
    """

    def __init__(self):
        print("[PathPlanner] STUB — not implemented.")

    def plan(self, current_pose, goal_pose) -> list | None:
        """
        [STUB] Plan a path from current_pose to goal_pose.

        Future implementation would use A*, Dijkstra, or RRT
        on a 2D occupancy grid built from SLAM output.

        current_pose : (x, y, theta) — from SLAM/localization
        goal_pose    : (x, y, theta) — user-specified or mission goal
        """
        # Not implemented
        return None
