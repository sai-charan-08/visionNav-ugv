"""
navigation/local_planner.py — Local Planner [STUB]
===================================================
Future: compute a short-horizon path around nearby obstacles
using sensor data and the navigation decision.

CURRENT STATUS: STUB — returns None. Not implemented.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LocalPlanner:
    """
    [STUB] Short-horizon local path planner.

    Future interface:
        plan = LocalPlanner()
        velocity_cmd = plan.compute(obstacles, free_space, decision)
        # Returns: {"linear_velocity": float, "angular_velocity": float}
    """

    def __init__(self):
        print("[LocalPlanner] STUB — not implemented.")

    def compute(self, obstacles, free_space, decision) -> dict | None:
        """
        [STUB] Compute velocity commands from navigation decision.

        Future implementation would:
        - Convert TURN LEFT/RIGHT into angular velocity
        - Scale MOVE FORWARD velocity based on clear distance
        - Apply obstacle avoidance field
        """
        # Not implemented — return None to signal no command
        return None
