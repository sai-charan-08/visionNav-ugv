"""
control/motor_controller.py — Motor Controller [STUB]
======================================================
Future: translate navigation commands into hardware motor signals
for the physical UGV platform.

CURRENT STATUS: STUB — prints commands only. No hardware interface.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class MotorController:
    """
    [STUB] Hardware motor controller interface.

    Future interface:
        controller = MotorController(port="COM3")
        controller.execute(decision)
        # Sends PWM/serial commands to motor drivers

    The stub logs the command to stdout without any hardware interaction.
    This allows the full navigation pipeline to run without physical hardware.
    """

    # Mapping from navigation decisions to stub motor commands
    _COMMAND_MAP = {
        config.NAV_FORWARD: {"left_speed": +0.5, "right_speed": +0.5},
        config.NAV_LEFT:    {"left_speed": -0.3, "right_speed": +0.5},
        config.NAV_RIGHT:   {"left_speed": +0.5, "right_speed": -0.3},
        config.NAV_STOP:    {"left_speed":  0.0, "right_speed":  0.0},
    }

    def __init__(self, port: str = None):
        self.port = port
        print(f"[MotorController] STUB — hardware not connected. Port={port or 'N/A'}")

    def execute(self, decision: str) -> dict | None:
        """
        [STUB] Execute a navigation decision.

        Parameters
        ----------
        decision : str — one of config.NAV_* constants

        Returns the stub motor command dict (no actual hardware signals sent).
        """
        cmd = self._COMMAND_MAP.get(decision, None)
        if cmd:
            print(f"[MotorController] STUB CMD: {decision} → {cmd}")
        else:
            print(f"[MotorController] STUB: Unknown decision '{decision}'")
        return cmd

    def stop(self):
        """[STUB] Emergency stop."""
        print("[MotorController] STUB: Emergency stop.")
        return self._COMMAND_MAP[config.NAV_STOP]
