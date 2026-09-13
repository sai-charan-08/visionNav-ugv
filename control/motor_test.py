"""
VisionNav UGV - Motor Controller Test

Tests the software motor-control interface.
No physical hardware is connected.
"""

from control.motor_controller import MotorController
import config


def main():

    print("=" * 60)
    print("VisionNav UGV - Motor Controller Test")
    print("=" * 60)

    controller = MotorController()

    commands = [
        config.NAV_FORWARD,
        config.NAV_LEFT,
        config.NAV_RIGHT,
        config.NAV_STOP,
    ]

    for decision in commands:

        print(f"\nNavigation decision: {decision}")

        motor_command = controller.execute(decision)

        print(
            f"Left wheel speed : {motor_command['left_speed']}"
        )

        print(
            f"Right wheel speed: {motor_command['right_speed']}"
        )

    print("\nTesting emergency stop...")

    stop_command = controller.stop()

    print(
        f"Left wheel speed : {stop_command['left_speed']}"
    )

    print(
        f"Right wheel speed: {stop_command['right_speed']}"
    )

    print("\n" + "=" * 60)
    print("MOTOR CONTROLLER TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()