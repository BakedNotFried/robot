#!/usr/bin/env python3

import time
import rclpy

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand

from robot.constants import (
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_OPEN_MAX,
    FOLLOWER_GRIPPER_JOINT_OPEN,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    FOLLOWER_GRIPPER_JOINT_CLOSE_MAX,
    FOLLOWER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)
from robot.robot_utils import (
    move_arms,
    move_grippers,
    torque_on,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

def opening_ceremony(
    robot: InterbotixManipulatorXS,
) -> None:
    """Move robot arm to start demonstration pose"""
    # reboot gripper motors, and set operating modes for all motors
    robot.core.robot_reboot_motors('single', 'gripper', True)
    robot.core.robot_set_operating_modes('group', 'arm', 'position')
    robot.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    robot.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    torque_on(robot)

    # move arm to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [robot],
        [start_arm_qpos],
        moving_time=2.0,
    )
    # move gripper to starting position
    move_grippers(
        [robot],
        [FOLLOWER_GRIPPER_JOINT_MID],
        moving_time=0.5
    )


def main() -> None:
    node = create_interbotix_global_node('robot')
    robot = InterbotixManipulatorXS(
        robot_model='wx250s',
        robot_name='robot',
        node=node,
        iterative_update_fk=False,
    )

    robot_startup(node)

    opening_ceremony(robot)

    # Teleoperation loop
    robot_gripper_command = JointSingleCommand(name='gripper')
    while rclpy.ok():
        # sync joint positions
        robot_joint_states = robot.core.joint_states.position
        # joints = [0.003067961661145091, -0.9710098505020142, 1.2103108167648315, 0.004601942375302315, -0.28378644585609436, 0.010737866163253784]
        # robot.arm.set_joint_positions(joints, blocking=False)
        # sync gripper positions
        # robot_gripper_command.cmd = LEADER2FOLLOWER_JOINT_FN(
        #     robot.core.joint_states.position[6]
        # )
        robot_gripper_command.cmd = FOLLOWER_GRIPPER_JOINT_CLOSE_MAX
        robot.gripper.core.pub_single.publish(robot_gripper_command)
        # sleep DT
        # get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

        # debug
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)
        print(robot_joint_states)
        print(robot_joint_states[6])
        print(len(robot_joint_states))

    robot_shutdown(node)


if __name__ == '__main__':
    main()
