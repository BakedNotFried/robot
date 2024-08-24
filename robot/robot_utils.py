import time
from typing import Sequence

from robot.constants import (
    DT
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np


def get_arm_joint_positions(bot: InterbotixManipulatorXS):
    return bot.arm.core.joint_states.position[:6]


def get_arm_gripper_positions(bot: InterbotixManipulatorXS):
    joint_position = bot.gripper.core.joint_states.position[6]
    return joint_position


def move_arms(
    bot_list: Sequence[InterbotixManipulatorXS],
    target_pose_list: Sequence[Sequence[float]],
    moving_time: float = 1.0,
) -> None:
    num_steps = int(moving_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    zipped_lists = zip(curr_pose_list, target_pose_list)
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zipped_lists
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)


def control_arms(
    bot_list: Sequence[InterbotixManipulatorXS],
    target_pose_list: Sequence[Sequence[float]],
    curr_pose_list: Sequence[Sequence[float]],
    moving_time: float = 1.0,
) -> None:
    num_steps = int(moving_time / DT)
    zipped_lists = zip(curr_pose_list, target_pose_list)
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zipped_lists
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)


def sleep_arms(
    bot_list: Sequence[InterbotixManipulatorXS],
    moving_time: float = 5.0,
    home_first: bool = True,
) -> None:
    """Command given list of arms to their sleep poses, optionally to their home poses first.

    :param bot_list: List of bots to command to their sleep poses
    :param moving_time: Duration in seconds the movements should take, defaults to 5.0
    :param home_first: True to command the arms to their home poses first, defaults to True
    """
    if home_first:
        move_arms(
            bot_list,
            [[0.0, -0.96, 1.16, 0.0, -0.3, 0.0]] * len(bot_list),
            moving_time=moving_time
        )
    move_arms(
        bot_list,
        [bot.arm.group_info.joint_sleep_positions for bot in bot_list],
        moving_time=moving_time,
    )

def move_grippers(
    bot_list: Sequence[InterbotixManipulatorXS],
    target_pose_list: Sequence[float],
    moving_time: float,
):
    gripper_command = JointSingleCommand(name='gripper')
    num_steps = int(moving_time / DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    zipped_lists = zip(curr_pose_list, target_pose_list)
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zipped_lists
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)

def control_grippers(
    bot_list: Sequence[InterbotixManipulatorXS],
    target_pose_list: Sequence[float],
    curr_pose_list: Sequence[float],
    gripper_command: JointSingleCommand,
    moving_time: float,
):
    num_steps = int(moving_time / DT)
    zipped_lists = zip(curr_pose_list, target_pose_list)
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zipped_lists
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)

def setup_follower_bot(bot: InterbotixManipulatorXS):
    bot.core.robot_reboot_motors('single', 'gripper', True)
    bot.core.robot_set_operating_modes('group', 'arm', 'position')
    bot.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    torque_on(bot)


def setup_leader_bot(bot: InterbotixManipulatorXS):
    bot.core.robot_set_operating_modes('group', 'arm', 'pwm')
    bot.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    torque_off(bot)


def set_standard_pid_gains(bot: InterbotixManipulatorXS):
    bot.core.robot_set_motor_registers('group', 'arm', 'Position_P_Gain', 800)
    bot.core.robot_set_motor_registers('group', 'arm', 'Position_I_Gain', 0)


def set_low_pid_gains(bot: InterbotixManipulatorXS):
    bot.core.robot_set_motor_registers('group', 'arm', 'Position_P_Gain', 100)
    bot.core.robot_set_motor_registers('group', 'arm', 'Position_I_Gain', 0)


def torque_off(bot: InterbotixManipulatorXS):
    bot.core.robot_torque_enable('group', 'arm', False)
    bot.core.robot_torque_enable('single', 'gripper', False)


def torque_on(bot: InterbotixManipulatorXS):
    bot.core.robot_torque_enable('group', 'arm', True)
    bot.core.robot_torque_enable('single', 'gripper', True)
