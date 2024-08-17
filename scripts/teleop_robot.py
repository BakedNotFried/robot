import rclpy
from rclpy.executors import SingleThreadedExecutor
import time
import modern_robotics as mr
import numpy as np

from interbotix_common_modules.common_robot.robot import InterbotixRobotNode
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd
from interbotix_xs_msgs.msg import JointSingleCommand

from omni_msgs.msg import OmniState

import robot.transform_utils as tr
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

class TeleopRobot(InterbotixRobotNode):
    def __init__(self):
        super().__init__(node_name='teleop_robot')
        
        # Init Robot
        self.robot = InterbotixManipulatorXS(
            robot_model='wx250s',
            robot_name='robot',
            node=self,
            iterative_update_fk=False,
        )
        self.robot_gripper_command = JointSingleCommand(name='gripper')
        self.robot_description: mrd.ModernRoboticsDescription = getattr(mrd, "wx250s")

        time.sleep(1)
        # self.opening_ceremony()
        
        # Fields
        self.tele_state = None  # x, y, z, r, p, y, open_gripper, close_gripper
        self.curr_tele_xyz = None
        self.curr_tele_rpy = None
        self.prev_tele_xyz = None
        self.prev_tele_rpy = None
        self.first_run = True

        # Scale Factor. For smoother control
        self.xyz_scale = 2.5

        # Subs
        self.create_subscription(OmniState, '/omni/state', self.tele_state_callback, 1)
        
        # Timer
        hz = 50
        dt = 1/hz
        self.timer = self.create_timer(dt, self.control)
        

    def FKin(self, joint_positions):
        joint_positions = list(joint_positions)
        return mr.FKinSpace(self.robot_description.M, self.robot_description.Slist, joint_positions)
    

    def IKin(self, T_sd, custom_guess=None, execute=True):
        if (custom_guess is None):
            initial_guesses = self.initial_guesses
        else:
            initial_guesses = [custom_guess]

        for guess in initial_guesses:
            theta_list, success = mr.IKinSpace(self.robot_description.Slist, self.robot_description.M, T_sd, guess, 0.001, 0.001)
            solution_found = True

            # Check to make sure a solution was found and that no joint limits were violated
            if success:
                theta_list = [int(elem * 1000)/1000.0 for elem in theta_list]
                for x in range(self.robot.arm.group_info.num_joints):
                    if not (self.robot.arm.group_info.joint_lower_limits[x] <= theta_list[x] <= self.robot.arm.group_info.joint_upper_limits[x]):
                        solution_found = False
                        break
            else:
                solution_found = False

            if solution_found:
                if execute:
                    # Commenting out because I am adding extra
                    # self.publish_positions_fast(theta_list)
                    self.T_sb = T_sd
                return theta_list, True
            else:
                print("Guess failed to converge...")

        print("No valid pose could be found")
        return theta_list, False
    
    def opening_ceremony(self):
        """Move robot arm to start demonstration pose"""
        # reboot gripper motors, and set operating modes for all motors
        self.robot.core.robot_reboot_motors('single', 'gripper', True)
        self.robot.core.robot_set_operating_modes('group', 'arm', 'position')
        self.robot.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
        self.robot.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)
        torque_on(self.robot)
        
        # move arm to starting position
        start_arm_qpos = START_ARM_POSE[:6]
        move_arms(
            [self.robot],
            [start_arm_qpos],
            moving_time=2.0,
        )
        
        # move gripper to starting position
        move_grippers(
            [self.robot],
            [FOLLOWER_GRIPPER_JOINT_MID],
            moving_time=0.5
        )

    def tele_state_callback(self, msg):
        self.tele_state = [msg.pose.position.y,
                           -msg.pose.position.x,
                           msg.pose.position.z,
                           -msg.rpy.z,
                           -msg.rpy.y,
                           -msg.rpy.x,
                           msg.open_gripper.data,
                           msg.close_gripper.data]

    def control(self):
        if self.tele_state:
            # Control
            self.curr_tele_xyz = self.tele_state[0:3]
            self.curr_tele_xyz = [self.xyz_scale * e for e in self.curr_tele_xyz]
            self.curr_tele_rpy = self.tele_state[3:6]

            # Robot State -> 9
            self.robot_state = self.robot.core.joint_states.position
            self.joint_angles = self.robot_state[:6]
            self.gripper_angle = self.robot_state[6]

            # Debug
            print(self.tele_state)
            print(self.joint_angles)
            print(self.gripper_angle)

            if self.first_run:
                print('first run')
                self.control_locked = True
                self.first_run = False
                self.prev_tele_xyz = self.curr_tele_xyz
                self.prev_tele_rpy = self.curr_tele_rpy
                
            else:
                tele_xyz_delta = [curr - prev  for curr, prev in zip(self.curr_tele_xyz, self.prev_tele_xyz)]
                tele_rpy_delta = [curr - prev  for curr, prev in zip(self.curr_tele_rpy, self.prev_tele_rpy)]

                if self.control_locked:
                    print('locked')
                    self.prev_tele_xyz = self.curr_tele_xyz
                    self.prev_tele_rpy = self.curr_tele_rpy
                    return

                if tele_xyz_delta == [0, 0, 0] and tele_rpy_delta == [0, 0, 0]:
                    self.prev_tele_xyz = self.curr_tele_xyz
                    self.prev_tele_rpy = self.curr_tele_rpy
                    return
                
                # Do Stuff
                # X, Y, Z control
                if not (tele_xyz_delta == [0, 0, 0]):
                    current_ee_pose = self.FKin(self.joint_angles, matrix=True)
                    xyz_transform = tr.RpToTrans(tr.eulerAnglesToRotationMatrix([0, 0, 0]), tele_xyz_delta)
                    # move_xyz = xyz_transform.dot(current_ee_pose)
                    move_xyz = np.dot(current_ee_pose, xyz_transform)
                    self.joint_angles, _ = self.IKin(move_xyz, custom_guess=self.joint_angles)

                # Roll, Pitch and Yaw control
                if not (tele_rpy_delta == [0, 0, 0]):
                    joint_angle_delta = [0, 0, 0, tele_rpy_delta[2], tele_rpy_delta[1], tele_rpy_delta[0]]
                    self.joint_angles = [a + b for a, b in zip(self.joint_angles, joint_angle_delta)]

                # Robot State Update
                # self.robot.arm.set_joint_positions(self.joint_angles, blocking=False)

                # Set the prev
                self.prev_tele_xyz = self.curr_tele_xyz
                self.prev_tele_rpy = self.curr_tele_rpy

        else:
            self.get_logger().info('Waiting for OmniState message...')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        teleop_robot = TeleopRobot()
        executor = SingleThreadedExecutor()
        executor.add_node(teleop_robot)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            teleop_robot.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
