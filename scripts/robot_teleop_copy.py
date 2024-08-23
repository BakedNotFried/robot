import rclpy
from rclpy.executors import SingleThreadedExecutor
import time
import modern_robotics as mr

from interbotix_common_modules.common_robot.robot import InterbotixRobotNode
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd
from interbotix_xs_msgs.msg import JointSingleCommand

from omni_msgs.msg import OmniState

import robot.transform_utils as tr
from robot.keyboard_interface import KeyboardInterface

from robot.constants import (
    ROBOT_GRIPPER_JOINT_OPEN,
    ROBOT_GRIPPER_JOINT_CLOSE_MAX,
    ROBOT_GRIPPER_JOINT_MID,
    START_ARM_POSE,
    SLEEP_ARM_POSE
)
from robot.robot_utils import (
    move_arms,
    torque_on,
)

class RobotTeleop(InterbotixRobotNode):
    def __init__(self):
        super().__init__(node_name='robot_teleop')
        
        # Init Robot
        self.robot = InterbotixManipulatorXS(
            robot_model='wx250s',
            robot_name='robot',
            node=self,
            iterative_update_fk=False,
        )
        self.robot_description: mrd.ModernRoboticsDescription = getattr(mrd, "wx250s")
        self.robot_gripper_cmd = JointSingleCommand(name='gripper')

        self.robot_startup()
        
        # Keyboard Interface. For lock/unlock control via a/s keys
        self.kb = KeyboardInterface()

        # Fields
        self.tele_state = None  # x, y, z, r, p, y, open_gripper, close_gripper
        self.gripper_delta = 0.05   # For altering gripper on update
        self.first_run = True
        self.prev_tele_xyz = None
        self.prev_tele_rpy = None

        # Scale Factor. For smoother control
        self.xyz_scale = 6

        # Establish the initial joint angless
        self.arm_joint_angles = START_ARM_POSE[:6]
        self.robot.arm.set_joint_positions(START_ARM_POSE[:6], blocking=False)
        self.gripper_joint = ROBOT_GRIPPER_JOINT_MID
        self.robot_gripper_cmd.cmd = self.gripper_joint
        self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)

        # Omni Teleoperation State Callback
        self.create_subscription(OmniState, '/omni/state', self.tele_state_callback, 1)
        
        # Control Loop Timer
        hz = 50
        dt = 1/hz
        self.teleop_control_timer = self.create_timer(dt, self.teleop_control)
        

    def FKin(self, joint_positions):
        # Returns SE(3) Pose Matrix
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
                    self.T_sb = T_sd
                return theta_list, True
            else:
                print("Guess failed to converge...")

        print("No valid pose could be found")
        return theta_list, False
    

    def robot_startup(self):
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
            moving_time=1.5,
        )

    def robot_sleep(self):
        # move arm to sleep pose
        start_arm_qpos = SLEEP_ARM_POSE[:6]
        move_arms(
            [self.robot],
            [start_arm_qpos],
            moving_time=2,
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


    def teleop_control(self):
        if self.tele_state:
            # Get the current teleoperate states. 
            curr_tele_xyz = self.tele_state[0:3]
            curr_tele_xyz = [self.xyz_scale * e for e in curr_tele_xyz]
            curr_tele_rpy = self.tele_state[3:6]
            close_gripper = self.tele_state[6]
            open_gripper = self.tele_state[7]

            # On the first run, set the previous states and do nothing
            if self.first_run:
                print('first run')
                self.prev_tele_xyz = curr_tele_xyz
                self.prev_tele_rpy = curr_tele_rpy
                self.first_run = False
                return
            
            # Update Lock/Unlock of the arm via keyboard interface
            self.kb.update()
            if self.kb.lock_robot:
                print('locked')
                self.prev_tele_xyz = curr_tele_xyz
                self.prev_tele_rpy = curr_tele_rpy
                return
            
            # Gripper Control
            if open_gripper:
                self.gripper_joint += self.gripper_delta
                if self.gripper_joint > ROBOT_GRIPPER_JOINT_OPEN:
                    self.gripper_joint = ROBOT_GRIPPER_JOINT_OPEN
                self.robot_gripper_cmd.cmd = self.gripper_joint
                self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)
            if close_gripper:
                self.gripper_joint -= self.gripper_delta
                if self.gripper_joint < ROBOT_GRIPPER_JOINT_CLOSE_MAX:
                    self.gripper_joint = ROBOT_GRIPPER_JOINT_CLOSE_MAX
                self.robot_gripper_cmd.cmd = self.gripper_joint
                self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)

            # Calculate deltas
            tele_xyz_delta = [curr - prev  for curr, prev in zip(curr_tele_xyz, self.prev_tele_xyz)]
            tele_rpy_delta = [curr - prev  for curr, prev in zip(curr_tele_rpy, self.prev_tele_rpy)]

            if tele_xyz_delta == [0, 0, 0] and tele_rpy_delta == [0, 0, 0]:
                self.prev_tele_xyz = curr_tele_xyz
                self.prev_tele_rpy = curr_tele_rpy
                return
            
            # X, Y, Z control
            if not (tele_xyz_delta == [0, 0, 0]):
                current_ee_pose = self.FKin(self.arm_joint_angles)
                xyz_transform = tr.RpToTrans(tr.eulerAnglesToRotationMatrix([0, 0, 0]), tele_xyz_delta)
                move_xyz = xyz_transform.dot(current_ee_pose)
                # move_xyz = np.dot(current_ee_pose, xyz_transform)
                self.arm_joint_angles, _ = self.IKin(move_xyz, custom_guess=self.arm_joint_angles)

            # Roll, Pitch and Yaw control
            if not (tele_rpy_delta == [0, 0, 0]):
                joint_angle_delta = [0, 0, 0, tele_rpy_delta[2], tele_rpy_delta[1], tele_rpy_delta[0]]
                self.arm_joint_angles = [a + b for a, b in zip(self.arm_joint_angles, joint_angle_delta)]

            # Robot State Update
            self.robot.arm.set_joint_positions(self.arm_joint_angles, blocking=False)

            # Set the prev
            self.prev_tele_xyz = curr_tele_xyz
            self.prev_tele_rpy = curr_tele_rpy

        else:
            self.get_logger().info('Waiting for OmniState message...')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        robot_teleop = RobotTeleop()
        executor = SingleThreadedExecutor()
        executor.add_node(robot_teleop)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            robot_teleop.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
