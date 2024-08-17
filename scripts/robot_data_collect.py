import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
import os
import time
import modern_robotics as mr
import numpy as np
import cv2
import h5py

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
)
from robot.robot_utils import (
    move_arms,
    torque_on,
)

# Record States
CONTROL = 0
RECORDING = 1
SAVE = 2
DISCARD = 3
IDLE = 4

DEBUG = True

class RobotDataCollector(InterbotixRobotNode):
    def __init__(self):
        super().__init__(node_name='robot_data_collector')
        
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
        
        # Keyboard Interface. For lock/unlock control via a/s keys and control of recording
        self.kb = KeyboardInterface()

        # Fields
        self.tele_state = None  # x, y, z, r, p, y, open_gripper, close_gripper
        self.gripper_delta = 0.05   # For altering gripper on update
        self.first_run = True

        # Scale Factor. For smoother control
        self.xyz_scale = 2.5

        # Establish the initial joint angless
        self.arm_joint_angles = START_ARM_POSE[:6]
        self.robot.arm.set_joint_positions(START_ARM_POSE[:6], blocking=False)
        self.gripper_joint = ROBOT_GRIPPER_JOINT_MID
        self.robot_gripper_cmd.cmd = self.gripper_joint
        self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)

        # Omni Teleoperation State Callback
        self.create_subscription(OmniState, '/omni/state', self.tele_state_callback, 1)

        # Cameras Setup
        self.field_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.overhead_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.wrist_image = np.zeros((480, 640, 3), dtype=np.uint8)

        callback_group = ReentrantCallbackGroup()
        self.create_subscription(
            Image, 
            '/cam_overhead/camera/color/image_raw', 
            self.overhead_image_callback, 
            1, 
            callback_group=callback_group
        )
        self.create_subscription(
            Image, 
            '/cam_field/camera/color/image_raw', 
            self.field_image_callback, 
            1, 
            callback_group=callback_group
        )
        self.create_subscription(
            Image, 
            '/cam_wrist/camera/color/image_raw', 
            self.wrist_image_callback, 
            1, 
            callback_group=callback_group
        )

        # Control Rate
        control_rate = 50.0
        control_ts = 1.0/control_rate

        # Sampling Rate
        sample_rate = 10.0  # Hz
        self.sample_tick = control_rate / sample_rate
        self.sample_count = 0

        # Recording Info
        DATA_DIR = "/home/qutrll/data/"
        save_dir = "test_data/"
        self.save = True
        if self.save:
            self.save_dir = DATA_DIR + save_dir
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # Save Data. q_pos is joint angles + gripper joint. action is same but t + 1
        self.data_dict = {
            'q_pos': [],
            'action': [],
        }
        self.im_oh = []
        self.im_field = []
        self.im_wrist = []

        # Control Loop Timer
        self.teleop_control_timer = self.create_timer(control_ts, self.teleop_control)


    def save_data(self):
        if self.save:
            print("Saving Data")

            # Get the next available file number
            existing_files = [f for f in os.listdir(self.save_dir) if f.endswith('.hdf5')]
            next_file_num = len(existing_files)

            # Create the HDF5 file with the next available number
            hdf5_path = os.path.join(self.save_dir, f'{next_file_num}.hdf5')

            with h5py.File(hdf5_path, 'w') as f:
                # Save state data
                f.create_dataset('q_pos', data=np.array(self.data_dict['q_pos']))
                f.create_dataset('action', data=np.array(self.data_dict['action']))

                # Save images
                f.create_dataset('oh_images', data=np.array(self.im_oh))
                f.create_dataset('wrist_images', data=np.array(self.im_field))
                f.create_dataset('field_images', data=np.array(self.im_wrist))

                # Create progress dataset
                demo_length = len(self.data_dict['q_pos'])
                progress = np.linspace(0, 1, demo_length, dtype=np.float32)
                f.create_dataset('progress', data=progress)

            print(f"Data Saved to HDF5 file: {hdf5_path}")
            print(f"Data of length: {demo_length}")
        else:
            print("Data Not Saved")
        
        self.reset_data()


    def reset_data(self):
        self.data_dict = {
            'q_pos': [],
            'action': [],
        }
        self.im_oh = []
        self.im_field = []
        self.im_wrist = []


    def process_image(self, msg):
        return np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    
    def display_images(self):
        combined_image = np.hstack((self.field_image, self.overhead_image, self.wrist_image))
        cv2.imshow('Camera Images', combined_image)
        cv2.waitKey(1)

    def overhead_image_callback(self, msg):
        self.overhead_image = self.process_image(msg)

    def field_image_callback(self, msg):
        self.field_image = self.process_image(msg)
        
    def wrist_image_callback(self, msg):
        self.wrist_image = self.process_image(msg)


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


    def reset_robot_orientation(self):
        # move arm to starting position
        start_arm_qpos = START_ARM_POSE[:6]
        move_arms(
            [self.robot],
            [start_arm_qpos],
            moving_time=1.5,
        )

        # Establish the initial joint angless
        self.arm_joint_angles = START_ARM_POSE[:6]
        self.robot.arm.set_joint_positions(START_ARM_POSE[:6], blocking=False)
        self.gripper_joint = ROBOT_GRIPPER_JOINT_MID
        self.robot_gripper_cmd.cmd = self.gripper_joint
        self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)
        
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
        # Update Keyboard Interface every tick
        self.kb.prev_record_state = self.kb.record_state
        self.kb.update()

        if DEBUG:
            self.display_images()
        
        if (self.kb.record_state == SAVE) and (self.kb.prev_record_state == RECORDING):
            # Save Data
            self.save_data()
            # Reset Robot Orientation
            self.kb.prev_record_state = IDLE
            self.reset_robot_orientation()
        elif (self.kb.record_state == DISCARD) and (self.kb.prev_record_state == RECORDING):
            print("Discarding Data")
            # Discard Data
            # Reset Robot Orientation
            self.kb.prev_record_state = IDLE
            self.reset_data()
            self.reset_robot_orientation()

        if self.kb.record_state == IDLE:
            return

        if (self.kb.record_state == CONTROL) or (self.kb.record_state == RECORDING):
            if self.tele_state:
                # Get the current teleoperate states. 
                curr_tele_xyz = self.tele_state[0:3]
                curr_tele_xyz = [self.xyz_scale * e for e in curr_tele_xyz]
                curr_tele_rpy = self.tele_state[3:6]
                open_gripper = self.tele_state[6]
                close_gripper = self.tele_state[7]

                # On the first run, set the previous states and do nothing
                if self.first_run:
                    print('first run')
                    self.prev_tele_xyz = curr_tele_xyz
                    self.prev_tele_rpy = curr_tele_rpy
                    self.first_run = False
                    return
                
                # Check Lock. Updated via KeyboardInterface
                if self.kb.lock_robot:
                    self.prev_tele_xyz = curr_tele_xyz
                    self.prev_tele_rpy = curr_tele_rpy

                    # Record Data
                    if (self.sample_count % self.sample_tick) == 0 and self.kb.record_state == RECORDING:
                        self.data_dict['q_pos'].append(self.arm_joint_angles + [self.gripper_joint])
                        self.data_dict['action'].append(self.arm_joint_angles + [self.gripper_joint])
                        self.im_oh.append(self.overhead_image.copy())
                        self.im_field.append(self.field_image.copy())
                        self.im_wrist.append(self.wrist_image.copy())
                    self.sample_count += 1

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

                    # Record Data
                    if (self.sample_count % self.sample_tick) == 0 and self.kb.record_state == RECORDING:
                        self.data_dict['q_pos'].append(self.arm_joint_angles + [self.gripper_joint])
                        self.data_dict['action'].append(self.arm_joint_angles + [self.gripper_joint])
                        self.im_oh.append(self.overhead_image.copy())
                        self.im_field.append(self.field_image.copy())
                        self.im_wrist.append(self.wrist_image.copy())
                    self.sample_count += 1

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

                # Record Data
                if (self.sample_count % self.sample_tick) == 0 and self.kb.record_state == RECORDING:
                    self.data_dict['q_pos'].append(self.arm_joint_angles + [self.gripper_joint])
                    self.data_dict['action'].append(self.arm_joint_angles + [self.gripper_joint])
                    self.im_oh.append(self.overhead_image.copy())
                    self.im_field.append(self.field_image.copy())
                    self.im_wrist.append(self.wrist_image.copy())
                self.sample_count += 1

            else:
                self.get_logger().info('Waiting for OmniState message...')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        data_collector = RobotDataCollector()
        executor = SingleThreadedExecutor()
        executor.add_node(data_collector)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            data_collector.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
