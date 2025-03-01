import rclpy
from rclpy.executors import SingleThreadedExecutor
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
import os
import time
import modern_robotics as mr
import numpy as np
import cv2
import h5py
import torch
import torch.nn.functional as F

from interbotix_common_modules.common_robot.robot import InterbotixRobotNode
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd
from interbotix_xs_msgs.msg import JointSingleCommand

from omni_msgs.msg import OmniState

import robot.transform_utils as tr
from robot.keyboard_interface import KeyboardInterface

from robot.world_model.model import WorldModelCNN

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

        # Init World Model
        torch.manual_seed(1337)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)
        np.random.seed(1337)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.device == torch.device("cuda"), "CUDA is not available"
        torch.set_float32_matmul_precision('high')

        self.world_model = WorldModelCNN()
        self.world_model = self.world_model.to(self.device)
        use_compile = True
        if use_compile:
            self.world_model = torch.compile(self.world_model)
        checkpoint_path = '.ckpt'
        checkpoint = torch.load(checkpoint_path)
        self.world_model.load_state_dict(checkpoint['model_state_dict'])

        # Fields
        self.gripper_delta = 0.05   # For altering gripper on update
        self.first_run = True
        self.prev_tele_xyz = None
        self.prev_tele_rpy = None

        # Scale Factor. For smoother control
        self.xyz_scale = 6

        # Establish the initial joint angles
        self.arm_joint_angles = START_ARM_POSE[:6]
        self.prev_joint_angles = self.arm_joint_angles
        self.robot.arm.set_joint_positions(START_ARM_POSE[:6], blocking=False)
        self.gripper_joint = ROBOT_GRIPPER_JOINT_MID
        self.prev_gripper_joint = self.gripper_joint
        self.robot_gripper_cmd.cmd = self.gripper_joint
        self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)
        self.progress = 0.0

        # Cameras Setup
        height = 240
        width = 424
        self.field_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.overhead_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.wrist_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Set up subscribers for image topics
        self.overhead_sub = Subscriber(self, Image, '/cam_overhead/camera/color/image_raw')
        self.field_sub = Subscriber(self, Image, '/cam_field/camera/color/image_raw')
        self.wrist_sub = Subscriber(self, Image, '/cam_wrist/camera/color/image_rect_raw')

        # Set up subscriber for OmniState
        # Control Callback based on Omni Teleoperation State. Check Omni State for publish rate - currently 60Hz
        self.omni_sub = Subscriber(self, OmniState, '/omni/state')

        # Set up ApproximateTimeSynchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.overhead_sub, self.field_sub, self.wrist_sub, self.omni_sub],
            queue_size=3,
            slop=0.015,
            allow_headerless=False
        )
        self.ts.registerCallback(self.synchronized_callback)

        # Preallocate tensors
        self.images_tensor = torch.empty(1, 1, 3, height, width, dtype=torch.float32, device=self.device)
        self.action_tensor = torch.empty(1, 7, dtype=torch.float32, device=self.device)
        self.q_pos_tensor = torch.empty(1, 7, dtype=torch.float32, device=self.device)
        self.progress_tensor = torch.empty(1, 1, dtype=torch.float32, device=self.device)


    def synchronized_callback(self, overhead_msg, field_msg, wrist_msg, omni_msg):
        # Process images
        self.overhead_image = self.process_image(overhead_msg)
        self.field_image = self.process_image(field_msg)
        self.wrist_image = self.process_image(wrist_msg)

        # Process OmniState
        self.robot_control_callback(omni_msg)

        if DEBUG:
            self.display_images()


    def robot_startup(self):
        """Move robot arm to start demonstration pose"""
        # reboot gripper motors, and set operating modes for all motors
        self.robot.core.robot_reboot_motors('single', 'gripper', True)
        self.robot.core.robot_set_operating_modes('group', 'arm', 'position')
        self.robot.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
        self.robot.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 500)
        torque_on(self.robot)
        
        # move arm to starting position
        start_arm_qpos = START_ARM_POSE[:6]
        move_arms(
            [self.robot],
            [start_arm_qpos],
            moving_time=1.5,
        )


    def robot_control_callback(self, msg):
            tele_state = [msg.pose.position.y,
                            -msg.pose.position.x,
                            msg.pose.position.z,
                            -msg.rpy.z,
                            -msg.rpy.y,
                            -msg.rpy.x,
                            msg.open_gripper.data,
                            msg.close_gripper.data]
            
            # Update Keyboard Interface every tick
            self.kb.prev_record_state = self.kb.record_state
            self.kb.update()

            if (self.kb.record_state == RECORDING) or (self.kb.record_state == CONTROL):
                # Get the current teleoperate states. 
                curr_tele_xyz = tele_state[0:3]
                curr_tele_xyz = [self.xyz_scale * e for e in curr_tele_xyz]
                curr_tele_rpy = tele_state[3:6]
                open_gripper = tele_state[7]
                close_gripper = tele_state[6]

                # On the first run, set the previous states and do nothing
                if self.first_run:
                    print('first run')
                    self.prev_tele_xyz = curr_tele_xyz
                    self.prev_tele_rpy = curr_tele_rpy
                    self.first_run = False

                    return
                
                # Check Lock. Updated via KeyboardInterface
                if self.kb.lock_robot:
                    # Set the prev
                    self.prev_tele_xyz = curr_tele_xyz
                    self.prev_tele_rpy = curr_tele_rpy
                    self.prev_joint_angles = self.arm_joint_angles
                    self.prev_gripper_joint = self.gripper_joint
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
                    # Set the prev
                    self.prev_tele_xyz = curr_tele_xyz
                    self.prev_tele_rpy = curr_tele_rpy
                    self.prev_joint_angles = self.arm_joint_angles
                    self.prev_gripper_joint = self.gripper_joint
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
                
                # World Model Inference
                # Allocate tensors
                self.images_tensor[0, 0] = torch.from_numpy(self.field_image).float().permute(2, 0, 1) / 255.0
                self.images_tensor[0, 0] = F.interpolate(self.images_tensor[0, 0], size=(224, 224), mode='bilinear', align_corners=False)
                self.action_tensor[0] = torch.tensor(self.arm_joint_angles + [self.gripper_joint], dtype=torch.float32, device=self.device)
                self.q_pos_tensor[0] = torch.tensor(self.prev_joint_angles + [self.prev_gripper_joint], dtype=torch.float32, device=self.device)
                self.progress_tensor[0] = torch.tensor(self.progress, dtype=torch.float32, device=self.device)
                # Forward pass
                with torch.no_grad():
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        next_image = self.world_model(self.images_tensor, self.action_tensor, self.q_pos_tensor, self.progress_tensor)

                # Robot State Update
                self.robot.arm.set_joint_positions(self.arm_joint_angles, blocking=False)

                # Set the prev
                self.prev_tele_xyz = curr_tele_xyz
                self.prev_tele_rpy = curr_tele_rpy
                self.prev_joint_angles = self.arm_joint_angles
                self.prev_gripper_joint = self.gripper_joint
                self.progress += 0.01

            # elif (self.kb.record_state == SAVE) and (self.kb.prev_record_state == RECORDING):
            #     # Save Data
            #     self.save_data()
            #     # Reset Robot Orientation
            #     self.kb.prev_record_state = IDLE
            #     self.reset_robot_orientation()
            # elif (self.kb.record_state == DISCARD) and (self.kb.prev_record_state == RECORDING):
            #     print("Discarding Data")
            #     # Discard Data
            #     self.reset_data()
            #     # Reset Robot Orientation
            #     self.kb.prev_record_state = IDLE
            #     self.reset_robot_orientation()
            elif self.kb.record_state == IDLE:
                return
        

    # def save_data(self):
    #     if self.save:
    #         print("Saving Data")

    #         # Get the next available file number
    #         existing_files = [f for f in os.listdir(self.save_dir) if f.endswith('.hdf5')]
    #         next_file_num = len(existing_files)

    #         # Create the HDF5 file with the next available number
    #         hdf5_path = os.path.join(self.save_dir, f'{next_file_num}.hdf5')

    #         with h5py.File(hdf5_path, 'w') as f:
    #             # Save state data
    #             f.create_dataset('q_pos', data=np.array(self.data_dict['q_pos']))
    #             f.create_dataset('action', data=np.array(self.data_dict['action']))

    #             # Save images
    #             f.create_dataset('oh_images', data=np.array(self.im_oh))
    #             f.create_dataset('field_images', data=np.array(self.im_field))
    #             f.create_dataset('wrist_images', data=np.array(self.im_wrist))
    #         print(f"Data Saved to HDF5 file: {hdf5_path}")
    #     else:
    #         print("Data Not Saved")
        
    #     self.reset_data()


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
