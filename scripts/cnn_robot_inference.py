from robot.policy3.model import PolicyCNNMLP

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
import time
import torch
import numpy as np
import cv2
import pdb

from interbotix_common_modules.common_robot.robot import InterbotixRobotNode
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd
from interbotix_xs_msgs.msg import JointSingleCommand

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
        # self.kb = KeyboardInterface()

        # Init Policy
        torch.manual_seed(1337)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)
        np.random.seed(1337)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.device == torch.device("cuda"), "CUDA is not available"
        torch.set_float32_matmul_precision('high')

        self.policy = PolicyCNNMLP()
        self.policy = self.policy.to(self.device)
        use_compile = True
        if use_compile:
            self.policy = torch.compile(self.policy)
        checkpoint_path = '/home/qutrll/data/pot_pick_place_ckpt/1/checkpoint_step_90000_seed_1337.ckpt'
        checkpoint = torch.load(checkpoint_path)
        self.policy.load_state_dict(checkpoint['model_state_dict'])

        # Cameras Setup
        height = 240
        width = 424
        self.overhead_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.field_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.wrist_image = np.zeros((height, width, 3), dtype=np.uint8)

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
            '/cam_wrist/camera/color/image_rect_raw', 
            self.wrist_image_callback, 
            1, 
            callback_group=callback_group
        )

        # Establish the initial joint angles
        self.arm_joint_angles = START_ARM_POSE[:6]
        self.robot.arm.set_joint_positions(START_ARM_POSE[:6], blocking=False)
        self.gripper_joint = ROBOT_GRIPPER_JOINT_MID
        self.robot_gripper_cmd.cmd = self.gripper_joint
        self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)

        # Output. Starts with arm joint angles, gripper joint, and progress
        self.output = self.arm_joint_angles + [self.gripper_joint] + [0.0]

        # Preallocate tensors
        self.images_tensor = torch.empty(1, 3, 3, height, width, dtype=torch.float32, device=self.device)
        self.q_pos_tensor = torch.empty(1, 7, dtype=torch.float32, device=self.device)
        self.progress_tensor = torch.empty(1, 1, dtype=torch.float32, device=self.device)

        # Control Loop Timer
        hz = 10
        dt = 1/hz
        self.control_timer = self.create_timer(dt, self.robot_control)


    def robot_control(self):
        self.display_images()

        # Convert images to tensors and normalize
        self.images_tensor[0, 0] = torch.from_numpy(self.overhead_image).float().permute(2, 0, 1) / 255.0
        self.images_tensor[0, 1] = torch.from_numpy(self.field_image).float().permute(2, 0, 1) / 255.0
        self.images_tensor[0, 2] = torch.from_numpy(self.wrist_image).float().permute(2, 0, 1) / 255.0

        # Convert prev action to tensor
        self.q_pos_tensor[0] = torch.tensor(self.output[:7], dtype=torch.float32, device=self.device)
        self.progress_tensor[0] = torch.tensor(self.output[7], dtype=torch.float32, device=self.device)

        # Forward pass
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                output = self.policy(self.images_tensor, self.q_pos_tensor, self.progress_tensor)
        
        # Convert to numpy, handling BFloat16
        output = output.float().cpu().numpy().squeeze()
        self.output = output.tolist()

        joints = self.output[:6]
        gripper = self.output[6]
        progress = self.output[7]

        input('Press Enter to continue...')

        # Control
        self.robot_gripper_cmd.cmd = gripper
        self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)
        self.robot.arm.set_joint_positions(joints, blocking=False)
        print(f"Progress: {progress}\n")


    def process_image(self, msg):
        return np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    
    def display_images(self):
        combined_image = np.hstack((self.overhead_image, self.field_image, self.wrist_image))
        cv2.imshow('Camera Images', combined_image)
        cv2.waitKey(1)

    def overhead_image_callback(self, msg):
        self.overhead_image = self.process_image(msg)

    def field_image_callback(self, msg):
        self.field_image = self.process_image(msg)
        
    def wrist_image_callback(self, msg):
        self.wrist_image = self.process_image(msg)

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

