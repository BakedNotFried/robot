import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
import time
import torch
import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
from collections import deque

# HIT
import json

from HIT.model_util import make_policy
import random
import pdb
# HIT

# Visual Encoder for Feature Comparison
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Robot
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
    control_arms,
    control_grippers
)

def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

class RobotInference(InterbotixRobotNode):
    def __init__(self):
        super().__init__(node_name='robot_inference')
        
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

        # Policy Setup
        policy_class = 'HIT'
        config_dir = "/home/qutrll/data/checkpoints/HIT/pot_pick_place/2/_pot_pick_place_HIT_resnet18_True/all_configs.json"
        config = json.load(open(config_dir))
        policy_config = config['policy_config']

        seed_everything(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.device == torch.device("cuda"), "CUDA is not available"

        self.policy = make_policy(policy_class, policy_config)
        self.policy.eval()
        loading_status = self.policy.deserialize(torch.load("/home/qutrll/data/checkpoints/HIT/pot_pick_place/2/_pot_pick_place_HIT_resnet18_True/policy_step_70000_seed_42.ckpt", map_location='cuda'))
        if not loading_status:
            print(f'Failed to load policy_last.ckpt')
        self.policy.cuda()

        # Init Visual Encoder
        weights = models.VGG19_Weights.IMAGENET1K_V1
        self.encoder = models.vgg19(weights=weights)
        self.encoder.classifier = nn.Sequential(*list(self.encoder.classifier.children())[:-1])
        self.encoder = self.encoder.to(self.device)

        # Create MSE Loss
        self.loss_fn = nn.MSELoss()

        # Cameras Setup
        self.height = 240
        self.width = 424
        self.overhead_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.field_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.wrist_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.predicted_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # self.create_subscription(
        #     Image, 
        #     '/cam_overhead/camera/color/image_raw', 
        #     self.overhead_image_callback, 
        #     1, 
        # )
        self.create_subscription(
            Image, 
            '/cam_field/camera/color/image_raw', 
            self.field_image_callback, 
            1, 
        )
        # self.create_subscription(
        #     Image, 
        #     '/cam_wrist/camera/color/image_rect_raw', 
        #     self.wrist_image_callback, 
        #     1, 
        # )

        # Establish the initial joint angles
        self.arm_joint_angles = START_ARM_POSE[:6]
        self.robot.arm.set_joint_positions(START_ARM_POSE[:6], blocking=False)
        self.gripper_joint = ROBOT_GRIPPER_JOINT_MID
        self.robot_gripper_cmd.cmd = self.gripper_joint
        self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)

        # Output. Starts with arm joint angles, gripper joint, and progress
        self.output = self.arm_joint_angles + [self.gripper_joint] + [0.0]

        # Preallocate tensors
        self.images_tensor = torch.empty(1, 1, 3, self.height, self.width, dtype=torch.float32, device=self.device)
        self.q_pos_tensor = torch.empty(1, 7, dtype=torch.float32, device=self.device)
        self.next_image = None

        # Setup Plotting
        # Initialize deques to store data for plotting
        self.mse_loss_data = deque(maxlen=100)
        self.progress_data = deque(maxlen=100)
        self.progress_gradient_data = deque(maxlen=100)
        self.x_data = deque(maxlen=100)
        self.frame_count = 0

        # Set up the plot
        self.use_displays = False
        if self.use_displays:
            plt.ion()
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 15))
            self.fig.tight_layout(pad=3.0)

            # Initialize lines for each plot
            self.mse_line, = self.ax1.plot([], [], 'r-')
            self.progress_line, = self.ax2.plot([], [], 'b-')
            self.gradient_line, = self.ax3.plot([], [], 'g-')

            # Set up the axes
            self.ax1.set_title('MSE Loss')
            self.ax2.set_title('Progress')
            self.ax3.set_title('Progress Gradient')

            for ax in (self.ax1, self.ax2, self.ax3):
                ax.set_xlim(0, 100)
                ax.grid(True)

            self.ax1.set_ylim(0, 1)
            self.ax2.set_ylim(0, 1)
            self.ax3.set_ylim(-0.1, 0.1)


    def process_image(self, msg):
        return np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    

    def update_plots(self, mse_loss, progress):
        self.frame_count += 1
        self.x_data.append(self.frame_count)
        
        self.mse_loss_data.append(mse_loss)
        self.progress_data.append(progress)
        
        # Calculate progress gradient
        if len(self.progress_data) > 1:
            gradient = self.progress_data[-1] - self.progress_data[-2]
        else:
            gradient = 0
        self.progress_gradient_data.append(gradient)

        # Ensure all data lists have the same length
        min_length = min(len(self.x_data), len(self.mse_loss_data), len(self.progress_data), len(self.progress_gradient_data))
        x_data = list(self.x_data)[-min_length:]
        mse_data = list(self.mse_loss_data)[-min_length:]
        progress_data = list(self.progress_data)[-min_length:]
        gradient_data = list(self.progress_gradient_data)[-min_length:]

        # Update the plot data
        self.mse_line.set_data(x_data, mse_data)
        self.progress_line.set_data(x_data, progress_data)
        self.gradient_line.set_data(x_data, gradient_data)

        # Adjust x-axis limit if necessary
        if self.frame_count >= 100:
            for ax in (self.ax1, self.ax2, self.ax3):
                ax.set_xlim(self.frame_count - 99, self.frame_count)

        # Redraw the plot
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Error updating plot: {e}")


    def display_images(self):
        # Resize the predicted image to match the field image size
        resized_predicted_image = cv2.resize(self.predicted_image, (self.width, self.height))
        
        # Create a combined image with field image and predicted image side by side
        combined_image = np.hstack((cv2.cvtColor(self.field_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(resized_predicted_image, cv2.COLOR_BGR2RGB)))
        
        # Add labels to the images
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_image, 'Actual', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_image, 'Imagined', (self.width + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add progress bar
        progress = self.output[7]  # Assuming progress is the last element in self.output
        bar_width = self.width * 2  # Full width of the combined image
        bar_height = 20
        bar_top = combined_image.shape[0] - bar_height - 10  # 10 pixels from the bottom
        
        # Draw the background of the progress bar
        cv2.rectangle(combined_image, (0, bar_top), (bar_width, bar_top + bar_height), (100, 100, 100), -1)
        
        # Draw the filled part of the progress bar
        filled_width = int(bar_width * progress / 1.0)
        cv2.rectangle(combined_image, (0, bar_top), (filled_width, bar_top + bar_height), (0, 255, 0), -1)
        
        # Add progress text
        progress_text = f'Progress: {progress:.2f}/1.0'
        cv2.putText(combined_image, progress_text, (10, bar_top - 10), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Update the plots
        mse_loss = self.mse_loss_data[-1] if self.mse_loss_data else 0
        progress = self.output[7]
        self.update_plots(mse_loss, progress)
        
        cv2.imshow('Current and Predicted Images', combined_image)
        cv2.waitKey(1)

    def overhead_image_callback(self, msg):
        self.overhead_image = self.process_image(msg)

    def field_image_callback(self, msg):
        self.field_image = self.process_image(msg)

        if self.use_displays and self.next_image is not None:
            self.display_images()

        # Convert images to tensors and normalize
        # self.images_tensor[0, 0] = torch.from_numpy(self.overhead_image).float().permute(2, 0, 1) / 255.0
        self.images_tensor[0, 0] = torch.from_numpy(self.field_image).float().permute(2, 0, 1) / 255.0
        reshaped_tensor = self.images_tensor.squeeze(1)
        self.reshaped_images_tensor = F.interpolate(reshaped_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        self.reshaped_images_tensor = self.reshaped_images_tensor.unsqueeze(0)
        # self.images_tensor[0, 2] = torch.from_numpy(self.wrist_image).float().permute(2, 0, 1) / 255.0

        # Convert prev action to tensor
        self.q_pos_tensor[0] = torch.tensor(self.output[:7], dtype=torch.float32, device=self.device)

        # VGG Encoder Forward Pass
        if self.next_image is not None:
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    # Resize images to match VGG input size (224x224)
                    current_image_resized = F.interpolate(self.images_tensor[:, 0], size=(224, 224), mode='bilinear', align_corners=False)
                    # next_image_resized = F.interpolate(self.next_image, size=(224, 224), mode='bilinear', align_corners=False)

                    # Encode images
                    current_image_encoded = self.encoder(current_image_resized)
                    next_image_encoded = self.encoder(self.next_image.squeeze(1))

            # Calculate MSE loss between encoded images
            mse_loss = self.loss_fn(current_image_encoded, next_image_encoded)
            self.mse_loss_data.append(mse_loss.item())

        # Policy Forward Pass
        with torch.no_grad():
            output, hs_img = self.policy.forward_inf(self.q_pos_tensor, self.reshaped_images_tensor)

        # World Model Forward Pass
        with torch.no_grad():
            self.next_image = self.policy.forward_world_model(hs_img, output)

        # For display
        self.predicted_image = self.next_image.squeeze().float().permute(1, 2, 0).cpu().numpy()
        self.predicted_image = (self.predicted_image * 255).astype(np.uint8)

        # Convert from torch, handling BFloat16
        output = output.float().cpu().numpy().squeeze()
        selection_index = 5
        output = output[selection_index]
        self.output = output.tolist()
        joints = self.output[:6]
        gripper = self.output[6]

        input('Press Enter to continue...')

        # Control Robot
        self.robot_gripper_cmd.cmd = gripper
        self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)
        control_arms(
            [self.robot],
            [joints],
            [self.arm_joint_angles],
            moving_time=0.4,
        )
        self.arm_joint_angles = joints
        
    def wrist_image_callback(self, msg):
        self.wrist_image = self.process_image(msg)

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
    
    def __del__(self):
        plt.close(self.fig)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        robot_teleop = RobotInference()
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

