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
import random

# Policy + World Model
from robot.policy3.model import PolicyCNNMLP
from robot.policy3.world_model import WorldModel

# For visual encoder
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from robot.keyboard_interface import KeyboardInterface

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

        # Set Seeds
        seed_everything(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.device == torch.device("cuda"), "CUDA is not available"
        torch.set_float32_matmul_precision('high')

        # Init Policy
        checkpoint_path = '/home/qutrll/data/checkpoints/cnn_mlp/1/checkpoint_step_150000_seed_42.ckpt'
        checkpoint = torch.load(checkpoint_path)
        self.policy = PolicyCNNMLP()
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        use_compile = False
        if use_compile:
            self.policy = torch.compile(self.policy)
        self.policy.load_state_dict(checkpoint['model_state_dict'])

        # Init World Model
        self.world_model = WorldModel(latent_dim=1024, action_dim=80)
        self.world_model = self.world_model.to(self.device)
        self.world_model.eval()
        if use_compile:
            self.world_model = torch.compile(self.world_model)
        self.world_model.load_state_dict(checkpoint['world_model_state_dict'])

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
        self.progress = None

        # Setup Plotting
        # Initialize deques to store data for plotting
        self.mse_loss_data = deque(maxlen=100)
        self.normalized_mse_loss_data = deque(maxlen=100)
        self.smoothed_mse_loss_data = deque(maxlen=100)
        self.progress_data = deque(maxlen=100)
        self.progress_gradient_data = deque(maxlen=100)
        self.progress_gradient_variance_data = deque(maxlen=100)
        self.progress_second_derivative_data = deque(maxlen=100)
        self.x_data = deque(maxlen=100)
        self.frame_count = 0
        # Variable for max MSE loss
        self.max_mse_loss = -np.inf
        self.variance_amplification = 10000

        # Set up the plot
        self.use_displays = True
        self.use_control = False
        if self.use_displays:
            plt.ion()
            self.fig, (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5) = plt.subplots(5, 1, figsize=(10, 25))
            self.fig.tight_layout(pad=3.0)

            # Initialize lines for each plot
            # self.mse_line, = self.ax1.plot([], [], 'r-', label='Raw MSE')
            # self.normalized_mse_line, = self.ax1.plot([], [], 'g-', label='Normalized MSE')
            self.smoothed_mse_line, = self.ax1.plot([], [], 'b-', label='Smoothed Normalized MSE')
            self.progress_line, = self.ax2.plot([], [], 'b-')
            self.gradient_line, = self.ax3.plot([], [], 'g-')
            self.variance_line, = self.ax5.plot([], [], 'r-')
            self.second_derivative_line, = self.ax4.plot([], [], 'm-')

            # Set up the axes
            self.ax1.set_title('MSE Loss')
            self.ax1.legend()
            self.ax2.set_title('Progress')
            self.ax3.set_title('Progress Gradient')
            self.ax4.set_title('Progress Gradient Gradient')
            self.ax5.set_title('Progress Gradient Variance')

            for ax in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5):
                ax.set_xlim(0, 100)
                ax.grid(True)

            self.ax1.set_ylim(0, 1)
            self.ax2.set_ylim(0, 1)
            self.ax3.set_ylim(-0.1, 0.1)
            self.ax4.set_ylim(-0.1, 0.1)  # Adjust this range as needed
            self.ax5.set_ylim(0, 10)  # Adjust this range as needed


    def process_image(self, msg):
        return np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    

    def update_plots(self, mse_loss, progress):
        self.frame_count += 1
        self.x_data.append(self.frame_count)

        self.mse_loss_data.append(mse_loss)
        
        # Normalize MSE loss
        if mse_loss > self.max_mse_loss:
            self.max_mse_loss = mse_loss
        self.normalized_mse_loss_data = [loss / self.max_mse_loss for loss in self.mse_loss_data]

        # Calculate smoothed normalized MSE loss (rolling average over 4 time steps)
        if len(self.normalized_mse_loss_data) >= 10:
            smoothed_mse = sum(list(self.normalized_mse_loss_data)[-10:]) / 10
        else:
            smoothed_mse = self.normalized_mse_loss_data[-1]
        self.smoothed_mse_loss_data.append(smoothed_mse)

        self.progress_data.append(progress)

        # Calculate progress gradient
        if len(self.progress_data) > 1:
            gradient = self.progress_data[-1] - self.progress_data[-2]
        else:
            gradient = 0
        self.progress_gradient_data.append(gradient)

        # Calculate second derivative of progress
        if len(self.progress_gradient_data) > 1:
            second_derivative = self.progress_gradient_data[-1] - self.progress_gradient_data[-2]
        else:
            second_derivative = 0
        self.progress_second_derivative_data.append(second_derivative)

        # Calculate progress gradient variance
        if len(self.progress_gradient_data) > 1:
            # variance = np.var(list(self.progress_gradient_data))
            # Calculate the variance of the last 10 gradients
            variance = np.var(list(self.progress_gradient_data)[-10:])
        else:
            variance = 0
        # Amplify the variance
        amplified_variance = self.amplify_variance(variance)
        self.progress_gradient_variance_data.append(amplified_variance)

        # Ensure all data lists have the same length
        min_length = min(len(self.x_data), len(self.mse_loss_data), len(self.normalized_mse_loss_data),
                         len(self.smoothed_mse_loss_data), len(self.progress_data), 
                         len(self.progress_gradient_data), len(self.progress_second_derivative_data),
                         len(self.progress_gradient_variance_data))
        
        x_data = list(self.x_data)[-min_length:]
        mse_data = list(self.mse_loss_data)[-min_length:]
        normalized_mse_data = list(self.normalized_mse_loss_data)[-min_length:]
        smoothed_mse_data = list(self.smoothed_mse_loss_data)[-min_length:]
        progress_data = list(self.progress_data)[-min_length:]
        gradient_data = list(self.progress_gradient_data)[-min_length:]
        second_derivative_data = list(self.progress_second_derivative_data)[-min_length:]
        variance_data = list(self.progress_gradient_variance_data)[-min_length:]

        # Adjust y-axis limit for variance plot if necessary
        max_variance = max(variance_data)
        if max_variance > self.ax5.get_ylim()[1]:
            self.ax5.set_ylim(0, max_variance * 1.1)  # Add 10% headroom

        # Update the plot data
        # self.mse_line.set_data(x_data, mse_data)
        # self.normalized_mse_line.set_data(x_data, normalized_mse_data)
        self.smoothed_mse_line.set_data(x_data, smoothed_mse_data)
        self.progress_line.set_data(x_data, progress_data)
        self.gradient_line.set_data(x_data, gradient_data)
        self.second_derivative_line.set_data(x_data, second_derivative_data)
        self.variance_line.set_data(x_data, variance_data)

        # Adjust x-axis limit if necessary
        if self.frame_count >= 100:
            for ax in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5):
                ax.set_xlim(self.frame_count - 99, self.frame_count)

        # Redraw the plot
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Error updating plot: {e}")


    def amplify_variance(self, variance):
        # Method 1: Simple multiplication
        # return variance * self.variance_amplification

        # Method 2: Logarithmic scaling (uncomment to use)
        return np.log1p(variance * self.variance_amplification)

        # Method 3: Square root scaling (uncomment to use)
        # return np.sqrt(variance * self.variance_amplification)


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
                    # Encode images
                    current_image_encoded = self.encoder(self.reshaped_images_tensor.squeeze(1))
                    next_image_encoded = self.encoder(self.next_image)

            # Calculate MSE loss between encoded images
            mse_loss = self.loss_fn(current_image_encoded.squeeze(), next_image_encoded.squeeze())
            # MSE loss on the raw images
            # mse_loss = self.loss_fn(self.reshaped_images_tensor.squeeze(), self.next_image.squeeze())
            self.mse_loss_data.append(mse_loss.item())
        
        if self.use_displays and self.next_image is not None:
            # Update the plots
            mse_loss = self.mse_loss_data[-1] if self.mse_loss_data else 0
            progress = self.output[7]
            self.update_plots(mse_loss, self.progress)

        # Policy Forward Pass
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                output, hs = self.policy(self.images_tensor, self.q_pos_tensor)

        # World Model Forward Pass
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                self.next_image = self.world_model(hs, output)
        self.predicted_image = self.next_image.squeeze().float().permute(1, 2, 0).cpu().numpy()
        self.predicted_image = (self.predicted_image * 255).astype(np.uint8)

        # Convert from torch, handling BFloat16
        output = output.float().cpu().numpy().squeeze()
        self.progress = output[0][-1]
        selection_index = 4
        output = output[selection_index]
        self.output = output.tolist()
        joints = self.output[:6]
        gripper = self.output[6]

        input('Press Enter to continue...')

        # Control Robot
        self.robot_gripper_cmd.cmd = gripper
        if self.use_control:
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

