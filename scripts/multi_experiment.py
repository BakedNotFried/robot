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
import torch.nn.functional as F

# CNNMLP Policy + World Model
from robot.policy3.model import PolicyCNNMLP
from robot.policy3.world_model import WorldModel as CNNWorldModel

# Simplest Policy
# from robot.simplest_policy.model import Policy

# VQ-BET Policy + World Model
import os
from pathlib import Path
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import torch.nn as nn
config_name = "train_widow"

# Mobile Policy
from robot.mobile_policy.multi_model import MobilePolicy

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def save_experiment_data(save_dir, image_actual_data, image_predicted_data, progress_data, joint_states_data):
    """
    Save experiment data to numpy files.
    
    Args:
    save_dir (str): Directory to save the data files
    image_actual_data (list): List of actual image tensors
    image_predicted_data (list): List of predicted image tensors
    progress_data (list): List of progress values
    joint_states_data (list): List of joint states
    """
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # List the number of experiments
    num_experiments = len(os.listdir(save_dir))

    # Create a new directory for the current experiment
    save_dir = os.path.join(save_dir, f'{num_experiments}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert lists to numpy arrays
    image_actual_array = np.array(image_actual_data)
    image_predicted_array = np.array(image_predicted_data)
    progress_array = np.array(progress_data)
    joint_states_array = np.array(joint_states_data)
    
    # Save each array to a separate file
    np.save(os.path.join(save_dir, 'image_actual_data.npy'), image_actual_array)
    np.save(os.path.join(save_dir, 'image_predicted_data.npy'), image_predicted_array)
    np.save(os.path.join(save_dir, 'progress_data.npy'), progress_array)
    np.save(os.path.join(save_dir, 'joint_states_data.npy'), joint_states_array)
    
    print(f"Data saved successfully in {save_dir}")
    print(f"Shapes of saved arrays:")
    print(f"image_actual_data: {image_actual_array.shape}")
    print(f"image_predicted_data: {image_predicted_array.shape}")
    print(f"progress_data: {progress_array.shape}")
    print(f"joint_states_data: {joint_states_array.shape}")

def save_ensemble_data(save_dir, image_actual_data, progress_data, joint_states_data, action_data, action_variance_data):
    """
    Save experiment data to numpy files.
    
    Args:
    save_dir (str): Directory to save the data files
    image_actual_data (list): List of actual image tensors
    image_predicted_data (list): List of predicted image tensors
    progress_data (list): List of progress values
    joint_states_data (list): List of joint states
    action_data (list): List of actions
    action_variance_data (list): List of action variances
    """
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # List the number of experiments
    num_experiments = len(os.listdir(save_dir))

    # Create a new directory for the current experiment
    save_dir = os.path.join(save_dir, f'{num_experiments}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert lists to numpy arrays
    image_actual_array = np.array(image_actual_data)
    progress_array = np.array(progress_data)
    joint_states_array = np.array(joint_states_data)
    action_array = np.array(action_data)
    action_variance_array = np.array(action_variance_data)
    
    # Save each array to a separate file
    np.save(os.path.join(save_dir, 'image_actual_data.npy'), image_actual_array)
    np.save(os.path.join(save_dir, 'progress_data.npy'), progress_array)
    np.save(os.path.join(save_dir, 'joint_states_data.npy'), joint_states_array)
    np.save(os.path.join(save_dir, 'action_data.npy'), action_array)
    np.save(os.path.join(save_dir, 'action_variance_data.npy'), action_variance_array)
    
    print(f"Data saved successfully in {save_dir}")
    print(f"Shapes of saved arrays:")
    print(f"image_actual_data: {image_actual_array.shape}")
    print(f"progress_data: {progress_array.shape}")
    print(f"joint_states_data: {joint_states_array.shape}")
    print(f"action_data: {action_array.shape}")
    print(f"action_variance_data: {action_variance_array.shape}")

class VQWorldModel(nn.Module):
    def __init__(self, latent_dim=256, action_dim=8):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, latents, actions):
        b, t, n = actions.shape
        actions = actions.reshape(b, t * n)
        combined = torch.cat([latents.squeeze(), actions.squeeze()], dim=-1)
        if len(combined.shape) == 1:
            combined = combined.unsqueeze(0)
        return self.decoder(combined)

# HIT Policy + World Model
import json
from HIT.model_util import make_policy

def create_hit_policy(seed, device):
        policy_class = 'HIT'
        config_dir = f"/home/qutrll/data/checkpoints/HIT/pot_pick_place/ensemble/{seed}/all_configs.json"
        config = json.load(open(config_dir))
        policy_config = config['policy_config']

        seed_everything(seed)

        policy = make_policy(policy_class, policy_config)
        policy.eval()
        ckpt_dir = f"/home/qutrll/data/checkpoints/HIT/pot_pick_place/ensemble/{seed}/policy_step_60000_seed_{seed}.ckpt"
        loading_status = policy.deserialize(torch.load(ckpt_dir, map_location='cuda'))
        if not loading_status:
            print(f'Failed to load policy_last.ckpt')
        policy.cuda()

        return policy

# Diffusion World Model
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# Diffusion Policy
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D

# Metrics
import lpips
from torchmetrics.functional.image import peak_signal_noise_ratio

from robot.keyboard_interface import KeyboardExperimentInterface

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
        
        # Keyboard Interface. For Experimental Control
        # self.kb = KeyboardExperimentInterface()

        # Set Sys
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.device == torch.device("cuda"), "CUDA is not available"
        torch.set_float32_matmul_precision('high')

        # Metrics
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_loss_fn.eval()

        # experiment variables
        self.use_displays = True
        self.use_plots = False
        self.use_control = True
        self.record_data = False
        self.keyboard_control = True
        self.joint_config_event = False
        self.num_experiment_steps = 30
        self.step_num = 0
        # event step 4 for VD
        self.event_step = 4
        # task type. options: pot, wipe, cupboard
        self.task_type = "pot"
        # trial type. options: IND, OODVD, OODHD
        self.trial_type = "IND"
        # options mlp, vq, hit, ensemble, dropout, dp
        self.policy_type = "mobile"
        # options dm or simp
        self.world_model_type = ""
        
        # Polcicies        
        if self.policy_type == "mobile":
            seed_everything(42)
            # Init Policy
            if self.task_type == "pot":
                checkpoint_path = '/home/qutrll/data/checkpoints/multi_task_mobile/1/checkpoint_step_200000_seed_42.ckpt'
            checkpoint = torch.load(checkpoint_path)
            self.policy = MobilePolicy()
            self.policy = self.policy.to(self.device)
            self.policy.eval()
            use_compile = False
            if use_compile:
                self.policy = torch.compile(self.policy)
            self.policy.load_state_dict(checkpoint['model_state_dict'])

        # Cameras Setup
        self.height = 240
        self.width = 424
        self.field_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.predicted_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.create_subscription(
            Image, 
            '/cam_field/camera/color/image_raw', 
            self.field_image_callback, 
            1, 
        )

        # Establish the initial joint angles
        self.arm_joint_angles = START_ARM_POSE[:6]
        self.robot.arm.set_joint_positions(START_ARM_POSE[:6], blocking=False)
        self.gripper_joint = ROBOT_GRIPPER_JOINT_MID
        self.robot_gripper_cmd.cmd = self.gripper_joint
        self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)
        self.task_index = torch.tensor([0], dtype=torch.long, device=self.device)

        # Output. Starts with arm joint angles, gripper joint, and progress
        self.output = self.arm_joint_angles + [self.gripper_joint] + [0.0]

        # Preallocate tensors for images and actions
        self.images_tensor = torch.empty(1, 1, 3, self.height, self.width, dtype=torch.float32, device=self.device)
        self.prev_images_tensor = torch.empty(1, 1, 3, 224, 224, dtype=torch.float32, device=self.device)
        self.q_pos_tensor = torch.empty(1, 7, dtype=torch.float32, device=self.device)
        self.action_tensor = torch.empty(1, 8, dtype=torch.float32, device=self.device)
        self.next_image = None
        self.progress = None

        # Setup Recording
        self.image_actual_data = []
        self.image_predicted_data = []
        self.progress_data = []
        self.joint_states_data = []
        # Ensemble
        self.action_data = []
        self.action_variance_data = []

        # Setup Plotting
        # Initialize deques to store data for plotting
        self.lpips_data = deque(maxlen=100)
        self.normalized_lpips_data = deque(maxlen=100)
        self.smoothed_lpips_data = deque(maxlen=100)
        self.psnr_data = deque(maxlen=100)
        self.progress_plot_data = deque(maxlen=100)
        self.progress_gradient_data = deque(maxlen=100)
        self.progress_gradient_variance_data = deque(maxlen=100)
        self.progress_second_derivative_data = deque(maxlen=100)
        self.x_data = deque(maxlen=100)
        self.frame_count = 0
        # Variable for max lpips loss
        self.max_lpips = -np.inf
        self.variance_amplification = 10000

        # Set up the plot
        if self.use_plots:
            plt.ion()
            self.fig, (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6, self.ax7) = plt.subplots(7, 1, figsize=(10, 25))
            self.fig.tight_layout(pad=3.0)

            # Initialize lines for each plot
            self.lpips_line, = self.ax1.plot([], [], 'r-')
            self.smoothed_lpips_line, = self.ax2.plot([], [], 'b-')
            self.progress_line, = self.ax3.plot([], [], 'b-')
            self.gradient_line, = self.ax4.plot([], [], 'g-')
            self.second_derivative_line, = self.ax5.plot([], [], 'm-')
            self.variance_line, = self.ax6.plot([], [], 'r-')
            self.psnr_line, = self.ax7.plot([], [], 'b-')

            # Set up the axes
            self.ax1.set_title('LPIPS')
            self.ax2.set_title('Normalised LPIPS')
            self.ax3.set_title('Progress')
            self.ax4.set_title('Progress Gradient')
            self.ax5.set_title('Progress Gradient Gradient')
            self.ax6.set_title('Progress Gradient Variance')
            self.ax7.set_title('PSNR')

            for ax in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6, self.ax7):
                ax.set_xlim(0, 100)
                ax.grid(True)

            self.ax1.set_ylim(0, 1)
            self.ax2.set_ylim(0, 1)
            self.ax3.set_ylim(0, 1)
            self.ax4.set_ylim(-0.1, 0.1)
            self.ax5.set_ylim(-0.1, 0.1)
            self.ax6.set_ylim(0, 10)
            self.ax7.set_ylim(0, 100)
        
        self.first_run = True


    def field_image_callback(self, msg):
        self.field_image = self.process_image(msg)

        if self.use_displays:
            self.display_images()
        
        # Convert images to tensors and normalize
        self.images_tensor[0, 0] = torch.from_numpy(self.field_image).float().permute(2, 0, 1) / 255.0
        reshaped_tensor = self.images_tensor.squeeze(1)
        self.reshaped_images_tensor = F.interpolate(reshaped_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        self.reshaped_images_tensor = self.reshaped_images_tensor.unsqueeze(0)
        self.q_pos_tensor[0] = torch.tensor(self.output[:7], dtype=torch.float32, device=self.device)

        if self.first_run:
            self.first_run = False
            self.prev_images_tensor = self.reshaped_images_tensor
        
        # Stack the images for the next iteration
        input_images = torch.cat([self.prev_images_tensor, self.reshaped_images_tensor], dim=1)

        if self.policy_type == "mobile":
            # Policy Forward Pass
            with torch.no_grad():
                output = self.policy(input_images, self.q_pos_tensor, self.task_index)
            
            # Convert from torch, handling BFloat16
            output = output.float().cpu().numpy().squeeze()
            selection_index = 4
            output = output[selection_index]
            self.output = output.tolist()
            joints = self.output[:6]
            gripper = self.output[6]
            self.progress = self.output[-1]

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

        if self.keyboard_control:
            user_input = input("Press Enter to continue...")
            if user_input == "w":
                self.reset_experiment("wipe")
            elif user_input == "c":
                self.reset_experiment("cupboard")  

        # Update the previous images tensor
        self.prev_images_tensor = self.reshaped_images_tensor


    def reset_record_data(self):
        self.image_actual_data = []
        self.image_predicted_data = []
        self.progress_data = []
        self.joint_states_data = []
        self.action_data = []
        self.action_variance_data = []


    def process_image(self, msg):
        return np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    

    def update_plots(self, lpips, progress):
        self.frame_count += 1
        self.x_data.append(self.frame_count)

        self.lpips_data.append(lpips)
        
        # Normalize lpips loss
        if lpips > self.max_lpips:
            self.max_lpips = lpips
        self.normalized_lpips_data = [loss / self.max_lpips for loss in self.lpips_data]

        # Calculate smoothed normalized lpips loss (rolling average over 4 time steps)
        if len(self.normalized_lpips_data) >= 10:
            smoothed_lpips = sum(list(self.normalized_lpips_data)[-10:]) / 10
        else:
            smoothed_lpips = self.normalized_lpips_data[-1]
        self.smoothed_lpips_data.append(smoothed_lpips)

        self.progress_plot_data.append(progress)

        # Calculate progress gradient
        if len(self.progress_plot_data) > 1:
            gradient = self.progress_plot_data[-1] - self.progress_plot_data[-2]
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
        min_length = min(len(self.x_data), len(self.lpips_data), len(self.normalized_lpips_data),
                        len(self.smoothed_lpips_data), len(self.progress_plot_data), 
                        len(self.progress_gradient_data), len(self.progress_second_derivative_data),
                        len(self.progress_gradient_variance_data), len(self.psnr_data))
        
        x_data = list(self.x_data)[-min_length:]
        lpips_data = list(self.lpips_data)[-min_length:]
        normalized_lpips_data = list(self.normalized_lpips_data)[-min_length:]
        smoothed_lpips_data = list(self.smoothed_lpips_data)[-min_length:]
        progress_data = list(self.progress_plot_data)[-min_length:]
        gradient_data = list(self.progress_gradient_data)[-min_length:]
        second_derivative_data = list(self.progress_second_derivative_data)[-min_length:]
        variance_data = list(self.progress_gradient_variance_data)[-min_length:]
        psnr_data = list(self.psnr_data)[-min_length:]

        # Adjust y-axis limit for variance plot if necessary
        max_variance = max(variance_data)
        if max_variance > self.ax5.get_ylim()[1]:
            self.ax5.set_ylim(0, max_variance * 1.1)  # Add 10% headroom

        # Update the plot data
        self.lpips_line.set_data(x_data, lpips_data)
        self.smoothed_lpips_line.set_data(x_data, smoothed_lpips_data)
        self.progress_line.set_data(x_data, progress_data)
        self.gradient_line.set_data(x_data, gradient_data)
        self.second_derivative_line.set_data(x_data, second_derivative_data)
        self.variance_line.set_data(x_data, variance_data)
        self.psnr_line.set_data(x_data, psnr_data)

        # Adjust x-axis limit if necessary
        if self.frame_count >= 100:
            for ax in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6, self.ax7):
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
        if self.predicted_image is not None:
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
        else:
            # Add progress bar
            progress = self.output[7]  # Assuming progress is the last element in self.output
            bar_width = self.width  # Full width of the combined image
            bar_height = 20
            bar_top = self.height - bar_height - 10  # 10 pixels from the bottom

            # Draw the background of the progress bar
            cv2.rectangle(self.field_image, (0, bar_top), (bar_width, bar_top + bar_height), (100, 100, 100), -1)

            # Draw the filled part of the progress bar
            filled_width = int(bar_width * progress / 1.0)
            cv2.rectangle(self.field_image, (0, bar_top), (filled_width, bar_top + bar_height), (0, 255, 0), -1)

            # Add progress text
            progress_text = f'Progress: {progress:.2f}/1.0'
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.field_image, progress_text, (10, bar_top - 10), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Just display the field image if the predicted image is not available
            cv2.imshow('Field Image', cv2.cvtColor(self.field_image, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)


        
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

    def reset_experiment(self, task):
        """Move robot arm to start demonstration pose"""
        # move arm to starting position
        start_arm_qpos = START_ARM_POSE[:6]
        control_arms(
            [self.robot],
            [start_arm_qpos],
            [self.arm_joint_angles],
            moving_time=1.5,
        )

        # Establish the initial joint angles
        self.arm_joint_angles = START_ARM_POSE[:6]
        self.robot.arm.set_joint_positions(START_ARM_POSE[:6], blocking=False)
        if task == "wipe":
            self.gripper_joint = ROBOT_GRIPPER_JOINT_MID
            self.task_index = torch.tensor([0], dtype=torch.long, device=self.device)
        elif task == "cupboard":
            self.gripper_joint = ROBOT_GRIPPER_JOINT_CLOSE_MAX
            self.task_index = torch.tensor([1], dtype=torch.long, device=self.device)
        self.robot_gripper_cmd.cmd = self.gripper_joint
        self.robot.gripper.core.pub_single.publish(self.robot_gripper_cmd)

        # Output. Starts with arm joint angles, gripper joint, and progress
        self.output = self.arm_joint_angles + [self.gripper_joint] + [0.0]

        # Preallocate tensors for images and actions
        self.images_tensor = torch.empty(1, 1, 3, self.height, self.width, dtype=torch.float32, device=self.device)
        self.prev_images_tensor = torch.empty(1, 1, 3, 224, 224, dtype=torch.float32, device=self.device)
        self.q_pos_tensor = torch.empty(1, 7, dtype=torch.float32, device=self.device)
        self.action_tensor = torch.empty(1, 8, dtype=torch.float32, device=self.device)
        self.next_image = None
        self.progress = None

        self.first_run = True
    
    def __del__(self):
        plt.close(self.fig)

# Get args
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

