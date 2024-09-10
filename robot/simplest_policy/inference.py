import torch
import torch.nn.functional as F
import os
import numpy as np

from policy3.model import PolicyCNNMLP
from policy3.world_model import WorldModel
from policy3.dataloader import DataLoaderLite
import policy3.config as config
import random
import tqdm

import pdb


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

# Seed
seed_everything(config.seed)
torch.set_float32_matmul_precision('high')

# Set device
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "CUDA is not available"

# Load data
episodes = os.listdir(config.dataset_dir)
num_episodes = len(episodes)
train_episodes = 1
train_indices = np.random.choice(num_episodes, size=train_episodes, replace=False)
train_loader = DataLoaderLite(config.dataset_dir, config.B, config.T, 'train', train_indices)

# Create model
model = PolicyCNNMLP().to(device)
use_compile = config.use_compile
if use_compile:
    print("Compiling model")
    model = torch.compile(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

# World model
world_model = WorldModel(latent_dim=1024, action_dim=80).to(device)
if use_compile:
    print("Compiling world model")
    world_model = torch.compile(world_model)

num_params = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

for step in tqdm.tqdm(range(0, config.max_steps + 1)):
    last_step = (step == config.max_steps - 1)

    actions, q_pos, images, next_progress, next_images = train_loader.next_batch()
    actions = actions.to(device)
    q_pos = q_pos.to(device)
    images = images.to(device)
    next_progress = next_progress.to(device)
    next_images = next_images.to(device)
    # Target for the policy network
    target = torch.cat([actions, next_progress], dim=2)

    # Policy Forward pass
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        output, hs = model(images, q_pos)
        policy_loss = F.mse_loss(output, target)
    # Decode the latent vector to the next image
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        predicted_image = world_model(hs.detach(), target)
        world_model_loss = F.mse_loss(predicted_image, next_images.squeeze())
    pdb.set_trace()

    input("Enter to cont...")
