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
from torch.optim.lr_scheduler import CosineAnnealingLR
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

# Create directories
os.makedirs(config.checkpoint_dir, exist_ok=True)

# Load data
episodes = os.listdir(config.dataset_dir)
num_episodes = len(episodes)
# num_episodes = 1
train_episodes = int(num_episodes)
train_indices = np.random.choice(num_episodes, size=train_episodes, replace=False)

train_loader = DataLoaderLite(config.dataset_dir, config.B, config.T, 'train', train_indices)

# Create model
model = PolicyCNNMLP().to(device)
# use_compile = config.use_compile
use_compile = True
if use_compile:
    print("Compiling model")
    model = torch.compile(model)
model.train()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

# World model
world_model = WorldModel(latent_dim=1792, action_dim=80).to(device)
if use_compile:
    print("Compiling world model")
    world_model = torch.compile(world_model)
world_model.train()

num_params = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

# Define loss function and optimizer
policy_optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

# World model optimizer
world_model_optimizer = torch.optim.AdamW(world_model.parameters(), lr=config.lr)

# Add cosine decay learning rate scheduler
policy_scheduler = CosineAnnealingLR(policy_optimizer, T_max=config.max_steps, eta_min=1e-6)
world_model_scheduler = CosineAnnealingLR(world_model_optimizer, T_max=config.max_steps, eta_min=1e-6)

# Load checkpoint if resuming
if config.resume_training:
    if config.resume_step is not None:
        checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_step_{config.resume_step}_seed_{config.seed}.ckpt')
    else:
        checkpoints = [f for f in os.listdir(config.checkpoint_dir) if f.startswith('checkpoint_step_') and f.endswith('.ckpt')]
        if not checkpoints:
            raise ValueError("No checkpoints found for resuming")
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2]))
        checkpoint_path = os.path.join(config.checkpoint_dir, latest_checkpoint)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    world_model.load_state_dict(checkpoint['world_model_state_dict'])
    world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer_state_dict'])
    start_step = checkpoint['step']
else:
    start_step = 0

# Training loop
min_val_loss = np.inf
best_ckpt_info = None

for step in tqdm.tqdm(range(start_step, config.max_steps + 1)):
    # Training step
    policy_optimizer.zero_grad()
    world_model_optimizer.zero_grad()

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

    # Backward pass
    policy_loss.backward()
    world_model_loss.backward()

    # Step
    policy_optimizer.step()
    world_model_optimizer.step()

    policy_scheduler.step()
    world_model_scheduler.step()

    # Save regular
    if (step > config.checkpoint_interval) and (step % config.checkpoint_interval == 0):
        ckpt_path = os.path.join(config.checkpoint_dir, f'checkpoint_step_{step}_seed_{config.seed}.ckpt')
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'policy_optimizer_state_dict': policy_optimizer.state_dict(),
            'policy_loss': policy_loss.item(),
            'world_model_state_dict': world_model.state_dict(),
            'world_model_optimizer_state_dict': world_model_optimizer.state_dict(),
        }, ckpt_path)
