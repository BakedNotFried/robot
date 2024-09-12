import torch
import torch.nn.functional as F
import os
import numpy as np
from model import MobilePolicy as Policy
from dataloader import DataLoaderLite
import config as config
import random
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import pdb
import csv


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

# Seed
seed_everything(config.seed)

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
model = Policy().to(device)
use_compile = False
if use_compile:
    print("Compiling model")
    model = torch.compile(model)
model.train()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params/1e6:.2f}M")

# Define loss function and optimizer
policy_optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

# Add cosine decay learning rate scheduler
policy_scheduler = CosineAnnealingLR(policy_optimizer, T_max=config.max_steps, eta_min=1e-6)

# Create a list to store loss values
loss_values = []

# Create a CSV file to save loss values
csv_path = os.path.join(config.checkpoint_dir, f'loss_values_seed_{config.seed}.csv')
with open(csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Step', 'Loss'])  # Write header

for step in tqdm.tqdm(range(0, config.max_steps + 1)):
    policy_optimizer.zero_grad()

    actions, q_pos, images, next_progress, next_images = train_loader.next_batch()
    actions = actions.to(device)
    q_pos = q_pos.to(device)
    images = images.to(device)
    next_progress = next_progress.to(device)
    # Target for the policy network
    target = torch.cat([actions, next_progress], dim=2)

    # Policy Forward pass
    output = model(images, q_pos)
    policy_loss = F.mse_loss(output, target)

    policy_loss.backward()

    policy_optimizer.step()
    policy_scheduler.step()

    loss_values.append(policy_loss.item())
    if step % 1000 == 0:
        print(f"Step: {step}, Loss: {policy_loss.item()}")
        with open(csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([step, policy_loss.item()])

    # Save regular
    if (step > config.checkpoint_interval) and (step % config.checkpoint_interval == 0):
        ckpt_path = os.path.join(config.checkpoint_dir, f'checkpoint_step_{step}_seed_{config.seed}.ckpt')
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'policy_optimizer_state_dict': policy_optimizer.state_dict(),
            'policy_loss': policy_loss.item(),
        }, ckpt_path)

np.save(os.path.join(config.checkpoint_dir, f'loss_seed_{config.seed}.npy'), np.array(loss_values))
