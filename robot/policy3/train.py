import torch
from torch import nn
import os
import numpy as np
from copy import deepcopy

from model import PolicyCNNMLP
from dataloader import DataLoaderLite
from logger import logger, log_train, log_val, log_best_model
import config

import pdb

# Set random seed
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)
np.random.seed(config.seed)

# Set device
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "CUDA is not available"

# Create directories
os.makedirs(config.checkpoint_dir, exist_ok=True)

# Load data
episodes = os.listdir(config.dataset_dir)
num_episodes = len(episodes)
train_episodes = int(num_episodes * config.split)
val_episodes = num_episodes - train_episodes
train_indices = np.random.choice(num_episodes, size=train_episodes, replace=False)
val_indices = np.array([i for i in range(num_episodes) if i not in train_indices])

logger.info("Loading data...")
train_loader = DataLoaderLite(config.dataset_dir, config.B, config.T, 'train', train_indices)
val_loader = DataLoaderLite(config.dataset_dir, config.B, config.T, 'val', val_indices)
logger.info("Data loaded")

# Create model
torch.set_float32_matmul_precision('high')
model = PolicyCNNMLP().to(device)
use_compile = True
if use_compile:
    logger.info("Compiling model")
    model = torch.compile(model)
    logger.info("Model compiled")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Number of parameters: {num_params}")

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

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
    
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    logger.info(f"Resuming from step {start_step}")
else:
    start_step = 0

# Training loop
min_val_loss = np.inf
best_ckpt_info = None

for step in range(start_step, config.max_steps):
    last_step = (step == config.max_steps - 1)

    # Training step
    model.train()
    optimizer.zero_grad()
    actions, q_pos, images, progress = train_loader.next_batch()
    actions = actions.to(device)
    q_pos = q_pos.to(device)
    images = images.to(device)
    progress = progress.to(device)
    target = torch.cat([actions, progress], dim=1)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        output = model(images, q_pos, progress)
        loss = loss_fn(output, target)

    loss.backward()
    optimizer.step()

    log_train(step, loss.item())

    # Validation
    if (step + 1) % config.validation_interval == 0 or last_step:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(config.validation_steps):
                actions, q_pos, images, progress = val_loader.next_batch()
                actions = actions.to(device)
                q_pos = q_pos.to(device)
                images = images.to(device)
                progress = progress.to(device)
                target = torch.cat([actions, progress], dim=1)
                
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    output = model(images, q_pos, progress)
                    loss = loss_fn(output, target)
                
                val_loss += loss.item()
        
        val_loss /= config.validation_steps
        log_val(step, val_loss)

        # Save the best model checkpoint based on validation loss
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_ckpt_info = (step+1, min_val_loss, deepcopy(model.state_dict()))

    # Save regular checkpoint
    if (step + 1) % config.checkpoint_interval == 0 or last_step:
        ckpt_path = os.path.join(config.checkpoint_dir, f'checkpoint_step_{step+1}_seed_{config.seed}.ckpt')
        torch.save({
            'step': step + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'val_loss': val_loss if (step + 1) % config.validation_interval == 0 or last_step else None
        }, ckpt_path)

# Save the best model
if best_ckpt_info is not None:
    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(config.checkpoint_dir, f'policy_best_seed_{config.seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    log_best_model(best_step, min_val_loss)
