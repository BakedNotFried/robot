import random
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F

from vq_bet_official.examples.dataloader_wm import DataLoaderLite

import pdb

config_name = "train_widow"


class WorldModel(nn.Module):
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
        combined = torch.cat([latents.squeeze(), actions.squeeze()], dim=-1)
        if len(combined.shape) == 1:
            combined = combined.unsqueeze(0)
        return self.decoder(combined)
    

def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path="configs", config_name=config_name, version_base="1.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    if "visual_input" in cfg and cfg.visual_input:
        print("use visual environment")
        cfg.model.gpt_model.config.input_dim = 1024
    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )

    # Instantiate the world Model
    world_model = WorldModel()
    world_model = world_model.to(cfg.device)

    # Create optimizer for world model
    optimizer_world_model = torch.optim.Adam(world_model.parameters(), lr=3e-4)

    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataloader
    dataset_dir = '/home/qutrll/data/pot_pick_place_2_10hz'
    episodes = os.listdir(dataset_dir)
    num_episodes = len(episodes)
    train_episodes = int(num_episodes)
    train_indices = np.random.choice(num_episodes, size=train_episodes, replace=False)
    action_selection = 1
    train_loader = DataLoaderLite('/home/qutrll/data/pot_pick_place_2_10hz', 32, action_selection, 'train', train_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is not available"

    num_steps = 500000
    for step in tqdm.trange(num_steps):
        # Zero out gradients
        optimizer["optimizer1"].zero_grad()
        optimizer["optimizer2"].zero_grad()
        optimizer_world_model.zero_grad()

        # Load Data
        action, q_pos, images, next_progress, next_images = train_loader.next_batch()
        action = action.unsqueeze(1)
        action = action.to(device)
        q_pos = q_pos.to(device)
        images = images.to(device)
        next_progress = next_progress.unsqueeze(1)
        next_progress = next_progress.to(device)
        next_images = next_images.to(device)
        action = torch.cat([action, next_progress], dim=2)

        # Forward pass through cbet model
        predicted_act, policy_loss, loss_dict, hs_output = cbet_model(images, q_pos, None, action)

        # Forward pass through world model
        predicted_next_image = world_model(hs_output.detach(), action)

        # Compute loss
        world_model_loss = F.mse_loss(predicted_next_image, next_images.squeeze())
        
        # Backprop on cbet model
        policy_loss.backward()
        optimizer["optimizer1"].step()
        optimizer["optimizer2"].step()

        # Backprop on world model
        world_model_loss.backward()
        optimizer_world_model.step()

        # if step > 10000 and step % 10000 == 0:
        #     cbet_model.save_model(save_path)
        #     world_model_path = save_path / "world_model.pt"
        #     torch.save(world_model.state_dict(), world_model_path)


if __name__ == "__main__":
    main()
