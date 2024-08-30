import random
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F

from vq_bet_official.examples.dataloader_wm import DataLoaderLite

import pdb

config_name = "train_widow"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

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
        if combined.dim() == 1:
            combined = combined.unsqueeze(0)
        return self.decoder(combined)

@hydra.main(config_path="configs", config_name=config_name, version_base="1.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    cfg.model.gpt_model.config.input_dim = 1024

    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    cbet_model.eval()
    cbet_model.load_model(save_path)

    # Instantiate the world Model
    world_model = WorldModel()
    world_model = world_model.to(cfg.device)
    world_model.eval()
    world_model_path = "/home/qutrll/data/checkpoints/vq_bet/model/2/world_model.pt"
    world_model.load_state_dict(torch.load(world_model_path))

    # Dataloader
    dataset_dir = '/home/qutrll/data/pot_pick_place_2_10hz'
    num_episodes = 1
    train_episodes = 1
    train_indices = np.random.choice(num_episodes, size=train_episodes, replace=False)
    train_loader = DataLoaderLite(dataset_dir, 1, 1, 'train', train_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is not available"

    while True:

        act, batch_q_pos, batch_images, _, _ = train_loader.next_batch()
        act = act.unsqueeze(1)
        act = act.to(device)
        obs = batch_images.to(device)
        q_pos = batch_q_pos.to(device)

        # inference
        with torch.no_grad():
            predicted_act, policy_loss, loss_dict, hs_output = cbet_model(obs, q_pos, None, None)
        
        # Forward pass through world model
        with torch.no_grad():
            predicted_next_image = world_model(hs_output, predicted_act)
        pdb.set_trace()
        input("Press Enter to continue...")
        print("predicted_act: ", predicted_act)


if __name__ == "__main__":
    main()
