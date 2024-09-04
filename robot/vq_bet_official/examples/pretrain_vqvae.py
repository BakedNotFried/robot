import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from vqvae.vqvae import *

from vq_bet_official.examples.dataloader_wm import DataLoaderLite

import pdb

import time

config_name = "pretrain_widow"

def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

@hydra.main(config_path="configs", config_name=config_name, version_base="1.2")
def main(cfg):
    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    vqvae_model = hydra.utils.instantiate(cfg.vqvae_model)

    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    # My Dataloader
    dataset_dir = '/home/qutrll/data/pot_pick_place_2_10hz'
    episodes = os.listdir(dataset_dir)
    num_episodes = len(episodes)
    # num_episodes = 1
    train_episodes = int(num_episodes)
    train_indices = np.random.choice(num_episodes, size=train_episodes, replace=False)
    train_loader = DataLoaderLite('/home/qutrll/data/pot_pick_place_2_10hz', 32, 10, 'train', train_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is not available"

    num_steps = 50000
    for _ in tqdm.trange(num_steps):

        action, _, _, next_progress, _ = train_loader.next_batch()
        action = action.to(device)
        next_progress = next_progress.to(device)
        action = torch.cat((action, next_progress), dim=2)

        (
            encoder_loss,
            vq_loss_state,
            vq_code,
            vqvae_recon_loss,
        ) = vqvae_model.vqvae_update(action)  # N T D


    state_dict = vqvae_model.state_dict()
    torch.save(state_dict, os.path.join(save_path, "trained_vqvae.pt"))
    print(save_path)


if __name__ == "__main__":
    main()
