import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from vqvae.vqvae import *
import wandb

from dataloader import DataLoaderLite

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
    # train_data, test_data = hydra.utils.instantiate(cfg.data)
    # train_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=False
    # )
    # My Dataloader
    dataset_dir = '/home/qutrll/data/pot_pick_place_2_10hz'
    episodes = os.listdir(dataset_dir)
    num_episodes = len(episodes)
    train_episodes = int(num_episodes)
    train_indices = np.random.choice(num_episodes, size=train_episodes, replace=False)
    train_loader = DataLoaderLite('/home/qutrll/data/pot_pick_place_2_10hz', 64, 1, 'train', train_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is not available"

    num_steps = 10000
    for _ in tqdm.trange(num_steps):

        act, _, _, _, _ = train_loader.next_batch()
        act = act.unsqueeze(1)
        act = act.to(device)

        (
            encoder_loss,
            vq_loss_state,
            vq_code,
            vqvae_recon_loss,
        ) = vqvae_model.vqvae_update(act)  # N T D

    state_dict = vqvae_model.state_dict()
    torch.save(state_dict, os.path.join(save_path, "trained_vqvae.pt"))
    print(save_path)


if __name__ == "__main__":
    main()
