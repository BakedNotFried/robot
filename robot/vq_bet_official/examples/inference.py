import random
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from dataloader import DataLoaderLite

import pdb

config_name = "train_widow"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


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

    # Dataloader
    dataset_dir = '/home/qutrll/data/pot_pick_place_2_10hz'
    num_episodes = 1
    train_episodes = 1
    train_indices = np.random.choice(num_episodes, size=train_episodes, replace=False)
    train_loader = DataLoaderLite(dataset_dir, 1, 1, 'train', train_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is not available"

    while True:

        act, _, batch_images, _, _ = train_loader.next_batch()
        act = act.unsqueeze(1)
        act = act.to(device)
        obs = batch_images.to(device)

        # inference
        with torch.no_grad():
            predicted_act, _, _ = cbet_model(obs, None, None)

        print("predicted_act: ", predicted_act)


if __name__ == "__main__":
    main()
