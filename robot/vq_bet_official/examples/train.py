import random
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
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

    if "visual_input" in cfg and cfg.visual_input:
        print("use visual environment")
        cfg.model.gpt_model.config.input_dim = 1024
    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )

    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataloader
    dataset_dir = '/home/qutrll/data/pot_pick_place_2_10hz'
    episodes = os.listdir(dataset_dir)
    num_episodes = len(episodes)
    train_episodes = int(num_episodes)
    train_indices = np.random.choice(num_episodes, size=train_episodes, replace=False)
    train_loader = DataLoaderLite('/home/qutrll/data/pot_pick_place_2_10hz', 32, 1, 'train', train_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is not available"

    num_steps = 1000000
    for step in tqdm.trange(num_steps):
        optimizer["optimizer1"].zero_grad()
        optimizer["optimizer2"].zero_grad()

        act, _, batch_images, _, _ = train_loader.next_batch()
        act = act.unsqueeze(1)
        act = act.to(device)
        obs = batch_images.to(device)

        predicted_act, loss, loss_dict = cbet_model(obs, None, act)
        loss.backward()
        optimizer["optimizer1"].step()
        optimizer["optimizer2"].step()

        if step > 10000 and step % 10000 == 0:
            cbet_model.save_model(save_path)


if __name__ == "__main__":
    main()
