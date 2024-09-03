import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vector_quantize_pytorch import FSQ
from tqdm import trange
import pdb

import os
import h5py
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt

class DataLoaderLite:
    def __init__(self, data_dir, B, T, split, indices):
        self.B = B
        self.T = T
        self.data_dir = data_dir
        episodes = os.listdir(data_dir)
        episodes = [episodes[i] for i in indices]
        assert split in ['train', 'val']
        print(f"Loading {split} episodes: {episodes}")
        self.episode_files = [os.path.join(data_dir, episode) for episode in episodes]
        self.num_episodes = len(self.episode_files)
        assert self.num_episodes > 0, "No episodes found"
        print(f"Found {self.num_episodes} episodes")
        keys = ['action', 'q_pos', 'oh_images', 'field_images', 'wrist_images', 'progress']
        
        # Load all episodes into memory
        self.actions = []
        self.q_pos = []
        # self.oh_images = []
        self.field_images = []
        # self.wrist_images = []
        self.progress = []
        
        for file_path in self.episode_files:
            with h5py.File(file_path, 'r') as file:
                data = self.extract_data_from_hdf5(file, keys)
                self.actions.append(data[0])
                self.q_pos.append(data[1])
                # self.oh_images.append(data[2])
                self.field_images.append(data[3])
                # self.wrist_images.append(data[4])
                self.progress.append(data[5])
    
    def extract_data_from_hdf5(self, file, keys):
        data = []
        for key in keys:
            if 'image' in key:
                d = torch.from_numpy(file[key][:]).float() / 255.0
                d = d.permute(0, 3, 1, 2)  # Change to (N, C, H, W) format
                # Resize the image to 224x224
                d = F.interpolate(d, size=(224, 224), mode='bilinear', align_corners=False)
            else:
                d = torch.from_numpy(file[key][:]).float()
            data.append(d)
        return data
    
    def next_batch(self):
        B, T = self.B, self.T
        # Select random indices for each batch
        indices = np.random.choice(self.num_episodes, size=B, replace=True)
        batch_action = []
        batch_q_pos = []
        batch_images = []
        batch_next_progress = []
        # Target for WorldModel
        batch_next_images = []
        
        for i in indices:
            # For each data type, select a random starting point and extract T frames
            episode_length = self.actions[i].shape[0]
            max_start = max(0, episode_length - T - 1)
            start = np.random.randint(0, max_start)
            
            # action = self.actions[i][start+T].squeeze()
            action = self.actions[i][start+1:start+T+1]
            q_pos = self.q_pos[i][start].squeeze()
            # oh_images = self.oh_images[i][start].unsqueeze(0)
            field_images = self.field_images[i][start].unsqueeze(0)
            # wrist_images = self.wrist_images[i][start].unsqueeze(0)
            # images = torch.cat([oh_images, field_images, wrist_images], dim=0)
            images = torch.cat([field_images], dim=0)
            # progress = self.progress[i][start].unsqueeze(0)
            next_progress = self.progress[i][start+1:start+T+1].unsqueeze(-1)
            
            # Target for WorldModel. Overhead image at start + T
            next_images = self.field_images[i][start+T].unsqueeze(0)
            
            batch_action.append(action)
            batch_q_pos.append(q_pos)
            batch_images.append(images)
            batch_next_progress.append(next_progress)
            batch_next_images.append(next_images)
        
        # Stack the batches
        batch_action = torch.stack(batch_action)
        batch_q_pos = torch.stack(batch_q_pos)
        batch_images = torch.stack(batch_images)
        batch_next_progress = torch.stack(batch_next_progress)
        batch_next_images = torch.stack(batch_next_images)
        
        return batch_action, batch_q_pos, batch_images, batch_next_progress, batch_next_images
    

class SimpleFSQAutoEncoder(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(256, len(levels), kernel_size=1),
        )
        
        self.fsq = FSQ(levels)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(len(levels), 256, kernel_size=1),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x, indices = self.fsq(x)
        x = self.decoder(x)
        return x.clamp(-1, 1), indices


def train(model, train_loader, opt, num_codes, train_iterations=10000, device="cuda"):


    for step in (pbar := trange(train_iterations)):
        opt.zero_grad()
        _, _, x, _, _ = train_loader.next_batch()
        x = x.to(device)
        x = x.squeeze()
        out, indices = model(x)
        rec_loss = (out - x).abs().mean()
        rec_loss.backward()

        opt.step()
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )

        # Save every 100000 steps
        if step > 100000 and step % 100000 == 0:
            torch.save(model.state_dict(), f"fsq_vqvae_{step}.pt")
    
    torch.save(model.state_dict(), "fsq_vqvae.pt")

    return


# Update the main execution
if __name__ == "__main__":
    dotrain = False
    if dotrain:
        lr = 3e-4
        train_iter = 100000
        levels = [8, 6, 5]  # target size 2^8, actual size 240
        num_codes = math.prod(levels)

        seed = 42
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == "cuda", "use geep"

        # Load data
        dataset_dir = "/home/qutrll/data/pot_pick_place_2_10hz/"
        episodes = os.listdir(dataset_dir)
        num_episodes = len(episodes)
        num_episodes = 1
        train_episodes = int(num_episodes)
        train_indices = np.random.choice(num_episodes, size=train_episodes, replace=False)
        train_loader = DataLoaderLite(dataset_dir, 8, 1, 'train', train_indices)

        print("Training 224x224 FSQ Autoencoder")
        torch.random.manual_seed(seed)
        model = SimpleFSQAutoEncoder(levels).to(device)
        model = torch.compile(model)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        train(model, train_loader, opt, num_codes, train_iterations=train_iter, device=device)
    
    else:
        seed = 42
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == "cuda", "use geep"

        levels = [8, 6, 5]  # target size 2^8, actual size 240
        num_codes = math.prod(levels)

        model = SimpleFSQAutoEncoder(levels).to(device)
        model = torch.compile(model)
        model.load_state_dict(torch.load("fsq_vqvae.pt"))
        model.eval()

        # Dataloader
        dataset_dir = "/home/qutrll/data/pot_pick_place_2_10hz/"
        episodes = os.listdir(dataset_dir)
        num_episodes = len(episodes)
        num_episodes = 1
        train_episodes = int(num_episodes)
        train_indices = np.random.choice(num_episodes, size=train_episodes, replace=False)
        train_loader = DataLoaderLite(dataset_dir, 8, 1, 'train', train_indices)

        # Get a batch
        while True:
            _, _, x, _, _ = train_loader.next_batch()
            x = x.to(device)
            x = x.squeeze()

            # Forward pass
            out, _ = model(x)
            print(out.shape)

            out = out[0].permute(1, 2, 0).cpu().detach().numpy()
            x = x[0].permute(1, 2, 0).cpu().detach().numpy()

            # Visualise
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(x, cmap="gray")
            plt.subplot(1, 2, 2)
            plt.title("Reconstructed")
            plt.imshow(out, cmap="gray")
            plt.show()

            input("Press Enter to continue...")

