import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import pdb

class DataLoaderLite:
    def __init__(self, data_dir1, data_dir2, B, T, split, indices1, indices2):
        self.B = B
        self.T = T
        assert split in ['train', 'val']
        
        episodes1 = os.listdir(data_dir1)
        episodes1 = [episodes1[i] for i in indices1]
        print(f"Loading {split} episodes: {episodes1}")
        self.episode_files1 = [os.path.join(data_dir1, episode) for episode in episodes1]
        self.num_episodes1 = len(self.episode_files1)
        assert self.num_episodes1 > 0, "No episodes found"
        print(f"Found {self.num_episodes1} episodes")
        episodes2 = os.listdir(data_dir2)
        episodes2 = [episodes2[i] for i in indices2]
        print(f"Loading {split} episodes: {episodes2}")
        self.episode_files2 = [os.path.join(data_dir2, episode) for episode in episodes2]
        self.num_episodes2 = len(self.episode_files2)
        assert self.num_episodes2 > 0, "No episodes found"
        print(f"Found {self.num_episodes2} episodes")

        self.episode_files = self.episode_files1 + self.episode_files2
        self.num_episodes = self.num_episodes1 + self.num_episodes2
        
        keys = ['action', 'q_pos', 'field_images', 'progress']

        # Load all episodes into memory
        self.actions = []
        self.q_pos = []
        self.field_images = []
        self.progress = []
        self.source_indices = []

        for file_path in self.episode_files:
            with h5py.File(file_path, 'r') as file:
                data = self.extract_data_from_hdf5(file, keys)
                self.actions.append(data[0])
                self.q_pos.append(data[1])
                self.field_images.append(data[2])
                self.progress.append(data[3])
                if data_dir1 in file_path:
                    self.source_indices.append(0)
                else:
                    self.source_indices.append(1)

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
        batch_next_images = []
        batch_source_indices = []  # New list to store source indices
        
        for i in indices:
            # For each data type, select a random starting point and extract T frames
            episode_length = self.actions[i].shape[0]
            max_start = max(0, episode_length - T - 1)
            start = np.random.randint(0, max_start)
            
            action = self.actions[i][start+1:start+T+1]
            q_pos = self.q_pos[i][start].squeeze()
            current_field_images = self.field_images[i][start]
            previous_field_images = self.field_images[i][start - 5]
            images = torch.stack([previous_field_images, current_field_images], dim=0)
            next_progress = self.progress[i][start+1:start+T+1].unsqueeze(-1)
            next_images = self.field_images[i][start+T].unsqueeze(0)
            source_idx = self.source_indices[i]  # Get the source index for this episode
            
            batch_action.append(action)
            batch_q_pos.append(q_pos)
            batch_images.append(images)
            batch_next_progress.append(next_progress)
            batch_next_images.append(next_images)
            batch_source_indices.append(source_idx)  # Add source index to the batch
        
        # Stack the batches
        batch_action = torch.stack(batch_action)
        batch_q_pos = torch.stack(batch_q_pos)
        batch_images = torch.stack(batch_images)
        batch_next_progress = torch.stack(batch_next_progress)
        batch_next_images = torch.stack(batch_next_images)
        batch_source_indices = torch.tensor(batch_source_indices)  # Convert to tensor

        return batch_action, batch_q_pos, batch_images, batch_next_progress, batch_next_images, batch_source_indices
