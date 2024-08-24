import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F

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
                d = d.permute(0, 3, 1, 2)
                # Resize the image to 224x224
                # d = F.interpolate(d, size=(224, 224), mode='bilinear', align_corners=False)
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
        batch_progress = []
        batch_next_progress = []
        
        for i in indices:
            # For each data type, select a random starting point and extract T frames
            episode_length = self.actions[i].shape[0]
            max_start = max(0, episode_length - T - 1)
            
            start = np.random.randint(0, max_start + 1)
            action = self.actions[i][start+T].squeeze()
            q_pos = self.q_pos[i][start].squeeze()
            # oh_images = self.oh_images[i][start].unsqueeze(0)
            field_images = self.field_images[i][start].unsqueeze(0)
            # wrist_images = self.wrist_images[i][start].unsqueeze(0)
            images = torch.cat([field_images], dim=0)
            # images = torch.cat([oh_images, field_images, wrist_images], dim=0)
            progress = self.progress[i][start].unsqueeze(0)
            next_progress = self.progress[i][start+T].unsqueeze(0)
            
            batch_action.append(action)
            batch_q_pos.append(q_pos)
            batch_images.append(images)
            batch_progress.append(progress)
            batch_next_progress.append(next_progress)
        
        # Stack the batches
        batch_action = torch.stack(batch_action)
        batch_q_pos = torch.stack(batch_q_pos)
        batch_images = torch.stack(batch_images)
        batch_progress = torch.stack(batch_progress)
        batch_next_progress = torch.stack(batch_next_progress)
        
        return batch_action, batch_q_pos, batch_images, batch_progress, batch_next_progress
