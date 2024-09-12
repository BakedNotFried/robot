import torch
from torch import nn
from torchvision import models
import pdb

class MobilePolicy(nn.Module):
    def __init__(self, seq_len=10):
        super().__init__()

        # Define the encoder
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        encoder = models.convnext_tiny(weights=weights)
        encoder.classifier = nn.Identity()
        self.encoder = encoder
        
        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 8 * seq_len),
        )

        # q_pos embedding
        self.q_pos_embedding = nn.Sequential(
            nn.Linear(7, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        # Task embedding
        self.task_embedding = nn.Embedding(2, 512)

        # Task MLP
        self.task_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, images, q_pos, task_index):

        # Encode the images and combine with q_pos
        batch_size, num_images, channels, height, width = images.shape
        reshaped_images = images.view(batch_size * num_images, channels, height, width)
        visual_features = self.encoder(reshaped_images).squeeze()
        feature_size = visual_features.shape[-1]
        visual_features = visual_features.view(batch_size, num_images * feature_size)
        # Embed q_pos
        q_pos_features = self.q_pos_embedding(q_pos)
        # Embed task
        task_embedding = self.task_embedding(task_index)
        task_embedding = self.task_mlp(task_embedding)

        combined_features = torch.cat([visual_features, q_pos_features, task_embedding], dim=1)


        # Pass combined features through the MLP
        output = self.mlp(combined_features)

        # Output is action + progress (B, seq_len, 8)
        output = output.view(batch_size, -1, 8)


        return output
    