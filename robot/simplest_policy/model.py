import torch
from torch import nn
import pdb

class Policy(nn.Module):
    def __init__(self, seq_len=10):
        super().__init__()

        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(301312, 512),
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
            nn.Linear(7, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, images, q_pos):

        # Images are shape B, 2, 3, 224, 224
        # Flateen the images
        batch_size = images.shape[0]
        images = images.view(batch_size, -1)

        # Embed q_pos
        q_pos = self.q_pos_embedding(q_pos)

        combined_features = torch.cat([images, q_pos], dim=1)

        # Pass combined features through the MLP
        output = self.mlp(combined_features)

        # Output is action + progress (B, seq_len, 8)
        output = output.view(batch_size, -1, 8)

        return output
    