import torch
from torch import nn
from torchvision import models

class PolicyCNNMLP(nn.Module):
    def __init__(self, seq_len=10):
        super().__init__()

        # Define the encoder
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        encoder = models.convnext_tiny(weights=weights)
        encoder.classifier = nn.Identity()
        self.encoder = encoder
        
        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.Linear(2312, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 8 * seq_len),
        )

        # q_pos embedding
        self.q_pos_embedding = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, images, q_pos):

        # Encode the images and combine with q_pos
        batch_size, num_images, channels, height, width = images.shape
        reshaped_images = images.view(batch_size * num_images, channels, height, width)
        encoded_features = self.encoder(reshaped_images).squeeze()
        feature_size = encoded_features.shape[-1]
        encoded_features = encoded_features.view(batch_size, num_images * feature_size)

        # Embed q_pos
        q_pos = self.q_pos_embedding(q_pos)

        combined_features = torch.cat([encoded_features, q_pos], dim=1)

        # Pass combined features through the MLP
        output = self.mlp(combined_features)

        # Output is action + progress (B, seq_len, 8)
        output = output.view(batch_size, -1, 8)

        return output, combined_features
    