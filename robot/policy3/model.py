import torch
from torch import nn
from torchvision import models

class PolicyCNNMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the encoder
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        encoder = models.convnext_tiny(weights=weights)
        encoder.classifier = nn.Identity()
        self.encoder = encoder
        
        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(776, 512),
            # nn.Linear(2312, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 8),
        )

    def forward(self, images, q_pos, progress):

        # Encode the images and combine with q_pos and progress
        batch_size, num_images, channels, height, width = images.shape
        reshaped_images = images.view(batch_size * num_images, channels, height, width)
        encoded_features = self.encoder(reshaped_images).squeeze()
        feature_size = encoded_features.shape[-1]
        encoded_features = encoded_features.view(batch_size, num_images * feature_size)

        combined_features = torch.cat([encoded_features, q_pos, progress], dim=1)

        # Pass combined features through the MLP
        output = self.mlp(combined_features)

        # Output is action + progress (B, 8)

        return output
    