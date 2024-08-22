import torch
from torch import nn
from torchvision import models

class WorldModelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder for images
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        encoder = models.convnext_tiny(weights=weights)
        encoder.classifier = nn.Identity()
        self.encoder = encoder

        # Decode latent to image
        self.decoder = nn.Sequential(
            nn.Linear(2319, 512 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, images, action, q_pos, progress):
        # Encode the images and combine with action, q_pos and progress
        batch_size, num_images, channels, height, width = images.shape
        reshaped_images = images.view(batch_size * num_images, channels, height, width)
        encoded_features = self.encoder(reshaped_images).squeeze()
        feature_size = encoded_features.shape[-1]
        encoded_features = encoded_features.view(batch_size, num_images * feature_size)
        combined_features = torch.cat([encoded_features, action, q_pos, progress], dim=1)

        # Decode features to generate the next frame
        next_frame = self.decoder(combined_features)
        return next_frame
