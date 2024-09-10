import torch
from torch import nn
import pdb

class WorldModel(nn.Module):
    def __init__(self, latent_dim=256, action_dim=8):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512 * 7 * 7),
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

    def forward(self, latents, actions):
        b, t, n = actions.shape
        actions = actions.reshape(b, t * n)
        combined = torch.cat([latents.squeeze(), actions.squeeze()], dim=-1)
        if len(combined.shape) == 1:
            combined = combined.unsqueeze(0)
        return self.decoder(combined)
