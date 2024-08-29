import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
from detr.main import build_ACT_model_and_optimizer
from torchvision import models

import pdb


class HITPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        args_override['model_type'] = "HIT"
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.feature_loss_weight = args_override['feature_loss_weight'] if 'feature_loss_weight' in args_override else 0.0
        try:
            self.state_idx = args_override['state_idx']
            self.action_idx = args_override['action_idx']
        except:
            self.state_idx = None
            self.action_idx = None
        
        # Create the world model
        self.world_model = WorldModel(latent_dim=512 * 49, action_dim=8 * 10)
    
    def __call__(self, qpos, image, actions=None, next_image=None):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        if self.action_idx is not None:
            actions = actions[:, :, self.action_idx]
            
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
 
            loss_dict = dict()
            # a_hat, _, hs_img_dict = self.model(qpos, image, actions, next_image)
            a_hat, _, hs_img = self.model(qpos, image, actions, next_image)

            # WORLD MODEL
            # Use hs_image to predict next_image.
            # Detach hs_img to prevent gradients from flowing back to the policy network
            hs_img_detached = hs_img.detach()
            
            # Reshape hs_img if necessary (assuming it's [batch_size, num_queries, latent_dim])
            batch_size, num_queries, latent_dim = hs_img_detached.shape
            hs_img_reshaped = hs_img_detached.reshape(batch_size, -1)
            actions_reshaped = actions.reshape(batch_size, -1)
            # Forward pass through world model
            predicted_next_image = self.world_model(hs_img_reshaped, actions_reshaped)
            predicted_next_image = predicted_next_image.unsqueeze(1)

            # World model loss
            world_model_loss = F.mse_loss(predicted_next_image, next_image)
            loss_dict['world_model_loss'] = world_model_loss

            # Policy loss
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            policy_loss = (all_l1).mean()
            loss_dict['policy_loss'] = policy_loss

            # if self.model.feature_loss and self.model.training:
            #     loss_dict['feature_loss'] = F.mse_loss(hs_img_dict['hs_img'], hs_img_dict['src_future']).mean()
            #     loss_dict['loss'] = loss_dict['l1'] + self.feature_loss_weight*loss_dict['feature_loss']
            # else:
            # loss_dict['loss'] = loss_dict['l1']
            return loss_dict
        else:
            a_hat, _, _ = self.model(qpos, image) # no action, sample from prior
            return a_hat

    def forward_inf(self, qpos, image):
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # image = normalize(image)
        a_hat, _, hs_img = self.model(qpos, image, None, None) # no action, sample from prior
        return a_hat, hs_img
    
    def forward_world_model(self, hs_img, actions):
        # Reshape hs_img if necessary (assuming it's [batch_size, num_queries, latent_dim])
        batch_size, num_queries, latent_dim = hs_img.shape
        hs_img_reshaped = hs_img.reshape(batch_size, -1)
        actions_reshaped = actions.reshape(batch_size, -1)
        # Forward pass through world model
        predicted_next_image = self.world_model(hs_img_reshaped, actions_reshaped)
        predicted_next_image = predicted_next_image.unsqueeze(1)
        return predicted_next_image
        
    def configure_optimizers(self):
        return self.optimizer   
        
    def serialize(self):
        return {
            'policy': self.model.state_dict(),
            'world_model': self.world_model.state_dict()
        }

    def deserialize(self, model_dict):
        self.model.load_state_dict(model_dict['policy'])
        self.world_model.load_state_dict(model_dict['world_model'])
        return True
        
class WorldModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
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
        combined = torch.cat([latents, actions], dim=-1)
        return self.decoder(combined)


# Ignore this for now
class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        args_override['model_type'] = "ACT"
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.vq = args_override['vq']
        print(f'KL Weight {self.kl_weight}')
        try:
            self.state_idx = args_override['state_idx']
            self.action_idx = args_override['action_idx']
        except:
            self.state_idx = None
            self.action_idx = None
    

    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        if self.action_idx is not None:
            actions = actions[:, :, self.action_idx]
            
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, env_state, actions, is_pad, vq_sample)
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            if self.vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=vq_sample) # no action, sample from prior
            return a_hat

    def forward_inf(self, qpos, image):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=None) # no action, sample from prior
        return a_hat
        
    def configure_optimizers(self):
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]

        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)

        return binaries
        
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)

