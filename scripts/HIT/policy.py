import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
from detr.main import build_ACT_model_and_optimizer

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
    
    def __call__(self, qpos, image, actions=None, is_pad=None):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        if self.action_idx is not None:
            actions = actions[:, :, self.action_idx]
            
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
 
            loss_dict = dict()
            a_hat, _, hs_img_dict = self.model(qpos, image)
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1).mean()
            loss_dict['l1'] = l1
            if self.model.feature_loss and self.model.training:
                loss_dict['feature_loss'] = F.mse_loss(hs_img_dict['hs_img'], hs_img_dict['src_future']).mean()
                loss_dict['loss'] = loss_dict['l1'] + self.feature_loss_weight*loss_dict['feature_loss']
            else:
                loss_dict['loss'] = loss_dict['l1']
            return loss_dict
        else:
            a_hat, _, _ = self.model(qpos, image) # no action, sample from prior
            return a_hat

    def forward_inf(self, qpos, image):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        a_hat, _, _ = self.model(qpos, image) # no action, sample from prior
        return a_hat
        
    def configure_optimizers(self):
        return self.optimizer   
        
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)
        


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

