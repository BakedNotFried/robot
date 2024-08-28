import torch
import numpy as np
import json

from model_util import make_policy
import random
import pdb

def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


if __name__ == '__main__':
    # Policy Setup
    policy_class = 'HIT'
    config_dir = "/home/qutrll/data/checkpoints/HIT/pot_pick_place/_pot_pick_place_HIT_resnet18_True/all_configs.json"
    config = json.load(open(config_dir))
    policy_config = config['policy_config']

    seed_everything(42)

    policy = make_policy(policy_class, policy_config)
    policy.eval()
    loading_status = policy.deserialize(torch.load("/home/qutrll/data/checkpoints/HIT/pot_pick_place/_pot_pick_place_HIT_resnet18_True/policy_step_100000_seed_42.ckpt", map_location='cuda'))
    if not loading_status:
        print(f'Failed to load policy_last.ckpt')
    policy.cuda()
        

    # Dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is not available"
    
    while True:
        # Inference

        q_pos = torch.rand(1, 7).to(device)
        images = torch.rand(1, 1, 3, 240, 424).to(device)        
        with torch.no_grad():
            output = policy.forward_inf(q_pos, images)

        print(output)
        print(output.shape)
        input("Press Enter to continue...")
    
 