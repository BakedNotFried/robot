import numpy as np
import matplotlib.pyplot as plt
import lpips
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio
import os
import pdb

policy_type = "hit_dm"
trial_type = "OODHD"
data_dir = f"/home/qutrll/data/experiments/pot/{policy_type}/{trial_type}/"
experiment_nums = os.listdir(data_dir)
assert len(experiment_nums) > 0
print(experiment_nums)

# Save dir
save_dir = f"/home/qutrll/data/experiments/pot/{policy_type}/results/{trial_type}/"
os.makedirs(save_dir, exist_ok=True)
print(save_dir)

image_actual_name = "image_actual_data.npy"
image_predicted_name = "image_predicted_data.npy"
progress_data_name = "progress_data.npy"

# Initialize lists to store metrics for all experiments
all_lpips = []
all_lpips_scaled = []
all_psnr = []
all_psnr_scaled = []
all_progress = []
all_progress_gradient = []
all_progress_gradient_gradient = []

# Initialize LPIPS loss function
lpips_loss_fn = lpips.LPIPS(net='alex').cuda()

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

for e_num in experiment_nums:
    # Load data
    image_actual = np.load(os.path.join(data_dir, e_num, image_actual_name))
    image_actual = image_actual[1:]
    image_predicted = np.load(os.path.join(data_dir, e_num, image_predicted_name))
    image_predicted = image_predicted[0:-1]
    progress_data = np.load(os.path.join(data_dir, e_num, progress_data_name))
    progress_data = progress_data[0:-1]
    
    # Calculate LPIPS
    lpips_values = []
    for i in range(len(image_actual)):
        actual = torch.from_numpy(image_actual[i]).cuda()
        predicted = torch.from_numpy(image_predicted[i]).cuda()
        lpips_value = lpips_loss_fn(normalize_to_neg_one_to_one(actual), normalize_to_neg_one_to_one(predicted)).item()
        lpips_values.append(lpips_value)
    
    # Calculate PSNR
    psnr_values = []
    for i in range(len(image_actual)):
        actual = torch.from_numpy(image_actual[i])
        predicted = torch.from_numpy(image_predicted[i])
        psnr_value = peak_signal_noise_ratio(predicted, actual).item()
        psnr_values.append(psnr_value)
    
    # Calculate Progress Gradient
    progress_gradient = np.gradient(progress_data)
    
    # Calculate Progress Gradient Gradient
    progress_gradient_gradient = np.gradient(progress_gradient)
    
    # Scale LPIPS and PSNR to 0-1
    lpips_scaled = (lpips_values) / (np.max(lpips_values))
    psnr_scaled = (psnr_values) / (np.max(psnr_values))
    
    # Append metrics for this experiment
    all_lpips.append(lpips_values)
    all_lpips_scaled.append(lpips_scaled)
    all_psnr.append(psnr_values)
    all_psnr_scaled.append(psnr_scaled)
    all_progress.append(progress_data)
    all_progress_gradient.append(progress_gradient)
    all_progress_gradient_gradient.append(progress_gradient_gradient)

# Calculate averages and variances across all experiments
avg_lpips = np.mean(all_lpips, axis=0)
var_lpips = np.var(all_lpips, axis=0)
avg_lpips_scaled = np.mean(all_lpips_scaled, axis=0)
var_lpips_scaled = np.var(all_lpips_scaled, axis=0)
avg_psnr = np.mean(all_psnr, axis=0)
var_psnr = np.var(all_psnr, axis=0)
avg_psnr_scaled = np.mean(all_psnr_scaled, axis=0)
var_psnr_scaled = np.var(all_psnr_scaled, axis=0)
avg_progress = np.mean(all_progress, axis=0)
var_progress = np.var(all_progress, axis=0)
avg_progress_gradient = np.mean(all_progress_gradient, axis=0)
var_progress_gradient = np.var(all_progress_gradient, axis=0)
avg_progress_gradient_gradient = np.mean(all_progress_gradient_gradient, axis=0)
var_progress_gradient_gradient = np.var(all_progress_gradient_gradient, axis=0)

# Save results
np.save(os.path.join(save_dir, 'avg_lpips.npy'), avg_lpips)
np.save(os.path.join(save_dir, 'var_lpips.npy'), var_lpips)
np.save(os.path.join(save_dir, 'avg_lpips_scaled.npy'), avg_lpips_scaled)
np.save(os.path.join(save_dir, 'var_lpips_scaled.npy'), var_lpips_scaled)
np.save(os.path.join(save_dir, 'avg_psnr.npy'), avg_psnr)
np.save(os.path.join(save_dir, 'var_psnr.npy'), var_psnr)
np.save(os.path.join(save_dir, 'avg_psnr_scaled.npy'), avg_psnr_scaled)
np.save(os.path.join(save_dir, 'var_psnr_scaled.npy'), var_psnr_scaled)
np.save(os.path.join(save_dir, 'avg_progress.npy'), avg_progress)
np.save(os.path.join(save_dir, 'var_progress.npy'), var_progress)
np.save(os.path.join(save_dir, 'avg_progress_gradient.npy'), avg_progress_gradient)
np.save(os.path.join(save_dir, 'var_progress_gradient.npy'), var_progress_gradient)
np.save(os.path.join(save_dir, 'avg_progress_gradient_gradient.npy'), avg_progress_gradient_gradient)
np.save(os.path.join(save_dir, 'var_progress_gradient_gradient.npy'), var_progress_gradient_gradient)

print("Results saved successfully!")
