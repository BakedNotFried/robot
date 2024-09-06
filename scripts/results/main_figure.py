import numpy as np
import matplotlib.pyplot as plt
import lpips
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio
import os
import pdb

policy_type = "hit_dm"
trial_types = ["IND", "OODVD", "OODHD"]

# Initialize LPIPS loss function
lpips_loss_fn_alex = lpips.LPIPS(net='alex').cuda()
lpips_loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

for trial_type in trial_types:
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
    all_lpips_alex = []
    all_lpips_vgg = []
    all_psnr = []
    all_progress = []
    all_progress_gradient = []
    all_progress_gradient_gradient = []

    for e_num in experiment_nums:
        # Load data
        image_actual = np.load(os.path.join(data_dir, e_num, image_actual_name))
        image_actual = image_actual[1:]
        image_predicted = np.load(os.path.join(data_dir, e_num, image_predicted_name))
        image_predicted = image_predicted[0:-1]
        progress_data = np.load(os.path.join(data_dir, e_num, progress_data_name))
        progress_data = progress_data[0:-1]
        
        # Calculate LPIPS
        lpips_values_alex = []
        for i in range(len(image_actual)):
            actual = torch.from_numpy(image_actual[i]).cuda()
            predicted = torch.from_numpy(image_predicted[i]).cuda()
            lpips_value = lpips_loss_fn_alex(normalize_to_neg_one_to_one(actual), normalize_to_neg_one_to_one(predicted)).item()
            lpips_values_alex.append(lpips_value)

        lpips_values_vgg = []
        for i in range(len(image_actual)):
            actual = torch.from_numpy(image_actual[i]).cuda()
            predicted = torch.from_numpy(image_predicted[i]).cuda()
            lpips_value = lpips_loss_fn_vgg(normalize_to_neg_one_to_one(actual), normalize_to_neg_one_to_one(predicted)).item()
            lpips_values_vgg.append(lpips_value)
        
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
        
        # Append metrics for this experiment
        all_lpips_alex.append(lpips_values_alex)
        all_lpips_vgg.append(lpips_values_vgg)
        all_psnr.append(psnr_values)
        all_progress.append(progress_data)
        all_progress_gradient.append(progress_gradient)
        all_progress_gradient_gradient.append(progress_gradient_gradient)

    # Calculate averages and variances across all experiments
    avg_lpips_alex = np.mean(all_lpips_alex, axis=0)
    var_lpips_alex = np.var(all_lpips_alex, axis=0)
    avg_lpips_vgg = np.mean(all_lpips_vgg, axis=0)
    var_lpips_vgg = np.var(all_lpips_vgg, axis=0)
    avg_psnr = np.mean(all_psnr, axis=0)
    var_psnr = np.var(all_psnr, axis=0)
    avg_progress = np.mean(all_progress, axis=0)
    var_progress = np.var(all_progress, axis=0)
    avg_progress_gradient = np.mean(all_progress_gradient, axis=0)
    var_progress_gradient = np.var(all_progress_gradient, axis=0)
    avg_progress_gradient_gradient = np.mean(all_progress_gradient_gradient, axis=0)
    var_progress_gradient_gradient = np.var(all_progress_gradient_gradient, axis=0)

    # Save results
    np.save(os.path.join(save_dir, 'avg_lpips_alex.npy'), avg_lpips_alex)
    np.save(os.path.join(save_dir, 'var_lpips_alex.npy'), var_lpips_alex)
    np.save(os.path.join(save_dir, 'avg_lpips_vgg.npy'), avg_lpips_vgg)
    np.save(os.path.join(save_dir, 'var_lpips_vgg.npy'), var_lpips_vgg)
    np.save(os.path.join(save_dir, 'avg_psnr.npy'), avg_psnr)
    np.save(os.path.join(save_dir, 'var_psnr.npy'), var_psnr)
    np.save(os.path.join(save_dir, 'avg_progress.npy'), avg_progress)
    np.save(os.path.join(save_dir, 'var_progress.npy'), var_progress)
    np.save(os.path.join(save_dir, 'avg_progress_gradient.npy'), avg_progress_gradient)
    np.save(os.path.join(save_dir, 'var_progress_gradient.npy'), var_progress_gradient)
    np.save(os.path.join(save_dir, 'avg_progress_gradient_gradient.npy'), avg_progress_gradient_gradient)
    np.save(os.path.join(save_dir, 'var_progress_gradient_gradient.npy'), var_progress_gradient_gradient)

    print("Results saved successfully!")
