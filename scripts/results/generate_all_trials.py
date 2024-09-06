import numpy as np
import matplotlib.pyplot as plt
import lpips
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio
import os

policy_type = "hit_dm"
trial_type = "IND"
data_dir = f"/home/qutrll/data/experiments/pot/{policy_type}/{trial_type}/"
experiment_nums = os.listdir(data_dir)
results_dir = f"/home/qutrll/data/experiments/pot/{policy_type}/results/{trial_type}/"
os.makedirs(results_dir, exist_ok=True)

# Initialize LPIPS loss function
lpips_loss_fn = lpips.LPIPS(net='alex').cuda()

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def calculate_metrics(image_actual, image_predicted, progress_data):
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
    lpips_scaled = lpips_values / np.max(lpips_values)
    psnr_scaled = psnr_values / np.max(psnr_values)
    
    return lpips_values, lpips_scaled, psnr_values, psnr_scaled, progress_data, progress_gradient, progress_gradient_gradient

# Calculate metrics for each experiment
all_metrics = []
for e_num in experiment_nums:
    image_actual = np.load(os.path.join(data_dir, e_num, "image_actual_data.npy"))[1:]
    image_predicted = np.load(os.path.join(data_dir, e_num, "image_predicted_data.npy"))[:-1]
    progress_data = np.load(os.path.join(data_dir, e_num, "progress_data.npy"))[:-1]
    
    metrics = calculate_metrics(image_actual, image_predicted, progress_data)
    all_metrics.append(metrics)

# Create a figure with subplots
fig, axs = plt.subplots(4, 2, figsize=(20, 30))
fig.suptitle(f'Metrics for {policy_type} - {trial_type}', fontsize=16)

# Helper function to plot data for all experiments
def plot_all_experiments(ax, data, title, color):
    for exp_data in data:
        ax.plot(range(len(exp_data)), exp_data, color=color, alpha=0.3)
    ax.set_title(title)
    ax.grid(True)

# Plot data for each metric
metrics = [
    ("LPIPS", 0, "blue"),
    ("LPIPS Scaled", 1, "green"),
    ("PSNR", 2, "red"),
    ("PSNR Scaled", 3, "purple"),
    ("Progress", 4, "orange"),
    ("Progress Gradient", 5, "brown"),
    ("Progress Gradient Gradient", 6, "pink")
]

for i, (title, metric_index, color) in enumerate(metrics):
    row = i // 2
    col = i % 2
    plot_all_experiments(axs[row, col], [m[metric_index] for m in all_metrics], title, color)

# Remove the empty subplot
axs[3, 1].remove()

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, f'{policy_type}_{trial_type}_all_experiments_metrics.png'))
plt.close()

print(f"Plot saved as {policy_type}_{trial_type}_all_experiments_metrics.png in {results_dir}")
