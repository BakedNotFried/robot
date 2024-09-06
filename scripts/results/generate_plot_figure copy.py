import numpy as np
import matplotlib.pyplot as plt
import os

policy_type = "hit_dm"
trial_type = "IND"
results_dir = f"/home/qutrll/data/experiments/pot/{policy_type}/results/{trial_type}/"

# Load the data
avg_lpips = np.load(os.path.join(results_dir, 'avg_lpips.npy'))
var_lpips = np.load(os.path.join(results_dir, 'var_lpips.npy'))
avg_lpips_scaled = np.load(os.path.join(results_dir, 'avg_lpips_scaled.npy'))
var_lpips_scaled = np.load(os.path.join(results_dir, 'var_lpips_scaled.npy'))
avg_psnr = np.load(os.path.join(results_dir, 'avg_psnr.npy'))
var_psnr = np.load(os.path.join(results_dir, 'var_psnr.npy'))
avg_psnr_scaled = np.load(os.path.join(results_dir, 'avg_psnr_scaled.npy'))
var_psnr_scaled = np.load(os.path.join(results_dir, 'var_psnr_scaled.npy'))
avg_progress = np.load(os.path.join(results_dir, 'avg_progress.npy'))
var_progress = np.load(os.path.join(results_dir, 'var_progress.npy'))
avg_progress_gradient = np.load(os.path.join(results_dir, 'avg_progress_gradient.npy'))
var_progress_gradient = np.load(os.path.join(results_dir, 'var_progress_gradient.npy'))
avg_progress_gradient_gradient = np.load(os.path.join(results_dir, 'avg_progress_gradient_gradient.npy'))
var_progress_gradient_gradient = np.load(os.path.join(results_dir, 'var_progress_gradient_gradient.npy'))

# Create a figure with subplots
fig, axs = plt.subplots(4, 2, figsize=(20, 30))
fig.suptitle(f'Metrics for {policy_type} - {trial_type}', fontsize=16)

# Helper function to plot data with variance envelope
def plot_with_variance(ax, x, y, var, title, color):
    ax.plot(x, y, color=color, linewidth=2)
    ax.fill_between(x, y - np.sqrt(var), y + np.sqrt(var), color=color, alpha=0.2)
    ax.set_title(title)
    ax.grid(True)

# Plot LPIPS
plot_with_variance(axs[0, 0], range(len(avg_lpips)), avg_lpips, var_lpips, 'LPIPS', 'blue')

# Plot LPIPS Scaled
plot_with_variance(axs[0, 1], range(len(avg_lpips_scaled)), avg_lpips_scaled, var_lpips_scaled, 'LPIPS Scaled', 'green')

# Plot PSNR
plot_with_variance(axs[1, 0], range(len(avg_psnr)), avg_psnr, var_psnr, 'PSNR', 'red')

# Plot PSNR Scaled
plot_with_variance(axs[1, 1], range(len(avg_psnr_scaled)), avg_psnr_scaled, var_psnr_scaled, 'PSNR Scaled', 'purple')

# Plot Progress
plot_with_variance(axs[2, 0], range(len(avg_progress)), avg_progress, var_progress, 'Progress', 'orange')

# Plot Progress Gradient
plot_with_variance(axs[2, 1], range(len(avg_progress_gradient)), avg_progress_gradient, var_progress_gradient, 'Progress Gradient', 'brown')

# Plot Progress Gradient Gradient
plot_with_variance(axs[3, 0], range(len(avg_progress_gradient_gradient)), avg_progress_gradient_gradient, var_progress_gradient_gradient, 'Progress Gradient Gradient', 'pink')

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, f'{policy_type}_{trial_type}_metrics.png'))
plt.close()

print(f"Plot saved as {policy_type}_{trial_type}_metrics.png in {results_dir}")
