import numpy as np
import matplotlib.pyplot as plt
import os

policy_type = "hit_dm"
trial_types = ["IND", "OODVD", "OODHD"]
for trial_type in trial_types:
    results_dir = f"/home/qutrll/data/experiments/pot/{policy_type}/results/{trial_type}/"

    # Load the data
    avg_lpips_alex = np.load(os.path.join(results_dir, 'avg_lpips_alex.npy'))
    var_lpips_alex = np.load(os.path.join(results_dir, 'var_lpips_alex.npy'))
    avg_lpips_vgg = np.load(os.path.join(results_dir, 'avg_lpips_vgg.npy'))
    var_lpips_vgg = np.load(os.path.join(results_dir, 'var_lpips_vgg.npy'))
    avg_psnr = np.load(os.path.join(results_dir, 'avg_psnr.npy'))
    var_psnr = np.load(os.path.join(results_dir, 'var_psnr.npy'))
    avg_progress = np.load(os.path.join(results_dir, 'avg_progress.npy'))
    var_progress = np.load(os.path.join(results_dir, 'var_progress.npy'))
    avg_progress_gradient = np.load(os.path.join(results_dir, 'avg_progress_gradient.npy'))
    var_progress_gradient = np.load(os.path.join(results_dir, 'var_progress_gradient.npy'))
    avg_progress_gradient_gradient = np.load(os.path.join(results_dir, 'avg_progress_gradient_gradient.npy'))
    var_progress_gradient_gradient = np.load(os.path.join(results_dir, 'var_progress_gradient_gradient.npy'))

    # Create a figure with subplots
    fig, axs = plt.subplots(4, 2, figsize=(20, 35))
    fig.suptitle(f'Metrics for {policy_type} - {trial_type}', fontsize=16)

    # Helper function to plot data with variance envelope
    def plot_with_variance(ax, x, y, var, title, color):
        ax.plot(x, y, color=color, linewidth=2)
        ax.fill_between(x, y - np.sqrt(var), y + np.sqrt(var), color=color, alpha=0.2)
        ax.set_title(title)
        ax.grid(True)

    # Plot individual metrics
    plot_with_variance(axs[0, 0], range(len(avg_lpips_alex)), avg_lpips_alex, var_lpips_alex, 'LPIPS Alex', 'blue')
    plot_with_variance(axs[0, 1], range(len(avg_lpips_vgg)), avg_lpips_vgg, var_lpips_vgg, 'LPIPS VGG', 'green')
    plot_with_variance(axs[1, 0], range(len(avg_psnr)), avg_psnr, var_psnr, 'PSNR', 'red')
    plot_with_variance(axs[1, 1], range(len(avg_progress)), avg_progress, var_progress, 'Progress', 'orange')
    plot_with_variance(axs[2, 0], range(len(avg_progress_gradient)), avg_progress_gradient, var_progress_gradient, 'Progress Gradient', 'brown')
    plot_with_variance(axs[2, 1], range(len(avg_progress_gradient_gradient)), avg_progress_gradient_gradient, var_progress_gradient_gradient, 'Progress Gradient Gradient', 'pink')

    # Plot all metrics together with variance envelope
    axs[3, 0].plot(range(len(avg_lpips_alex)), avg_lpips_alex, color='blue', label='LPIPS Alex')
    axs[3, 0].fill_between(range(len(avg_lpips_alex)), avg_lpips_alex - np.sqrt(var_lpips_alex), avg_lpips_alex + np.sqrt(var_lpips_alex), color='blue', alpha=0.2)

    axs[3, 0].plot(range(len(avg_lpips_vgg)), avg_lpips_vgg, color='green', label='LPIPS VGG')
    axs[3, 0].fill_between(range(len(avg_lpips_vgg)), avg_lpips_vgg - np.sqrt(var_lpips_vgg), avg_lpips_vgg + np.sqrt(var_lpips_vgg), color='green', alpha=0.2)

    axs[3, 0].plot(range(len(avg_psnr)), avg_psnr/100, color='red', label='PSNR (scaled)')
    axs[3, 0].fill_between(range(len(avg_psnr)), avg_psnr/100 - np.sqrt(var_psnr)/100, avg_psnr/100 + np.sqrt(var_psnr)/100, color='red', alpha=0.2)

    axs[3, 0].plot(range(len(avg_progress)), avg_progress, color='orange', label='Progress')
    axs[3, 0].fill_between(range(len(avg_progress)), avg_progress - np.sqrt(var_progress), avg_progress + np.sqrt(var_progress), color='orange', alpha=0.2)

    axs[3, 0].plot(range(len(avg_progress_gradient)), avg_progress_gradient, color='brown', label='Progress Gradient')
    axs[3, 0].fill_between(range(len(avg_progress_gradient)), avg_progress_gradient - np.sqrt(var_progress_gradient), avg_progress_gradient + np.sqrt(var_progress_gradient), color='brown', alpha=0.2)

    axs[3, 0].plot(range(len(avg_progress_gradient_gradient)), avg_progress_gradient_gradient, color='pink', label='Progress Gradient Gradient')
    axs[3, 0].fill_between(range(len(avg_progress_gradient_gradient)), avg_progress_gradient_gradient - np.sqrt(var_progress_gradient_gradient), avg_progress_gradient_gradient + np.sqrt(var_progress_gradient_gradient), color='pink', alpha=0.2)

    axs[3, 0].set_title('All Metrics')
    axs[3, 0].set_ylim(-0.1, 1.1)
    axs[3, 0].grid(True)
    axs[3, 0].legend()

    # Remove the empty subplot
    fig.delaxes(axs[3, 1])

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, f'{policy_type}_{trial_type}_metrics.png'))
    figures_dir = f"/home/qutrll/data/experiments/pot/{policy_type}/results/figures/"
    plt.savefig(os.path.join(figures_dir, f'{policy_type}_{trial_type}_metrics.png'))
    plt.close()

    print(f"Plot saved as {policy_type}_{trial_type}_metrics.png in {results_dir}")
