# Dataset settings
dataset_dir = "/home/qutrll/data/pot_pick_place_2_10hz/"
batch_size = 8
sequence_length = 10
split = 0.95

# Model settings
learning_rate = 3e-5

# Training settings
resume_training = False
resume_step = 0
max_steps = 200001
validation_interval = 1000
validation_steps = 100
checkpoint_interval = 20000
checkpoint_dir = "/home/qutrll/data/checkpoints/multi_task_mobile/1/"

# Architecture settings
architecture = "convnext_tiny + MLP"

# DEBUG
debug = False
use_compile = False
hpc = False

# Set random seed
seed = 42

# Device settings
device = "cuda:3" if hpc else "cuda"

# Directories
dataset_dir1 = "/home/qutrll/data/cupboard_10hz/"
dataset_dir2 = "/home/qutrll/data/wipe_10hz/"

# Training parameters (redundant with above, but kept for consistency with original code)
B = batch_size
T = sequence_length
lr = learning_rate
