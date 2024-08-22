import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# DEBUG
debug = False
hpc = False

# Set random seed
seed = 1337

# Device settings
if hpc:
    device = "cuda:3"
else:
    device = "cuda"

# Directories
if hpc:
    checkpoint_dir = config['checkpoint_dir']
    dataset_dir = config['dataset_dir']
else:
    dataset_dir = config['local_dataset_dir']
    checkpoint_dir = config['local_checkpoint_dir']

# Training parameters
B = config['batch_size']
T = config['sequence_length']
split = config['split']
max_steps = config['max_steps']
validation_interval = config['validation_interval']
checkpoint_interval = config['checkpoint_interval']
validation_steps = config['validation_steps']
lr = float(config['learning_rate'])
resume_training = config['resume_training']
resume_step = config['resume_step']
