import numpy as np
import matplotlib.pyplot as plt

# Load loss values
loss_values = np.load('/home/qutrll/data/checkpoints/multi_task_mobile/1/loss_seed_42.npy')
loss_values = loss_values[10000:]

# Plot loss values
plt.plot(loss_values)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss vs Steps')
plt.show()
