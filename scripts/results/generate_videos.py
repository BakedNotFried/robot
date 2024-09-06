import numpy as np
import cv2
import os
import pdb

policy_type = "hit_dm"
trial_type = "OODHD"
data_dir = f"/home/qutrll/data/experiments/pot/{policy_type}/{trial_type}/"
experiment_nums = os.listdir(data_dir)
assert len(experiment_nums) > 0
print(experiment_nums)

def create_side_by_side_video(actual_images, predicted_images, output_path, fps=2):
    height, width = actual_images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))

    for actual, predicted in zip(actual_images, predicted_images):
        # Convert from float to uint8 if necessary
        if actual.dtype != np.uint8:
            actual = (actual * 255).astype(np.uint8)
        if predicted.dtype != np.uint8:
            predicted = (predicted * 255).astype(np.uint8)
        
        # Ensure images are in BGR format for OpenCV
        if actual.shape[2] == 3:
            actual = cv2.cvtColor(actual, cv2.COLOR_RGB2BGR)
        if predicted.shape[2] == 3:
            predicted = cv2.cvtColor(predicted, cv2.COLOR_RGB2BGR)

        # Assert 3 channels and dtype uint8
        assert actual.shape[2] == 3 and actual.dtype == np.uint8
        assert predicted.shape[2] == 3 and predicted.dtype == np.uint8

        # Create side-by-side image
        side_by_side = np.hstack((actual, predicted))

        # Add labels
        cv2.putText(side_by_side, 'Actual', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side_by_side, 'Predicted', (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(side_by_side)

    out.release()

for e_num in experiment_nums:
    # Load data
    image_actual = np.load(os.path.join(data_dir, e_num, "image_actual_data.npy"))
    image_actual = image_actual[1:]  # Remove the first frame as before
    image_predicted = np.load(os.path.join(data_dir, e_num, "image_predicted_data.npy"))
    image_predicted = image_predicted[:-1]  # Remove the last frame as before

    image_actual = np.transpose(image_actual.squeeze(), (0, 2, 3, 1))
    image_predicted = np.transpose(image_predicted.squeeze(), (0, 2, 3, 1))

    # Create and save the video
    output_path = os.path.join(data_dir, e_num, f"experiment_{e_num}_video.mp4")
    create_side_by_side_video(image_actual, image_predicted, output_path)
    
    print(f"Video created for experiment {e_num}: {output_path}")

print("All videos have been created successfully!")
