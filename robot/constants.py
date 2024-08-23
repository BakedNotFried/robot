import os

DT = 0.02
try:
    from rclpy.duration import Duration
    from rclpy.constants import S_TO_NS
    DT_DURATION = Duration(seconds=0, nanoseconds=DT * S_TO_NS)
except ImportError:
    pass

COLOR_IMAGE_TOPIC_NAME = '/{}/camera/color/image_raw'
DATA_DIR = os.path.expanduser('~/data')
FPS = 50

JOINT_NAMES = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
START_ARM_POSE = [
    0 , -0.76238847,  0.44485444, -0.01994175,  1.7564081,  -0.15953401, 0.02239, -0.02239
]
SLEEP_ARM_POSE = [0, -1.80, 1.55, 0, 0.8, 0, 0]
ROBOT_GRIPPER_JOINT_OPEN_MAX = 1.9957
ROBOT_GRIPPER_JOINT_OPEN = 1.6214
# ROBOT_GRIPPER_JOINT_CLOSE = 0.6197
ROBOT_GRIPPER_JOINT_CLOSE = 0.0
ROBOT_GRIPPER_JOINT_CLOSE_MAX = -0.4433
# ROBOT_GRIPPER_JOINT_MID = (ROBOT_GRIPPER_JOINT_OPEN + ROBOT_GRIPPER_JOINT_CLOSE) / 2
ROBOT_GRIPPER_JOINT_MID = (ROBOT_GRIPPER_JOINT_OPEN + 0.6197) / 2
