### Task parameters
DATA_DIR = "/home/qutrll/data/data_fold_clothes-20240827T015655Z-001"

TASK_CONFIGS = {
    'data_fold_clothes':{
        'dataset_dir': DATA_DIR + '/data_fold_clothes',
        'camera_names': ['cam_left', 'cam_right'],
        'observation_name': ['qpos','hand_action','wrist'],
        'state_dim':35,
        'action_dim':40,
        'state_mask': [0]*11 + [1]*24,
        'action_mask': [0]*11 + [1]*8 + [0]*5 + [1]*16 #11 for leg, 8 for arm, 5 for imu, 16 for gripper 
    },

    'pot_pick_place':{
    'dataset_dir': DATA_DIR + '/pot_pick_place',
    'camera_names': ['cam_field_prev','cam_field'],
    'observation_name': ['qpos'],
    'state_dim':7,
    'action_dim':8,
    'state_mask': None,
    'action_mask': None,
    },
}
