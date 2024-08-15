#!/usr/bin/env python3

import time

from robot.robot_utils import (
    ImageRecorder,
    move_arms,
    move_grippers,
    Recorder,
    setup_follower_bot,
    setup_leader_bot,
)

from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

if __name__ == '__main__':
    node = create_interbotix_global_node('robot')
    image_recorder = ImageRecorder(is_debug=True, node=node)
    while True:
        print(image_recorder.print_diagnostics())
        time.sleep(1)

