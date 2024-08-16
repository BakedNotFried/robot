import rclpy
import time
from rclpy.executors import SingleThreadedExecutor
from interbotix_common_modules.common_robot.robot import InterbotixRobotNode
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from omni_msgs.msg import OmniState
from interbotix_xs_msgs.msg import JointSingleCommand
from robot.constants import (
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_OPEN_MAX,
    FOLLOWER_GRIPPER_JOINT_OPEN,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    FOLLOWER_GRIPPER_JOINT_CLOSE_MAX,
    FOLLOWER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)
from robot.robot_utils import (
    move_arms,
    move_grippers,
    torque_on,
)

class TeleopRobot(InterbotixRobotNode):
    def __init__(self):
        super().__init__(node_name='teleop_robot')
        
        # Init Robot
        self.robot = InterbotixManipulatorXS(
            robot_model='wx250s',
            robot_name='robot',
            node=self,
            iterative_update_fk=False,
        )
        self.robot_gripper_command = JointSingleCommand(name='gripper')
        time.sleep(1)
        self.robot_joint_states = self.robot.core.joint_states.position
        
        # Fields
        self.omni_state = None  # x, y, z, r, p, y, gripper, lock
        
        # Subs
        self.create_subscription(OmniState, '/omni/state', self.phantom_state_callback, 1)
        
        # Timer
        hz = 50
        dt = 1/hz
        self.timer = self.create_timer(dt, self.control)
        
        # self.opening_ceremony()

    def opening_ceremony(self):
        """Move robot arm to start demonstration pose"""
        # reboot gripper motors, and set operating modes for all motors
        self.robot.core.robot_reboot_motors('single', 'gripper', True)
        self.robot.core.robot_set_operating_modes('group', 'arm', 'position')
        self.robot.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
        self.robot.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)
        torque_on(self.robot)
        
        # move arm to starting position
        start_arm_qpos = START_ARM_POSE[:6]
        move_arms(
            [self.robot],
            [start_arm_qpos],
            moving_time=2.0,
        )
        
        # move gripper to starting position
        move_grippers(
            [self.robot],
            [FOLLOWER_GRIPPER_JOINT_MID],
            moving_time=0.5
        )

    def phantom_state_callback(self, msg):
        self.omni_state = [msg.pose.position.y,
                           -msg.pose.position.x,
                           msg.pose.position.z,
                           -msg.rpy.z,
                           -msg.rpy.y,
                           -msg.rpy.x,
                           msg.open_gripper.data,
                           msg.close_gripper.data]

    def control(self):
        if self.omni_state:
            print(self.omni_state)
            print(self.robot.core.joint_states.position)
        else:
            self.get_logger().info('Waiting for OmniState message...')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        teleop_robot = TeleopRobot()
        executor = SingleThreadedExecutor()
        executor.add_node(teleop_robot)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            teleop_robot.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
