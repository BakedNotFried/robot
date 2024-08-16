#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from omni_msgs.msg import OmniState

class TeleopRobot(Node):
    def __init__(self):
        super().__init__('TeleopRobot')
        # Fields
        self.omni_state = None  # x, y, z, r, p, y, gripper, lock
        # Subs
        self.create_subscription(OmniState, '/omni/state', self.phantom_state_callback, 1)
        # Timer
        self.create_timer(0.1, self.control)  # Call control every 100ms

    def phantom_state_callback(self, msg):
        self.omni_state = [msg.pose.position.y, 
                             -msg.pose.position.x, 
                             msg.pose.position.z, 
                             -msg.rpy.z, 
                             -msg.rpy.y, 
                             -msg.rpy.x, 
                             int(msg.close_gripper),
                             int(msg.locked)]

    def control(self):
        if self.omni_state:
            print(self.omni_state)
        else:
            self.get_logger().info('Waiting for OmniState message...')

def main(args=None):
    rclpy.init(args=args)
    
    teleop_robot = TeleopRobot()
    
    try:
        rclpy.spin(teleop_robot)
    except KeyboardInterrupt:
        pass
    finally:
        teleop_robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
