import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import cv2
import numpy as np
from sensor_msgs.msg import Image
from interbotix_common_modules.common_robot.robot import InterbotixRobotNode

class Cameras(InterbotixRobotNode):
    def __init__(self):
        super().__init__(node_name='cameras')
        
        callback_group = ReentrantCallbackGroup()
        
        self.create_subscription(
            Image, 
            '/cam_field/camera/color/image_raw', 
            self.field_image_callback, 
            10, 
            callback_group=callback_group
        )
        self.create_subscription(
            Image, 
            '/cam_overhead/camera/color/image_raw', 
            self.overhead_image_callback, 
            10, 
            callback_group=callback_group
        )
        self.create_subscription(
            Image, 
            '/cam_wrist/camera/color/image_raw', 
            self.wrist_image_callback, 
            10, 
            callback_group=callback_group
        )
        
        self.field_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.overhead_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.wrist_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        hz = 60
        dt = 1/hz
        self.create_timer(dt, self.display_images, callback_group=callback_group)
        
    def field_image_callback(self, msg):
        self.field_image = self.process_image(msg)
        
    def overhead_image_callback(self, msg):
        self.overhead_image = self.process_image(msg)
        
    def wrist_image_callback(self, msg):
        self.wrist_image = self.process_image(msg)
        
    def process_image(self, msg):
        image = np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def display_images(self):
        combined_image = np.hstack((self.field_image, self.overhead_image, self.wrist_image))
        cv2.imshow('Camera Images', combined_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    try:
        cameras = Cameras()
        executor = SingleThreadedExecutor()
        executor.add_node(cameras)
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            cameras.destroy_node()
            cv2.destroyAllWindows()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
