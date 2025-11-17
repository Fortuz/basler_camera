#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from pypylon import pylon
import cv2
from cv_bridge import CvBridge

class BaslerCameraNode(Node):
    def __init__(self):
        super().__init__('basler_camera')
        self.publisher_ = self.create_publisher(Image, 'camera/image_color', 10)
        self.bridge = CvBridge()

        self.get_logger().info("Starting Basler camera...")

        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        self.camera.Open()

        # Pixel format can be one of the following RGB8, BGR8, BayerRG8
        self.camera.PixelFormat.SetValue('RGB8')

        self.camera.StartGrabbing()

        self.timer = self.create_timer(0.01, self.capture_frame)

    def capture_frame(self):
        if self.camera.IsGrabbing():
            grab = self.camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
            if grab.GrabSucceeded():
                img = grab.Array  # numpy array
                msg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
                self.publisher_.publish(msg)
        else:
            self.get_logger().warn("Camera not grabbing!")

def main(args=None):
    rclpy.init(args=args)
    node = BaslerCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
