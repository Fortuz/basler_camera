#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import yaml

class ManualCalibrationNode(Node):
    def __init__(self):
        super().__init__('manual_camera_calibration')

        # Parameters
        self.declare_parameter("topic", "/camera/image_color")
        self.declare_parameter("save_path", "manual_camera.yaml")
        self.declare_parameter("samples_required", 5)  # number of frames to calibrate
        self.declare_parameter("min_points", 4)        # minimum points per frame

        self.topic = self.get_parameter("topic").value
        self.save_path = self.get_parameter("save_path").value
        self.samples_required = self.get_parameter("samples_required").value
        self.min_points = self.get_parameter("min_points").value

        # ROS subscription
        self.subscription = self.create_subscription(Image, self.topic, self.image_callback, 10)

        # Calibration data
        self.imgpoints_list = []  # 2D points clicked
        self.objpoints_list = []  # 3D points (world)
        self.click_points = []    # current frame clicks
        self.sample_count = 0
        self.image_size = None

        # GUI setup
        cv2.namedWindow("Manual Calibration")
        cv2.setMouseCallback("Manual Calibration", self.mouse_callback)

        self.get_logger().info("Manual calibration node started. Click points in the GUI and press 'n' to save each sample.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_points.append((x, y))
            self.get_logger().info(f"Clicked point: {x},{y}")

    def image_callback(self, msg: Image):
        # Convert ROS Image to NumPy array
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if self.image_size is None:
            self.image_size = (msg.width, msg.height)

        # Draw persistent clicked points
        display_img = img.copy()
        for idx, (x, y) in enumerate(self.click_points):
            cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)  # green circle
            cv2.putText(display_img, str(idx+1), (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Manual Calibration", display_img)
        key = cv2.waitKey(1) & 0xFF

        # Save sample
        if key == ord('n'):
            if len(self.click_points) < self.min_points:
                self.get_logger().warn(f"Please click at least {self.min_points} points before saving.")
                return

            # Generate corresponding 3D object points (z=0)
            objpoints = []
            for pt in self.click_points:
                x_world, y_world = float(pt[0]), float(pt[1])  # replace with actual world coords if known
                objpoints.append([x_world, y_world, 0.0])
            objpoints = np.array(objpoints, dtype=np.float32)

            self.objpoints_list.append(objpoints)
            self.imgpoints_list.append(np.array(self.click_points, dtype=np.float32))
            self.sample_count += 1
            self.get_logger().info(f"Saved sample {self.sample_count}/{self.samples_required}")

            # Clear points for next frame
            self.click_points = []

            # Auto-calibrate if enough samples
            if self.sample_count >= self.samples_required:
                self.calibrate()
                rclpy.shutdown()

        # Quit
        elif key == ord('q'):
            self.get_logger().info("Exiting manual calibration.")
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def calibrate(self):
        self.get_logger().info("Calibrating camera from manual points...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints_list, self.imgpoints_list, self.image_size, None, None
        )

        if not ret:
            self.get_logger().error("Calibration failed.")
            return

        data = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist(),
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
        }

        with open(self.save_path, "w") as f:
            yaml.dump(data, f)

        self.get_logger().info(f"Saved manual calibration file to: {self.save_path}")
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = ManualCalibrationNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
