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
        self.declare_parameter("pattern_width", 210.0)  # A4 paper width in mm
        self.declare_parameter("pattern_height", 297.0) # A4 paper height in mm

        self.topic = self.get_parameter("topic").value
        self.save_path = self.get_parameter("save_path").value
        self.samples_required = self.get_parameter("samples_required").value
        self.min_points = self.get_parameter("min_points").value
        self.pattern_width = self.get_parameter("pattern_width").value
        self.pattern_height = self.get_parameter("pattern_height").value

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

        self.get_logger().info("Manual calibration node started.")
        self.get_logger().info(f"Place a rectangular object ({self.pattern_width}x{self.pattern_height}mm) in view.")
        self.get_logger().info("Click 4 corners (top-left, top-right, bottom-right, bottom-left) then press 'n'.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_points.append((x, y))
            self.get_logger().info(f"Clicked point: {x},{y}")

    def image_callback(self, msg: Image):
        # Convert ROS Image to NumPy array
        if msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == 'bgr8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif msg.encoding == 'mono8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            self.get_logger().error(f"Unsupported image encoding: {msg.encoding}")
            return
            
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
            if len(self.click_points) != 4:
                self.get_logger().warn("Please click exactly 4 corner points (top-left, top-right, bottom-right, bottom-left).")
                return

            # Generate corresponding 3D object points for rectangle corners
            # Assuming corners clicked in order: top-left, top-right, bottom-right, bottom-left
            objpoints = np.array([
                [0.0, 0.0, 0.0],                                    # top-left
                [self.pattern_width, 0.0, 0.0],                     # top-right
                [self.pattern_width, self.pattern_height, 0.0],      # bottom-right
                [0.0, self.pattern_height, 0.0]                     # bottom-left
            ], dtype=np.float32)

            self.objpoints_list.append(objpoints)
            # Ensure image points are in correct format for OpenCV
            imgpoints = np.array(self.click_points, dtype=np.float32).reshape(-1, 1, 2)
            self.imgpoints_list.append(imgpoints)
            self.sample_count += 1
            self.get_logger().info(f"Saved sample {self.sample_count}/{self.samples_required}")

            # Clear points for next frame
            self.click_points = []

            # Auto-calibrate if enough samples
            if self.sample_count >= self.samples_required:
                self.calibrate()
                rclpy.shutdown()
            else:
                self.get_logger().info(f"Move the pattern to a different pose and click 4 corners again.")

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

        # Calculate reprojection error for validation
        total_error = 0
        for i in range(len(self.objpoints_list)):
            projected_points, _ = cv2.projectPoints(
                self.objpoints_list[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(self.imgpoints_list[i], projected_points, cv2.NORM_L2) / len(projected_points)
            total_error += error
        
        mean_error = total_error / len(self.objpoints_list)
        self.get_logger().info(f"Mean reprojection error: {mean_error:.3f} pixels")
        
        if mean_error > 2.0:
            self.get_logger().warn("High reprojection error - calibration may be inaccurate.")
        
        # Extract focal lengths and principal point
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        self.get_logger().info(f"Focal lengths: fx={fx:.2f}, fy={fy:.2f}")
        self.get_logger().info(f"Principal point: cx={cx:.2f}, cy={cy:.2f}")

        data = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist(),
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
            "reprojection_error": float(mean_error),
            "pattern_size_mm": [self.pattern_width, self.pattern_height]
        }

        with open(self.save_path, "w") as f:
            yaml.dump(data, f)

        self.get_logger().info(f"Saved calibration to: {self.save_path}")
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = ManualCalibrationNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()