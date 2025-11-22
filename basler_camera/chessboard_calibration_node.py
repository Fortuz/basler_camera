#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import yaml

class ChessboardCalibrationNode(Node):
    def __init__(self):
        super().__init__('chessboard_camera_calibration')

        # Parameters
        self.declare_parameter("topic", "/camera/image_color")
        self.declare_parameter("save_path", "camera_calibration.yaml")
        self.declare_parameter("samples_required", 20)  # More samples for better calibration
        self.declare_parameter("chessboard_rows", 9)    # Internal corners
        self.declare_parameter("chessboard_cols", 8)    # Internal corners
        self.declare_parameter("square_size", 50.0)     # Size in mm

        self.topic = self.get_parameter("topic").value
        self.save_path = self.get_parameter("save_path").value
        self.samples_required = self.get_parameter("samples_required").value
        self.chessboard_size = (
            self.get_parameter("chessboard_cols").value,
            self.get_parameter("chessboard_rows").value
        )
        self.square_size = self.get_parameter("square_size").value

        # ROS subscription
        self.subscription = self.create_subscription(Image, self.topic, self.image_callback, 10)

        # Calibration data
        self.objpoints_list = []  # 3D points in real world space
        self.imgpoints_list = []  # 2D points in image plane
        self.sample_count = 0
        self.image_size = None

        # Prepare object points (0,0,0), (25,0,0), (50,0,0) ....,(200,125,0)
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1,2)
        self.objp *= self.square_size

        # Criteria for corner sub-pixel accuracy
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.get_logger().info("Chessboard calibration node started.")
        self.get_logger().info(f"Using {self.chessboard_size[0]}x{self.chessboard_size[1]} chessboard with {self.square_size}mm squares")
        self.get_logger().info("Hold chessboard in view and press 's' to capture sample")

    def image_callback(self, msg: Image):
        # Convert ROS Image to NumPy array
        if msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif msg.encoding == 'bgr8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif msg.encoding == 'mono8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            gray = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            self.get_logger().error(f"Unsupported image encoding: {msg.encoding}")
            return
            
        if self.image_size is None:
            self.image_size = (msg.width, msg.height)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        display_img = img.copy()
        
        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
            
            # Draw corners
            cv2.drawChessboardCorners(display_img, self.chessboard_size, corners2, ret)
            
            # Show status
            cv2.putText(display_img, f"Chessboard found! Press 's' to save sample {self.sample_count+1}/{self.samples_required}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_img, "Chessboard not found - adjust position/lighting", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display_img, f"Samples: {self.sample_count}/{self.samples_required}", 
                   (10, display_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Chessboard Calibration", display_img)
        key = cv2.waitKey(1) & 0xFF

        # Save sample
        if key == ord('s') and ret:
            self.objpoints_list.append(self.objp)
            self.imgpoints_list.append(corners2)
            self.sample_count += 1
            self.get_logger().info(f"Saved sample {self.sample_count}/{self.samples_required}")

            if self.sample_count >= self.samples_required:
                self.calibrate()
                rclpy.shutdown()
            else:
                self.get_logger().info(f"Move chessboard to different position/angle and press 's' again")

        # Quit
        elif key == ord('q'):
            self.get_logger().info("Exiting calibration.")
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def calibrate(self):
        self.get_logger().info("Calibrating camera from chessboard samples...")
        
        # Camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints_list, self.imgpoints_list, self.image_size, None, None,
            flags=cv2.CALIB_FIX_PRINCIPAL_POINT
        )

        if not ret:
            self.get_logger().error("Calibration failed.")
            return

        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.objpoints_list)):
            projected_points, _ = cv2.projectPoints(
                self.objpoints_list[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(self.imgpoints_list[i], projected_points, cv2.NORM_L2) / len(projected_points)
            total_error += error
        
        mean_error = total_error / len(self.objpoints_list)
        self.get_logger().info(f"Mean reprojection error: {mean_error:.3f} pixels")
        
        if mean_error > 1.0:
            self.get_logger().warn("High reprojection error - consider recalibrating with more samples.")
        else:
            self.get_logger().info("Good calibration achieved!")
        
        # Extract parameters
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        self.get_logger().info(f"Focal lengths: fx={fx:.2f}, fy={fy:.2f}")
        self.get_logger().info(f"Principal point: cx={cx:.2f}, cy={cy:.2f}")
        self.get_logger().info(f"Distortion coeffs: {dist_coeffs.flatten()}")

        # Save calibration
        data = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist(),
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
            "reprojection_error": float(mean_error),
            "chessboard_size": list(self.chessboard_size),
            "square_size_mm": self.square_size,
            "samples_used": self.sample_count
        }

        with open(self.save_path, "w") as f:
            yaml.dump(data, f)

        self.get_logger().info(f"Saved calibration to: {self.save_path}")
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = ChessboardCalibrationNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()