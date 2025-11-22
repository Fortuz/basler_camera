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
        self.captured_images = []
        self.sample_count = 0
        self.image_size = None
        self.chessboard_dim = (9, 10)
        self.cell_size = 0.025  # 25mm squares

        # GUI setup
        cv2.namedWindow("Zhang Calibration")

        self.get_logger().info("Zhang calibration node started. Press 'n' to capture chessboard images.")


    def get_corner_chessboard(self, image, chessboard_dim):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_dim, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners.reshape(-1, 2)
        return None

    def image_callback(self, msg: Image):
        # Convert ROS Image to NumPy array
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if self.image_size is None:
            self.image_size = (msg.width, msg.height)

        # Check for chessboard
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_dim, None)
        
        # Display image
        display_img = img.copy()
        if ret:
            cv2.drawChessboardCorners(display_img, self.chessboard_dim, corners, ret)
            cv2.putText(display_img, f"Chessboard detected! Press 'n' to capture ({self.sample_count}/{self.samples_required})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_img, f"No chessboard detected ({self.sample_count}/{self.samples_required})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Zhang Calibration", display_img)
        key = cv2.waitKey(1) & 0xFF

        # Capture image
        if key == ord('n') and ret:
            self.captured_images.append(img.copy())
            self.sample_count += 1
            self.get_logger().info(f"Captured image {self.sample_count}/{self.samples_required}")
            
            if self.sample_count >= self.samples_required:
                self.run_calibration()
                rclpy.shutdown()
        elif key == ord('n') and not ret:
            self.get_logger().warn("No chessboard detected!")

        # Quit
        elif key == ord('q'):
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def fill_X(self, chessboard_dim, cell_size):
        # Initialize the array for storing the world coordinates
        X = np.zeros((chessboard_dim[0] * chessboard_dim[1], 3), np.float32)

        # Fill X with the world coordinates in the correct order
        for i in range(chessboard_dim[0]):  # Number of columns
            for j in range(chessboard_dim[1]):  # Number of rows
                index = j * chessboard_dim[0] + i
                X[index] = [i * cell_size, j * cell_size, 0]
        return X

    def reproj_error(self, X, U, H):
        X_homogeneous = np.hstack((X[:,:2], np.ones((X.shape[0], 1))))
        U_projected_homogeneous = np.dot(H, X_homogeneous.T).T
        U_projected = U_projected_homogeneous[:, :2] / U_projected_homogeneous[:, 2, np.newaxis]

        error = np.sqrt(np.mean((U - U_projected) ** 2))
        self.get_logger().info(f'Reprojection error: {error:.4f}')
        return error

    def zhangs_method_calibration(self):
        X = self.fill_X(self.chessboard_dim, self.cell_size)
        H_list = []
        
        for i, image in enumerate(self.captured_images):
            U = self.get_corner_chessboard(image, self.chessboard_dim)
            if U is None:
                self.get_logger().warn(f"Could not find corners in image {i+1}")
                continue
            
            A = []
            for j in range(len(X)):
                x, y, _ = X[j]
                u, v = U[j]
                
                z = 1
                w = 1
                A.append([0, 0, 0, -w*x, -w*y, -w*z, v*x, v*y, v*z])
                A.append([w*x, w*y, w*z, 0, 0, 0, -u*x, -u*y, -u*z])
            
            A = np.array(A)
            
            eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T, A))
            h = eigenvectors[:, np.argmin(eigenvalues)]
            
            H = h.reshape(3, 3)
            H = H / H[2, 2]

            error = self.reproj_error(X, U, H)
            H_list.append(H)
            
        return H_list

    def compute_intrinsics(self, H_list):

        V = []
        for H in H_list:
            V.append(np.array([
                H[0, 0]*H[0, 1], H[0, 0]*H[1, 1] + H[1, 0]*H[0, 1],
                H[1, 0]*H[1, 1], H[2, 0]*H[0, 1] + H[0, 0]*H[2, 1],
                H[2, 0]*H[1, 1] + H[1, 0]*H[2, 1], H[2, 0]*H[2, 1]
            ]))
            V.append(np.array([
                H[0, 0]*H[0, 0] - H[0, 1]*H[0, 1], 2*(H[0, 0]*H[1, 0] - H[0, 1]*H[1, 1]),
                H[1, 0]*H[1, 0] - H[1, 1]*H[1, 1], 2*(H[2, 0]*H[0, 0] - H[2, 1]*H[0, 1]),
                2*(H[2, 0]*H[1, 0] - H[2, 1]*H[1, 1]), H[2, 0]*H[2, 0] - H[2, 1]*H[2, 1]
            ]))

        V = np.array(V)
        _, _, vh = np.linalg.svd(V)
        b = vh[-1] / vh[-1, -1]

        v0 = (b[1]*b[3] - b[0]*b[4]) / (b[0]*b[2] - b[1]**2)
        lambda_ = b[5] - (b[3]**2 + v0*(b[1]*b[3] - b[0]*b[4])) / b[0]
        alpha = np.sqrt(lambda_ / b[0])
        beta = np.sqrt(lambda_ * b[0] / (b[0]*b[2] - b[1]**2))
        gamma = -b[1] * alpha**2 * beta / lambda_
        u0 = gamma * v0 / beta - b[3] * alpha**2 / lambda_

        K = np.array([
            [alpha, gamma, u0],
            [0,     beta,  v0],
            [0,     0,     1]
        ])

        return K

    def get_r_t_vec(self, K, H_list):
        r1r2t = np.linalg.inv(K)@H_list[0]
        r1 = r1r2t[:,0]
        r2 = r1r2t[:,1]
        r3 = np.cross(r1,r2)
        t = r1r2t[:,2]
        return (r1,r2,r3,t)

    def run_calibration(self):
        self.get_logger().info("Running Zhang's calibration method...")
        
        H_list = self.zhangs_method_calibration()
        if len(H_list) == 0:
            self.get_logger().error("No valid homographies found!")
            return
            
        K = self.compute_intrinsics(H_list)
        r1,r2,r3,t = self.get_r_t_vec(K, H_list)
        
        self.get_logger().info("ZHANG'S CALIBRATION RESULTS:")
        self.get_logger().info(f"Camera matrix:\n{K}")
        
        # Create calibration.yaml file
        calibration_data = {
            'image_width': int(self.image_size[0]),
            'image_height': int(self.image_size[1]),
            'camera_name': 'basler_camera',
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': K.flatten().tolist()
            },
            'distortion_model': 'plumb_bob',
            'distortion_coefficients': {
                'rows': 1,
                'cols': 5,
                'data': [0.0, 0.0, 0.0, 0.0, 0.0]  # Zhang's method doesn't compute distortion
            },
            'rectification_matrix': {
                'rows': 3,
                'cols': 3,
                'data': np.eye(3).flatten().tolist()
            },
            'projection_matrix': {
                'rows': 3,
                'cols': 4,
                'data': np.hstack([K, np.zeros((3,1))]).flatten().tolist()
            }
        }

        with open("calibration.yaml", "w") as f:
            yaml.dump(calibration_data, f, default_flow_style=False)

        self.get_logger().info("Saved calibration to calibration.yaml")
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ManualCalibrationNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
