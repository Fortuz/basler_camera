import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
import yaml
import os

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # Load YOLOv11n finetuned model
        self.model = YOLO('src/basler_camera/resource/model_best.pt')

        self.model.to('cuda')  # Use GPU for inference
        # Initialize CvBridge for ROS2 image conversion
        self.bridge = CvBridge()

        # Load camera calibration
        self.declare_parameter("calibration_file", "calibration.yaml")
        self.calibration_file = self.get_parameter("calibration_file").value
        self.camera_matrix = None
        self.dist_coeffs = None
        self.load_calibration()

        # Subscribe to image topic
        self.subscriber = self.create_subscription(
            Image,
            'camera/image_color',
            self.image_callback,
            10
        )

        self.get_logger().info("YOLO Node started and waiting for images.")

    def load_calibration(self):
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    calib_data = yaml.safe_load(f)
                
                # Handle new ROS camera_info format from Zhang's calibration
                if 'camera_matrix' in calib_data and isinstance(calib_data['camera_matrix'], dict):
                    # New format with rows, cols, data structure
                    camera_data = calib_data['camera_matrix']['data']
                    self.camera_matrix = np.array(camera_data, dtype=np.float32).reshape(3, 3)
                    
                    distortion_data = calib_data['distortion_coefficients']['data']
                    self.dist_coeffs = np.array(distortion_data, dtype=np.float32)
                else:
                    # Old format - direct arrays
                    self.camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
                    self.dist_coeffs = np.array(calib_data['distortion_coefficients'], dtype=np.float32)
                
                self.get_logger().info(f"Loaded calibration from {self.calibration_file}")
                self.get_logger().info(f"Camera matrix:\n{self.camera_matrix}")
                self.get_logger().info(f"Distortion coeffs: {self.dist_coeffs}")
            else:
                self.get_logger().warn(f"Calibration file {self.calibration_file} not found. Running without undistortion.")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {e}")
            self.camera_matrix = None
            self.dist_coeffs = None

    def image_callback(self, msg):
        try:
            # Convert ROS2 Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            # Apply camera calibration (undistort image)
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                cv_image = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
            
            canvas = cv_image.copy()

            # Run YOLO detection
            results = self.model(cv_image)

            # Process results
            for result in results:
                # Check if we have keypoints (for pose detection)
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.xy
                    if keypoints is not None and len(keypoints) > 0:
                        for idx, robot in enumerate(keypoints):
                            robot = robot.cpu().numpy()

                            if len(robot) < 4:
                                continue

                            points = np.array([
                                robot[0],  # FL
                                robot[1],  # FR
                                robot[2],  # BR
                                robot[3],  # BL
                            ], dtype=np.float32)

                            points_int = points.astype(int)

                            # Log keypoint positions
                            self.get_logger().info(f"Robot {idx}: FL=({points[0][0]:.1f},{points[0][1]:.1f}), "
                                                 f"FR=({points[1][0]:.1f},{points[1][1]:.1f}), "
                                                 f"BR=({points[2][0]:.1f},{points[2][1]:.1f}), "
                                                 f"BL=({points[3][0]:.1f},{points[3][1]:.1f})")

                            # Draw bounding lines
                            for i in range(4):
                                cv2.line(canvas, tuple(points_int[i]),
                                         tuple(points_int[(i + 1) % 4]), (100, 255, 100), 2)

                            # Draw keypoints
                            point_names = ['FL', 'FR', 'BR', 'BL']
                            point_colors = [(0, 0, 255), (0, 255, 0), (0, 255, 0), (0, 0, 255)]
                            for i, pt in enumerate(points_int):
                                cv2.circle(canvas, tuple(pt), 6, point_colors[i], -1)
                                cv2.putText(canvas, point_names[i], tuple(pt + [6, -6]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, point_colors[i], 2)

                # Handle regular object detection (bounding boxes)
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for idx, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Calculate center and dimensions
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Log object position and details
                        self.get_logger().info(f"Object {idx}: Class={class_id}, Conf={confidence:.3f}, "
                                             f"Center=({center_x:.1f},{center_y:.1f}), "
                                             f"BBox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), "
                                             f"Size=({width:.1f}x{height:.1f})")
                        
                        # Draw bounding box
                        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Draw confidence and class
                        label = f"Class {class_id}: {confidence:.2f}"
                        cv2.putText(canvas, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the result
            cv2.imshow("YOLO Detection", canvas)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in image callback: {str(e)}")

    def cleanup(self):
        cv2.destroyAllWindows()
        self.get_logger().info("YOLO Node stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()