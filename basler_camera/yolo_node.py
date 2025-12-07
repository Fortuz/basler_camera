import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
import yaml
import os
import json

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # Load YOLOv11n finetuned model
        self.model = YOLO('src/basler_camera/resource/model_best.pt')

        self.model.to('cuda')  # Use GPU for inference
        # Initialize CvBridge for ROS2 image conversion
        self.bridge = CvBridge()
        
        # Performance optimization
        self.frame_skip_count = 0
        self.process_every_n_frames = 2  # Process every 2nd frame for better performance

        # Load camera calibration
        self.declare_parameter("calibration_file", "camera_calibration.yaml")
        self.declare_parameter("enable_undistortion", True)
        self.calibration_file = self.get_parameter("calibration_file").value
        self.enable_undistortion = self.get_parameter("enable_undistortion").value
        self.camera_matrix = None
        self.dist_coeffs = None
        self.opt_camera_matrix = None
        self.load_calibration()
        self.labels = ['HumanNormal', 'HumanOwner', 'Mecanumbot', 'MecanumbotHead', 'TennisBall']


        # Subscribe to image topic
        self.subscriber = self.create_subscription(
            Image,
            'camera/image_color',
            self.image_callback,
            10
        )

        # Publisher for object tracking data
        self.tracking_publisher = self.create_publisher(String, 'object_tracking', 10)

        self.get_logger().info("YOLO Node started and waiting for images.")

    def process_tracking_data(self, detected_objects, canvas):
        """Process detected objects and calculate position + orientation (Z as yaw)"""
        tracking_data = {}

        # Process each class
        for class_name, objects in detected_objects.items():
            for obj in objects:
                obj_info = {
                    'position': {
                        'x': float(obj['center'][0]),
                        'y': float(obj['center'][1]),
                        'z': 0.0  # Default orientation (yaw angle)
                    },
                    'confidence': obj['confidence'],
                    'visible': obj['visible']
                }

                # Add to tracking data
                if class_name not in tracking_data:
                    tracking_data[class_name] = []
                tracking_data[class_name].append(obj_info)

        # Calculate MecanumBot orientation using MecanumHead
        if 'Mecanumbot' in detected_objects and 'MecanumbotHead' in detected_objects:
            for bot_idx, bot in enumerate(detected_objects['Mecanumbot']):
                # Find closest head to this bot
                best_head = None
                min_distance = float('inf')

                for head in detected_objects['MecanumbotHead']:
                    dx = head['center'][0] - bot['center'][0]
                    dy = head['center'][1] - bot['center'][1]
                    distance = np.sqrt(dx**2 + dy**2)

                    if distance < min_distance:
                        min_distance = distance
                        best_head = head

                if best_head is not None:
                    # Calculate orientation (yaw angle in degrees)
                    dx = best_head['center'][0] - bot['center'][0]
                    dy = best_head['center'][1] - bot['center'][1]
                    yaw_rad = np.arctan2(dy, dx)
                    yaw_deg = float(np.degrees(yaw_rad))

                    # Update the bot's Z orientation
                    tracking_data['Mecanumbot'][bot_idx]['position']['z'] = yaw_deg

                    # Draw orientation arrow on canvas
                    bot_center = (int(bot['center'][0]), int(bot['center'][1]))
                    head_center = (int(best_head['center'][0]), int(best_head['center'][1]))
                    cv2.arrowedLine(canvas, bot_center, head_center, (255, 0, 255), 3, tipLength=0.3)

                    # Draw orientation angle text
                    cv2.putText(canvas, f"Yaw: {yaw_deg:.1f}°",
                               (bot_center[0] + 10, bot_center[1] + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    self.get_logger().info(f"MecanumBot orientation: {yaw_deg:.1f}° (Z-axis yaw)")

        return tracking_data

    def load_calibration(self):
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    calib_data = yaml.safe_load(f)
                
                if 'camera_matrix' in calib_data and isinstance(calib_data['camera_matrix'], dict):
                    camera_data = calib_data['camera_matrix']['data']
                    self.camera_matrix = np.array(camera_data, dtype=np.float32).reshape(3, 3)
                    
                    distortion_data = calib_data['distortion_coefficients']['data']
                    self.dist_coeffs = np.array(distortion_data, dtype=np.float32)
                    
                    w = calib_data['image_width']
                    h = calib_data['image_height']

                    new_k, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 0, (w, h))

                    self.get_logger().info(f"New camera matrix:\n{self.camera_matrix}")
                    self.get_logger().info(f"New distortion coeffs: {self.dist_coeffs}")
                    self.opt_camera_matrix = new_k

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
            # Skip frames for better performance
            self.frame_skip_count += 1
            if self.frame_skip_count % self.process_every_n_frames != 0:
                return
            
            # Convert ROS2 Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            # Apply camera calibration (undistort image) if enabled
            if self.enable_undistortion and self.camera_matrix is not None and self.dist_coeffs is not None:
                img_undist = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs, None, self.opt_camera_matrix)
                canvas = img_undist.copy()
            else:
                canvas = cv_image.copy()

            # Run YOLO detection
            results = self.model(canvas)

            # Track detected objects for orientation calculation
            detected_objects = {}

            # Process results
            for result in results:

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

                    # Store detected object data
                    class_name = self.labels[class_id]
                    obj_data = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'center': (center_x, center_y),
                        'bbox': (x1, y1, x2, y2),
                        'size': (width, height),
                        'visible': True
                    }

                    if class_name not in detected_objects:
                        detected_objects[class_name] = []
                    detected_objects[class_name].append(obj_data)

                    # Log object position and details
                    self.get_logger().info(f"Object {idx}: Class={class_name}, Conf={confidence:.3f}, "
                                            f"Center=({center_x:.1f},{center_y:.1f}), "
                                            f"BBox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), "
                                            f"Size=({width:.1f}x{height:.1f})")

                    # Draw bounding box
                    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Draw confidence and class
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(canvas, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate orientation for MecanumBot using MecanumHead
            tracking_data = self.process_tracking_data(detected_objects, canvas)

            # Publish tracking data
            if tracking_data:
                msg = String()
                msg.data = json.dumps(tracking_data, indent=2)
                self.tracking_publisher.publish(msg)

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