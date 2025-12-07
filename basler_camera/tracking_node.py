import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')

        # Dictionary to store last known positions for each object class
        self.last_positions = {}

        # Subscribe to object tracking topic
        self.subscription = self.create_subscription(
            String,
            'object_tracking',
            self.tracking_callback,
            10
        )

        self.get_logger().info("Tracking Node started and listening to object_tracking topic.")

    def tracking_callback(self, msg):
        """Callback function to handle incoming tracking data"""
        try:
            # Parse JSON data
            tracking_data = json.loads(msg.data)

            # Update last known positions for visible objects
            for class_name, objects in tracking_data.items():
                for idx, obj in enumerate(objects):
                    if obj['visible']:
                        # Store last known position for this object
                        if class_name not in self.last_positions:
                            self.last_positions[class_name] = {}
                        self.last_positions[class_name][idx] = obj['position'].copy()

            # Print header
            self.get_logger().info("=" * 80)
            self.get_logger().info("OBJECT TRACKING DATA:")
            self.get_logger().info("=" * 80)

            # Print data for each object class
            for class_name, objects in tracking_data.items():
                self.get_logger().info(f"\n{class_name}:")
                self.get_logger().info("-" * 40)

                for idx, obj in enumerate(objects):
                    position = obj['position']
                    confidence = obj['confidence']
                    visible = obj['visible']

                    # If not visible, use last known position
                    if not visible and class_name in self.last_positions and idx in self.last_positions[class_name]:
                        position = self.last_positions[class_name][idx]
                        status = "Not Visible (Last Known Position)"
                    elif visible:
                        status = "Visible"
                    else:
                        status = "Not Visible (No Previous Position)"

                    self.get_logger().info(f"  Object {idx + 1}:")
                    self.get_logger().info(f"    Status: {status}")
                    self.get_logger().info(f"    Position: x={position['x']:.2f}, y={position['y']:.2f}, z={position['z']:.2f}")
                    self.get_logger().info(f"    Confidence: {confidence:.3f}")

            self.get_logger().info("=" * 80)

        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse tracking data: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in tracking callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
