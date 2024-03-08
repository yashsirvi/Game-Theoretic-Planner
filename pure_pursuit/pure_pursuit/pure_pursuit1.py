#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Header

class PurePursuit1(Node):
    def __init__(self):
        super().__init__('pure_pursuit1')
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.timer = self.create_timer(1.0, self.publish_ackermann_drive)

    def publish_ackermann_drive(self):
        msg = AckermannDriveStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = 0.5
        msg.drive.steering_angle_velocity = 1.0
        msg.drive.speed = 0.1
        msg.drive.acceleration = 1.0

        self.publisher.publish(msg)
        self.get_logger().info('Publishing AckermannDriveStamped message')

def main(args=None):
    rclpy.init(args=args)
    ackermann_publisher = PurePursuit1()
    rclpy.spin(ackermann_publisher)
    ackermann_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
