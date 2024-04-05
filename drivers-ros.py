#!/usr/bin/env python
import rospy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from sensor_msgs.msg import LaserScan

from drivers import DisparityExtenderStable # the policy/motion planner that you create

def scan_callback(msg):
    scan = msg.ranges
    print("------------------------SCAN------------------------")
    print(scan)
    print("------------------------SCAN------------------------")
    speed, steer =  planner.process_lidar(scan)

    res = AckermannDriveStamped()
    res.drive = AckermannDrive()
    res.drive.steering_angle = steer
    res.drive.speed = speed

    pub_controls.publish(res)


planner = DisparityExtenderStable()
control_topic = rospy.get_param("~control_topic", "/mux/ackermann_cmd_mux/input/navigation")
pub_controls = rospy.Publisher(control_topic, AckermannDriveStamped, queue_size=1)
sub_topic = "/scan"
sub_scan = rospy.Subscriber(sub_topic, LaserScan, callback=scan_callback)
rospy.init_node("path_publisher")
# rospy.sleep(1.0)
rospy.spin()