#!/usr/bin/env python
import rospy
from sim_camera import Camera
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import time

import cv2

class CamVis(object):
    def __init__(self):
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        self.cam = Camera()

    def get_pose(self, data):
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        quat = [1., 1., 1., 1.]
        quat[0] = data.pose.pose.orientation.x
        quat[1] = data.pose.pose.orientation.y
        quat[2] = data.pose.pose.orientation.z
        quat[3] = data.pose.pose.orientation.w
        _, _, yaw = euler_from_quaternion(quat)
        return x, y, yaw

    def show_rgb(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("camera", bgr)
        cv2.waitKey(1)

    def odom_callback(self, data):
        x, y, theta = self.get_pose(data)
        img = self.cam.img_at([x, y, theta])
        self.show_rgb(img)

if __name__ == '__main__':
    cam_agent = CamVis()
    time.sleep(1)
    rospy.init_node('camera_agent')
    rospy.spin()