#!/usr/bin/env python
import rospy
import message_filters
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

pub = rospy.Publisher('camera_pose', PoseStamped, queue_size=10)


def callback(position, rotation):
    t = PoseStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "camera"
    t.pose.position.x = position.pose.pose.position.x
    t.pose.position.y = position.pose.pose.position.y
    t.pose.orientation.x = rotation.orientation.x
    t.pose.orientation.y = rotation.orientation.y
    t.pose.orientation.z = rotation.orientation.z
    t.pose.orientation.w = rotation.orientation.w
    # rospy.loginfo(t)

    pub.publish(t)


def camera_pose_node():
    rospy.init_node('camera_pose')

    position = message_filters.Subscriber('/ground_truth/state', Odometry)
    rotation = message_filters.Subscriber('/ardrone/imu', Imu)

    ts = message_filters.ApproximateTimeSynchronizer([position, rotation], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    try:
        camera_pose_node()
    except rospy.ROSInterruptException:
        pass
