#!/usr/bin/env python
import geometry_msgs.msg
import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

br = tf2_ros.TransformBroadcaster()
t = geometry_msgs.msg.TransformStamped()


def init_transform(initial_position):
    t.header.frame_id = "world"
    t.child_frame_id = "camera"
    t.header.stamp = rospy.Time.now()
    t.transform.translation.x = initial_position.pose.pose.position.x
    t.transform.translation.y = initial_position.pose.pose.position.y
    t.transform.translation.z = 0


def callback_position(position):
    global t

    t.header.stamp = rospy.Time.now()
    t.transform.translation.x = position.pose.pose.position.x
    t.transform.translation.y = position.pose.pose.position.y
    t.transform.translation.z = 0


def callback_rotation(rotation):
    global br, t

    t.transform.rotation.x = rotation.orientation.x
    t.transform.rotation.y = rotation.orientation.y
    t.transform.rotation.z = rotation.orientation.z
    t.transform.rotation.w = rotation.orientation.w
    # rospy.loginfo(t)

    br.sendTransform(t)


def camera_pose_node():
    rospy.init_node("camera_pose")

    initial_position = rospy.wait_for_message("/ground_truth/state", Odometry)
    init_transform(initial_position)

    rospy.Subscriber("/ground_truth/state", Odometry, callback_position)
    rospy.Subscriber("/ardrone/imu", Imu, callback_rotation)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        camera_pose_node()
    except rospy.ROSInterruptException:
        pass
