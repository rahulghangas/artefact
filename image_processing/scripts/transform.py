#!/usr/bin/env python
import rospy
import message_filters
import tf2_ros
import geometry_msgs.msg
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def callback(position, rotation):
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"
    t.child_frame_id = "camera"
    t.transform.translation.x = position.pose.pose.position.x
    t.transform.translation.y = position.pose.pose.position.y
    t.transform.translation.z = 0
    t.transform.rotation.x = rotation.orientation.x
    t.transform.rotation.y = rotation.orientation.y
    t.transform.rotation.z = rotation.orientation.z
    t.transform.rotation.w = rotation.orientation.w
    # rospy.loginfo(t)

    br.sendTransform(t)


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
