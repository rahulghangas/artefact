#!/usr/bin/env python
import rospy

#Because of transformations
import tf
import tf2_msgs.msg
import tf2_ros
import geometry_msgs.msg

coord = [(-2, 7), (5, 3), (-3, -5), (3,-2)]

def static_broadcast():
    rospy.init_node('fixed_tf2_broadcaster')
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()
        
    global coord    
    list_tf = list()
    for i in range(len(coord)):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = "object%s" %str(i)
        t.transform.translation.x = float(coord[i][0])
        t.transform.translation.y = float(coord[i][1])
        t.transform.translation.z = 0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        list_tf.append(t)

    broadcaster.sendTransform(list_tf)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    try:
        static_broadcast()
    except rospy.ROSInterruptException:
        pass
