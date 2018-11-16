#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

def callback(camera_feed):
    bridge = CvBridge()
    try:
      cv_image = bridge.imgmsg_to_cv2(camera_feed)
    except CvBridgeError as e:
      print(e)
    
    #rospy.loginfo(cv_image)

    cv.imshow('Drone_camera', cv_image)

    cv.waitKey(1)
    
def listener():

    rospy.init_node('image_reformat', anonymous=True)

    rospy.Subscriber("ardrone/front/image_raw", Image, callback)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv.destroyAllWindows()

if __name__ == '__main__':
    listener()

