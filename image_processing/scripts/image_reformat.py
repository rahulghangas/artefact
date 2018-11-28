#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
import tf_conversions
import tf2_ros
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, TransformStamped
import image_geometry as ig

colors = [(255,0,0), (0,255,0), (0,0,255), (0,0,0)]

class Server:
    def __init__(self):
        self.drone_camera = ig.PinholeCameraModel()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.bridge = CvBridge()
    
    def callback_camera(self, camera_feed):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(camera_feed)
        except CvBridgeError as e:
            print(e)

        width, height = cv_image.shape[:2]

        for i in range(4):
            try:
                trans = self.tfBuffer.lookup_transform('camera', 'object%s'%str(i), rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                return

            pixel_coord = self.drone_camera.project3dToPixel((-trans.transform.translation.y, 0, trans.transform.translation.x))

            if trans.transform.translation.x >= 0:
                cv_image = cv.circle(cv_image, (int(pixel_coord[0]), int(pixel_coord[1])), 3, colors[i], 3)
                #print("Printed circle at " + str(trans.transform.translation.x) + ", " + str(trans.transform.translation.y), pixel_coord[0], pixel_coord[1])
        
            #print (pixel_coord[0], pixel_coord[1])

    #rospy.loginfo(cv_image)
        cv.imshow('Drone_camera', cv_image)
        cv.waitKey(1)


def listener():
    rospy.init_node('image_reformat', anonymous=True)
    server = Server()

    camera_init_msg = rospy.wait_for_message("ardrone/front/camera_info", CameraInfo)
    server.drone_camera.fromCameraInfo(camera_init_msg)

    rospy.Subscriber("/ardrone/front/image_raw", Image, server.callback_camera)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv.destroyAllWindows()


if __name__ == '__main__':
    listener()
