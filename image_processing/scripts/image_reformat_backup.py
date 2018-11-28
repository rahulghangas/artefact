#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
import tf_conversions
import cv2 as cv
import numpy as np
from numpy import sin, cos
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
import image_geometry as ig

coord = [(7, 7), (5, 3), (-3, -5), (3, -2)]

current_x = float()
current_y = float()

drone_camera = ig.PinholeCameraModel()


def callback(camera_feed, pose):
    quaternion = (
    pose.pose.orientation.x,
    pose.pose.orientation.y,
    pose.pose.orientation.z,
    pose.pose.orientation.w)

    euler = tf_conversions.transformations.euler_from_quaternion(quaternion)
    camera_matrix = np.zeros(shape=(4,4))
    camera_matrix = calculate_matrix(pose.pose.position, euler[2], 0, euler[0])

    camera_coords = list()

    for tuple in coord:
        transformed_matrix = camera_matrix.dot(np.array([tuple[0], tuple[1], 0, 1]))
        camera_coords.append((transformed_matrix[0], transformed_matrix[1]))


    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(camera_feed)
    except CvBridgeError as e:
        print(e)

    width, height = cv_image.shape[:2]

    for tuple in camera_coords:
        pixel_coord = drone_camera.project3dToPixel((tuple[0], 0, tuple[1]))

        if pixel_coord[0] < width -1 and pixel_coord[1] < height -1:
            cv_image = cv.circle(cv_image, (int(pixel_coord[0]), int(pixel_coord[1])), 3, (0, 0, 255), 3)
        
        print (pixel_coord[0], pixel_coord[1])

    #rospy.loginfo(cv_image)
    cv.imshow('Drone_camera', cv_image)
    cv.waitKey(1)


def calculate_matrix(position, yaw, pitch, roll):
    array = np.array([(cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll), 0),
                      (sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll) - cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll) + cos(yaw)*sin(roll), 0),
                      (-sin(pitch),         cos(pitch)*sin(roll),                               cos(pitch)*cos(roll),                               0),
                      (position.x,          position.y,                                         position.z,                                         1)])
    #return array
    return np.linalg.inv(array.transpose())


def listener():
    rospy.init_node('image_reformat', anonymous=True)

    camera_init_msg = rospy.wait_for_message("ardrone/front/camera_info", CameraInfo)
    drone_camera.fromCameraInfo(camera_init_msg)

    camera_feed = message_filters.Subscriber("ardrone/front/image_raw", Image)
    pose = message_filters.Subscriber("camera_pose", PoseStamped)

    ts = message_filters.TimeSynchronizer([camera_feed, pose], 10)
    ts.registerCallback(callback)
    #rospy.Subscriber("/ardrone/front/image_raw", Image, callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv.destroyAllWindows()


if __name__ == '__main__':
    listener()
