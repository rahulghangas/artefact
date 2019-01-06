#!/usr/bin/env python
from threading import Thread, Lock
import rospy
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
import image_geometry as ig

from glumpy import app, gl, glm, gloo
import numpy as np

# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]

# Default height and width of window.
img_height = 512
img_width = 512

# Default angles of rotation for cube objects
phi, theta = 40, 30

bag = dict()

# Global Lock used for Threads. Two vital threads work side by side in this program. The main thread runs the ROS node
# while a secondary thread renders the camera feed overlaid by 3d objects using OpenGL. The reason why the ros node is
# not also run from a secondary thread is because the messages sent by the node are raised as standard
# python signals which are asynchronous and cannot be raised from a secondary thread.
lock = Lock()


# The server class handles all messages received by the ROS node.
class Server:

    # The __init__ method initialises some important objects required by the class to function.
    def __init__(self):

        # A generalised Pinhole camera model provided by the image_geometry module in ROS. Used as an approximate model
        # of the drone camera. Should be modified with correct camera parameters before usage.
        self.drone_camera = ig.PinholeCameraModel()

        # A buffer which is then used for the TransformListener provided by the ROS tf2 library.  Used to queue up
        # static 3-D coordinates of objects and dynamic 3-D coordinates of the drone. Can be used to transform object
        # coordinates to the drone's frame of reference
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # A bridging library used to transform ROS images to OpenCV images
        self.bridge = CvBridge()

    # This method is only called once to replace default img_height and img_width with actual values. Also, the initial
    # image is used as the default background texture for OpeGL so that size of numpy array for texture can be defined
    # and the array doesn't need to be reshaped in future.
    def init_camera_parameters(self, first_camera_feed):
        try:
            global initial_image
            initial_image = self.bridge.imgmsg_to_cv2(first_camera_feed)
            global img_width, img_height
            img_height, img_width = initial_image.shape[:2]

        except CvBridgeError as e:
            print(e)

    def callback_camera(self, camera_feed):

        global lock, quad

        lock.acquire()
        try:
            quad['texture'] = self.bridge.imgmsg_to_cv2(camera_feed)[..., ::-1]
        except:
            return
        finally:
            lock.release()

        for i in range(4):
            try:
                trans = self.tfBuffer.lookup_transform('camera', 'object%s' % str(i), rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                return

            # pixel_coord = self.drone_camera.project3dToPixel((-trans.transform.translation.y,
            #                                                   -trans.transform.translation.z,
            #                                                   trans.transform.translation.x))

            # if trans.transform.translation.x >= 0 and pixel_coord:
            lock.acquire()

            try:
                bag[i]['view'] = glm.translation(-trans.transform.translation.y, 
                                                    -trans.transform.translation.z, 
                                                    -trans.transform.translation.x)
            finally:
                lock.release()
            # else:
            #     lock.acquire()
            #     try:
            #         bag[i]['view'] = glm.translation(0, 0, 10.0)
            #     finally:
            #         lock.release()
                # to_draw_new.add(bag[i])
                # cv_image = cv.circle(cv_image, (int(pixel_coord[0]), int(pixel_coord[1])), 3, colors[i], 3)
                # print("Printed circle at " + str(trans.transform.translation.x) + ", " + str(
                # trans.transform.translation.y), pixel_coord[0], pixel_coord[1])
            # print (pixel_coord[0], pixel_coord[1])

        # rospy.loginfo(cv_image)
        # cv.imshow('Drone_camera', cv_image)
        # cv.waitKey(1)


def listener():
    rospy.init_node('image_reformat', anonymous=True)
    server = Server()

    camera_info = rospy.wait_for_message("ardrone/front/camera_info", CameraInfo)
    server.drone_camera.fromCameraInfo(camera_info)

    global projection_matrix
    projection_matrix = camera_info.P

    image_parameter_init_msg = rospy.wait_for_message("ardrone/front/image_raw", Image)
    server.init_camera_parameters(image_parameter_init_msg)

    Thread(target=opengl).start()

    rospy.Subscriber("/ardrone/front/image_raw", Image, server.callback_camera)
    rospy.spin()


def opengl():
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]        

    vertex = """
    uniform mat4   model;         // Model matrix
    uniform mat4   view;          // View matrix
    uniform mat4   projection;    // Projection matrix
    attribute vec3 position;      // Vertex position
    attribute vec3 u_color;
    varying vec3 v_color;
    void main()
    {
        v_color = u_color;
        gl_Position = projection * view * model * vec4(position,1.0);
    }
    """

    fragment = """
    varying vec3 v_color;
    void main()
    {
        gl_FragColor = vec4(v_color, 1.0);
    }
    """
    vertex2 = """
        attribute vec3 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            gl_Position = vec4(position,1.0);
            v_texcoord = texcoord;
        }
    """

    fragment2 = """
        uniform sampler2D texture;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(texture, v_texcoord);
        }
    """

    x = 0.1
    V = np.zeros(8, [("position", np.float32, 3)])
    V["position"] = [[x*1, x*1, x*1], [x*-1, x*1, x*1], [x*-1, x*-1, x*1], [x*1, x*-1, x*1],
                     [x*1, x*-1, x*-1], [x*1, x*1, x*-1], [x*-1, x*1, x*-1], [x*-1, x*-1, x*-1]]
    V = V.view(gloo.VertexBuffer)
    
    I = np.array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 1,
                  1, 6, 7, 1, 7, 2, 7, 4, 3, 7, 3, 2, 4, 7, 6, 4, 6, 5], dtype=np.uint32)
    I = I.view(gloo.IndexBuffer)

    O = np.array([0,1, 1,2, 2,3, 3,0,
     4,7, 7,6, 6,5, 5,4,
     0,5, 1,6, 2,7, 3,4 ], dtype=np.uint32)
    O = O.view(gloo.IndexBuffer)

    for iterator in range(4):
        cube = gloo.Program(vertex, fragment)
        cube.bind(V)
        cube['model'] = np.eye(4, dtype=np.float32)
        cube['view'] = glm.translation(0, 0, -30)
        bag[iterator] = cube

    global quad
    quad = gloo.Program(vertex2, fragment2, count=4)
    quad['position'] = [(-1, -1, 0), (-1, +1, 0), (+1, -1, 0), (+1, +1, 0)]
    quad['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
    quad['texture'] = initial_image[..., ::-1]

    window = app.Window(width=img_width, height=img_height, color=(1, 1, 1, 1))

    @window.event
    def on_draw(dt):
        global phi, theta

        window.clear()
        gl.glDisable(gl.GL_DEPTH_TEST)

        # try:
        #    quad['texture'] = cam_img_texture
        # except:
        #    quad['texture'] = cv.imread("/home/rahul/Pictures/GitKraken_001.png")[..., ::-1]
        quad.draw(gl.GL_TRIANGLE_STRIP)

        gl.glEnable(gl.GL_DEPTH_TEST)
        # Filled cube

        model = np.eye(4, dtype=np.float32)
        glm.rotate(model, theta, 0, 0, 1)
        glm.rotate(model, phi, 0, 1, 0)

        for obj in bag.values():
            obj['u_color'] = 1, 0, 0
            obj.draw(gl.GL_TRIANGLES, I)

            gl.glDepthMask(gl.GL_FALSE)
            obj['u_color'] = 0, 0, 0
            obj.draw(gl.GL_LINES, O)
            gl.glDepthMask(gl.GL_TRUE)

            obj['model'] = model

        # cube.draw(gl.GL_TRIANGLES, I)

        # Make cube rotate
        theta += 1.0  # degrees
        phi += 1.0  # degrees



    @window.event
    def on_resize(width, height):
        view_matrix = np.zeros((4,4))
        view_matrix[0][0] = 2.0 * projection_matrix[0] / img_width
        view_matrix[1][1] = -2.0 * projection_matrix[5] / img_height

        view_matrix[2][0] = 1.0 - 2.0 * projection_matrix[2] / img_width
        view_matrix[2][1] = 2.0 * projection_matrix[6] / img_height - 1.0
        view_matrix[2][2] = (30 + 1) / float(1 - 30)
        view_matrix[2][3] = -1.0

        view_matrix[3][2] = 2.0 * 30 * 1 / (1 - 30)

        for obj in bag.values():
            obj['projection'] = view_matrix
        # print(glm.perspective(45.0, width / float(height), 2.0, 100.0))

    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)

    app.run(framerate=30)


if __name__ == '__main__':
    listener()
