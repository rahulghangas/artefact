#!/usr/bin/env python
import math
from threading import Lock, Thread

import image_geometry as ig
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from scipy import interpolate
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import euler_from_quaternion

from glumpy import app, gl, glm, gloo

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
        # of the drone camera. Should be modified with correct camera parameters before usage. Can be used to do
        # 3d projections directly. Currently unused
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
    # and the array doesn't need to be reshaped again.
    def init_camera_parameters(self, first_camera_feed):
        try:
            global initial_image
            initial_image = self.bridge.imgmsg_to_cv2(first_camera_feed)
            global img_width, img_height
            img_height, img_width = initial_image.shape[:2]

        except CvBridgeError as e:
            print(e)

    # Callback function of the ros node that susbscribes to the camera topic
    def callback_camera(self, camera_feed):

        # global variables that are described as follows
        # path - glumpy object for augmented reality path in OpenGL
        # lock - global lock as described above
        # quad - glumpy object for background texture given by camera topic
        global path, lock, quad

        lock.acquire()
        try:
            # try to update background texture or return to ros loop if fails
            quad["texture"] = self.bridge.imgmsg_to_cv2(camera_feed)
        except:
            return
        finally:
            lock.release()

        # coords_3d is a numpy arraythat is stores 3d_ccordinates of all objects and is later extrapolated to a curve that
        # represents the path
        coords_3d = np.zeros((len(bag), 3))
        for i in range(len(bag)):
            try:
                # get the transform of teh objects w.r.t the drone camera and update coords_3d with the coordinates
                trans = self.tfBuffer.lookup_transform(
                    "camera_base_link", "object%s" % str(i), rospy.Time()
                )
                coords_3d[i] = [
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z,
                ]
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                return

            lock.acquire()

            try:
                # Define the view matrix of the objects w.r.t the drone camera
                bag[i]["view"] = glm.translation(
                    -trans.transform.translation.y, 0.0, -trans.transform.translation.x
                )
            except:
                return
            finally:
                lock.release()

        # Interpolate the coordinates in coords 3d to define the curve
        tck, u_ = interpolate.splprep(coords_3d.T, s=0.0)
        bez_x, bez_y, bez_z = np.array(
            interpolate.splev(np.linspace(0, 1, 25 * 4), tck)
        )

        # Specific index in position defines the current point on the curve
        # Specific index in position defines the previous point on the curve
        # Specific index in side represents left or right, namely -1 or +1
        position = list()
        last_position = list()
        side = list()

        # Initialise for first position with corresponding last position as itself because it has no predecessor
        position.append([-bez_y[1], 0.8, -bez_x[1]])
        position.append([-bez_y[1], 0.8, -bez_x[1]])
        last_position.append([-bez_y[1], 0.8, -bez_x[1]])
        last_position.append([-bez_y[1], 0.8, -bez_x[1]])
        side += [-1.0, 1.0]

        # For each point on the curve append itself and its predecssor twice, once for each side
        for i in range(1, len(bez_x)):
            position.append([-bez_y[i], 0.8, -bez_x[i]])
            position.append([-bez_y[i], 0.8, -bez_x[i]])

            last_position.append([-bez_y[i - 1], 0.8, -bez_x[i - 1]])
            last_position.append([-bez_y[i - 1], 0.8, -bez_x[i - 1]])

            side += [-1.0, 1.0]

        # update the position, last_position and side list to the path object in glumpy
        lock.acquire()
        try:
            path["position"] = position
            path["last_position"] = last_position
            path["side"] = side
        except:
            return
        finally:
            lock.release()


# listener function of the node
def listener():
    rospy.init_node("image_reformat", anonymous=True)

    # The initialisation and callback functions are encapsulated in a class called Server
    server = Server()

    # Get the camera_info from the topic only once. No need to subscribe to it
    camera_info = rospy.wait_for_message("bebop/camera_info", CameraInfo)
    # Update the pinhole camera model with the coorect camera parameters
    server.drone_camera.fromCameraInfo(camera_info)

    # Extract the projection matrix from the camera_info. Will be used to define the the projection matrix in OpenGL
    global projection_matrix
    projection_matrix = camera_info.P

    # Get the image once to define the deafult background texture and size of OpenGL buffers
    image_parameter_init_msg = rospy.wait_for_message("bebop/image_raw", Image)
    server.init_camera_parameters(image_parameter_init_msg)

    # Start the opengl thread that runs concurrently to the current (main) thread
    Thread(target=opengl).start()

    # Subscribe to the drone camera image topic
    rospy.Subscriber("bebop/image_raw", Image, server.callback_camera)
    rospy.spin()


def opengl():

    # The vertex shader for augmented reality objects
    vertex = """
    uniform mat4   model;         // Model matrix
    uniform mat4   view;          // View matrix
    uniform mat4   projection;    // Projection matrix
    attribute vec3 position;      // Vertex position
    attribute vec3 u_color;       // Color defined on whether surface (custom color) or edge (black)
    varying vec3 v_color;
    void main()
    {
        v_color = u_color;
        gl_Position = projection * view * model * vec4(position,1.0);
    }
    """

    # The fragment shader for augmented reality objects
    # Also used as fragment shader for interpolated path object
    fragment = """
    varying vec3 v_color;
    void main()
    {
        gl_FragColor = vec4(v_color, 1.0);
    }
    """

    # The vertex shader for the background texture (i.e -> camera image)
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

    # The fragment shader for the background texture (i.e -> camera image)
    fragment2 = """
        uniform sampler2D texture;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(texture, v_texcoord);
        }
    """

    # The fragment shader for the background texture (i.e -> camera image)
    vertex3 = """
    attribute vec3 last_position;
    attribute float side;
    uniform mat4   projection;    // Projection matrix defined in function on_resize()
    attribute vec3 position;      // Vertex position
    attribute vec3 u_color;       // Color defined on whether surface (custom color) or edge (black)
    varying vec3 v_color;
    void main()
    {   
        // line_vec is a vector from last position to current position 
        vec3 line_vec;
        line_vec = position - last_position;

        // Unit normal vector to line_vec and parallel to ground plane, used to define the width of the curve
        // by establishing two points on the left and right of the current position
        vec3 normal;
        normal = vec3(-line_vec.z, 0.0, line_vec.x);
        normal = normal/(sqrt(normal.x*normal.x + normal.z*normal.z))*0.8;

        // Add normal vector to position to define new position, side of new_position is unknown
        vec3 new_position;
        new_position = position + normal;

        // vector from last position to new position
        vec3 new_line_vec;
        new_line_vec = new_position - last_position;

        // multiplier is used to identify if new_psoition is on the correct side defined by float side. If yes,
        // multiplier is +1.0 else -1.0
        float multiplier;
        multiplier = 1.0;

        // Check if new_position is on correct side else change multiplier to -1.0 
        if (-line_vec.x*new_line_vec.y  + line_vec.y*new_line_vec.x < 0.0){
            if(side == 1.0){
                multiplier = -1.0;
            }

        }else{
            if(side == -1.0){
                multiplier = -1.0;
            }
        }

        // Redefine new_position using multiplier to ge the point on the correct side
        new_position = position + normal*multiplier;

        // Define the view_matrix as pos
        mat4 pos;
        pos[0] = vec4(1.0, 0.0, 0.0, 0.0);
        pos[1] = vec4(0.0, 1.0, 0.0, 0.0);
        pos[2] = vec4(0.0, 0.0, 1.0, 0.0);
        pos[3] = vec4(new_position.x, new_position.y, new_position.z, 1.0);

        v_color = u_color;
        gl_Position = projection * pos * vec4(0.0, 0.0, 0.0, 1.0);
    }
    """

    # Define the vertices and indices for each augmented reality object
    x = 0.1
    V = np.zeros(8, [("position", np.float32, 3)])
    V["position"] = [
        [x * 1, x * 1, x * 1],
        [x * -1, x * 1, x * 1],
        [x * -1, x * -1, x * 1],
        [x * 1, x * -1, x * 1],
        [x * 1, x * -1, x * -1],
        [x * 1, x * 1, x * -1],
        [x * -1, x * 1, x * -1],
        [x * -1, x * -1, x * -1],
    ]
    V = V.view(gloo.VertexBuffer)

    I = np.array(
        [
            0,
            1,
            2,
            0,
            2,
            3,
            0,
            3,
            4,
            0,
            4,
            5,
            0,
            5,
            6,
            0,
            6,
            1,
            1,
            6,
            7,
            1,
            7,
            2,
            7,
            4,
            3,
            7,
            3,
            2,
            4,
            7,
            6,
            4,
            6,
            5,
        ],
        dtype=np.uint32,
    )
    I = I.view(gloo.IndexBuffer)

    # Index buffer for object edges
    O = np.array(
        [0, 1, 1, 2, 2, 3, 3, 0, 4, 7, 7, 6, 6, 5, 5, 4, 0, 5, 1, 6, 2, 7, 3, 4],
        dtype=np.uint32,
    )
    O = O.view(gloo.IndexBuffer)

    for iterator in range(4):
        cube = gloo.Program(vertex, fragment)
        cube.bind(V)
        cube["model"] = np.eye(4, dtype=np.float32)
        cube["view"] = glm.translation(0, 0, -30)
        bag[iterator] = cube

    # Define vertex for background texture and index buffer
    global quad
    quad = gloo.Program(vertex2, fragment2, count=4)
    quad["position"] = [(-1, -1, 0), (-1, +1, 0), (+1, -1, 0), (+1, +1, 0)]
    quad["texcoord"] = [(0, 1), (0, 0), (1, 1), (1, 0)]
    quad["texture"] = initial_image[..., ::-1]

    global path
    path = gloo.Program(vertex3, fragment, count=4 * 25 * 2)

    # Initialise the vertex buffers for path
    path["position"] = np.zeros((4 * 25 * 2, 3))
    path["last_position"] = np.zeros((4 * 25 * 2, 3))
    path["side"] = np.zeros(4 * 25 * 2)

    # Color of path
    path["u_color"] = 0, 0.5, 0

    # Define the Index Buffer for edges of the augmented reality path.
    bline_I = list()
    bline_I += [2, 3]
    for i in range(2, 4 * 25 * 2 - 2, 2):
        bline_I += [i, i + 2]
        bline_I += [i + 1, i + 3]
    bline_I = np.array(bline_I, dtype=np.uint32)
    bline_I = bline_I.view(gloo.IndexBuffer)

    # Initialise the OpenGL window
    window = app.Window(width=img_width, height=img_height, color=(1, 1, 1, 1))

    @window.event
    def on_draw(dt):
        global phi, theta

        # Phi and Theta are the cube rotation paramters
        window.clear()

        lock.acquire()
        try:
            # Disable depth of OpenGL to update background tecture
            gl.glDisable(gl.GL_DEPTH_TEST)

            quad.draw(gl.GL_TRIANGLE_STRIP)

            # R-enable depth
            gl.glEnable(gl.GL_DEPTH_TEST)

            # Color of path
            path["u_color"] = 0, 1, 1
            # Filled path
            path.draw(gl.GL_TRIANGLE_STRIP)
            # Mask depth
            gl.glDepthMask(gl.GL_FALSE)
            # Color of edge lines of path
            path["u_color"] = 0, 0, 0
            # Width of edge lines
            gl.glLineWidth(10.0)
            # Draw edge lines with index buffer bline_I
            path.draw(gl.GL_LINES, bline_I)
            # Reset line width
            gl.glLineWidth(1.0)
            gl.glDepthMask(gl.GL_TRUE)

            # Define the model matrix with updated rotation
            model = np.eye(4, dtype=np.float32)
            glm.rotate(model, theta, 0, 0, 1)
            glm.rotate(model, phi, 0, 1, 0)

            for obj in bag.values():
                obj["u_color"] = 1, 0, 0

                # Filled cube
                obj.draw(gl.GL_TRIANGLES, I)

                # Another method to disable depth, instead of disabling it, mask it
                gl.glDepthMask(gl.GL_FALSE)
                # Black color for edge lines of cube
                obj["u_color"] = 0, 0, 0
                # Draw the edge lines with the given index buffer
                obj.draw(gl.GL_LINES, O)
                # Unmask OpenGL depth aparamter
                gl.glDepthMask(gl.GL_TRUE)

                # Model matrix is used to define orientation ,in this case, used to rotate cube
                obj["model"] = model

            # Update cube rotations
            theta += 2.0  # degrees
            phi += 2.0  # degrees
        finally:
            lock.release()

    @window.event
    def on_resize(width, height):

        # Redefine projection matrix from OpenCV style to OpenGL style
        # OpenCV defines 3d image from left top corner as (0,0,0) and x and y increase towards right and down respectively.
        # z increases positively outwards
        # OpenGL defines 3d image from center as (0,0,0)  and x and y increase towards right and up respectiively.
        # z increases negatively outwards
        # Clipping (1m - 30m)
        # Source - https://blog.noctua-software.com/opencv-opengl-projection-matrix.html
        view_matrix = np.zeros((4, 4))
        view_matrix[0][0] = 2.0 * projection_matrix[0] / img_width
        view_matrix[1][1] = -2.0 * projection_matrix[5] / img_height

        view_matrix[2][0] = 1.0 - 2.0 * projection_matrix[2] / img_width
        view_matrix[2][1] = 2.0 * projection_matrix[6] / img_height - 1.0
        view_matrix[2][2] = (30 + 1) / float(1 - 30)
        view_matrix[2][3] = -1.0

        view_matrix[3][2] = 2.0 * 30 * 1 / (1 - 30)

        for obj in bag.values():
            obj["projection"] = view_matrix

        # Use same projection amtrix for path but redefine clipping to (0.01m - 30m)
        view_matrix[2][2] = (30 + 0.01) / float(0.01 - 30)
        view_matrix[3][2] = 2.0 * 30 * 0.01 / (0.01 - 30)
        path["projection"] = view_matrix

    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)

    app.run(framerate=30)


if __name__ == "__main__":
    listener()
