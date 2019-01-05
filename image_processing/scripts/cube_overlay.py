from threading import Thread
import rospy
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import image_geometry as ig

from glumpy import app, gl, glm, gloo, data
import numpy as np
from glumpy.transforms import OrthographicProjection, Position
import os

img_width = 900
img_height = 900
phi, theta = 40, 30
count = 0


def opengl():
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]

    vertex = """
    attribute vec3 position;      // Vertex position
    void main()
    {
        gl_Position = <transform>;
    }
    """

    fragment = """
    void main()
    {
        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
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

    V = np.zeros(8, [("position", np.float32, 3)])
    V["position"] = [[100, 100, 100], [0, 100, 100], [0, 0, 100], [100, 0, 100],
                     [100, 0, -100], [100, 100, -100], [0, 100, -100], [0, 0, -100]]
    V = V.view(gloo.VertexBuffer)
    I = np.array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 1,
                  1, 6, 7, 1, 7, 2, 7, 4, 3, 7, 3, 2, 4, 7, 6, 4, 6, 5], dtype=np.uint32)
    I = I.view(gloo.IndexBuffer)

    # for iterator in range(4):
    cube = gloo.Program(vertex, fragment)
    cube.bind(V)
    cube["transform"] = OrthographicProjection(Position("position"))
    # bag[iterator] = cube

    quad = gloo.Program(vertex2, fragment2, count=4)
    quad['position'] = [(-1, -1, 0), (-1, +1, 0), (+1, -1, 0), (+1, +1, 0)]
    quad['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
    quad['texture'] = data.get(os.path.realpath("/home/rahul/Pictures/GitKraken_001.png"))

    window = app.Window(width=img_width, height=img_height, color=(1, 1, 1, 1))
    window.attach(cube["transform"])

    @window.event
    def on_draw(dt):
        global phi, theta, count

        window.clear()
        gl.glDisable(gl.GL_DEPTH_TEST)
        quad.draw(gl.GL_TRIANGLE_STRIP)

        gl.glEnable(gl.GL_DEPTH_TEST)
        # Filled cube

        # for obj in to_draw:
            # obj['view'] = glm.translation(0, 0, -40)
        cube.draw(gl.GL_TRIANGLES, I)

        # Make cube rotate
        #theta += 1.0  # degrees
        #phi += 1.0  # degrees
        #model = np.eye(4, dtype=np.float32)
        #glm.rotate(model, theta, 0, 0, 1)
        #glm.rotate(model, phi, 0, 1, 0)
        #cube['model'] = model

    @window.event
    def on_resize(width, height):
        # for obj in bag.values():
        pass

    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)

    app.run()


if __name__ == '__main__':
    Thread(target=opengl).start()
