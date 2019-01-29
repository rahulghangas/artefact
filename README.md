# Artefact

ARTEFACT stands for Augmented Reality Tool Enhancing Flight ACTions.
It aims to improve the flight experience with UAVs by inserting  spatially positioned objects despite the imprecision of UAV positioning systems.

# Installation guide

1. Install on target host
  1. ros-kinetic
  2. Install following packages using terminal
	1. **only if needed** sudo install python2-pip
	2. sudo apt-get install libglfw3-dev libglfw3
	3. python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
	4. sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
	5. python -m pip install --user glumpy
	6. python -m pip install --user triangle
	7. python -m pip install --user cython
	8. sudo apt install ros-kinetic-ardrone-autonomy
	9. sudo apt install ros-kinetic-joy
	10. sudo apt install ros-kinetic-gps-common
	11. sudo apt install ros-kinetic-hector*
2. Clone repository to src folder of your workspace
3. Run "catkin_build" from root. 
4. To start simulation, "roslaunch sim_launcher sim.launch"
5. To fly quadrotor, plug in an xbox360 or logitech joystick controller.
