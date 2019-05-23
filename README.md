# Artefact

ARTEFACT stands for Augmented Reality Tool Enhancing Flight ACTions.
It aims to improve the flight experience with UAVs by inserting  spatially positioned objects despite the imprecision of UAV positioning systems.

# Installation guide

+ Install on target host
	1. ros-kinetic
	2. Install following packages using terminal
	```sh
	$ sudo apt-get install build-essential python-rosdep python-catkin-tools
	$ sudo install python-pip
	$ sudo apt-get install libglfw3-dev libglfw3
	$ python -m pip install --user numpy cython scipy matplotlib ipython jupyter sympy nose 
	$ sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
	$ python -m pip install --user glumpy
	$ python -m pip install --user triangle
	$ sudo apt install ros-kinetic-ardrone-autonomy
	$ sudo apt install ros-kinetic-joy
	$ sudo apt install ros-kinetic-gps-common
	$ sudo apt install ros-kinetic-hector*
	```
+ Clone repository to src folder of your workspace
+ run following command from workspace root 
```sh
	$ rosdep update && rosdep install --from-paths src -i
```
+ Run "catkin_make" from workspace root. 
+ To start simulation, "roslaunch sim_launcher sim.launch" OR
+ To use with real drone, "roslaunch sim_launcher bebop.launch"
+ To fly quadrotor, plug in an xbox360 or logitech joystick controller.

![](resources/index.jpeg)
![](resources/index1.jpeg)
![](resources/index2.jpeg)
