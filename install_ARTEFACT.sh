#!/bin/bash

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt --assume-yes install ros-kinetic-desktop-full
sudo rosdep init
rosdep update
if ! grep -q "source /opt/ros/kinetic/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
    . ~/.bashrc
else
    . ~/.bashrc
fi
sleep 2
sudo apt --assume-yes install python-rosinstall python-rosinstall-generator python-wstool build-essential
sudo apt --assume-yes install python-rosdep python-catkin-tools
sudo apt --assume-yes install python-pip
sudo -H pip install --upgrade pip
sudo apt --assume-yes install libglfw3-dev libglfw3
python -m pip install --user numpy scipy
sudo apt -assume-yes install libglu1-mesa-dev freeglut3-dev mesa-common-dev
python -m pip install --user cython
python -m pip install --user triangle
python -m pip install --user PyOpenGL PyOpenGL_accelerate
python -m pip install --user glumpy
sudo apt --assume-yes install ros-kinetic-ardrone-autonomy ros-kinetic-joy ros-kinetic-gps-common ros-kinetic-hector*
sleep 2
. /opt/ros/kinetic/setup.bash
mkdir -p ~/artefact_ws/src
cd ~/artefact_ws/src
git clone https://github.com/rahulghangas/artefact.git
cd ..
rosdep update && rosdep install --from-paths src -y -i
catkin init
catkin build bebop_autonomy
catkin_make --override-build-tool-check
