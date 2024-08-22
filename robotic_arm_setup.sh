#!/bin/bash

# Exit on any error
set -e

# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Define ROS2 distribution variable
ROS_DISTRO=humble

# Create and build the ROS2 workspace
echo "Creating and building the ROS2 workspace..."
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build --symlink-install

# Install necessary dependencies
echo "Installing dependencies..."
sudo apt update
sudo apt install -y python3-vcstool
sudo apt install -y ros-humble-test-msgs ros-humble-control-toolbox ros-humble-gazebo-ros-pkgs ros-humble-xacro ros-humble-joint-state-publisher-gui

# Download and build ROS2 Control Framework
echo "Downloading and building ROS2 Control Framework..."
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/
vcs import --input https://raw.githubusercontent.com/ros-controls/ros2_control_ci/master/ros_controls.$ROS_DISTRO.repos src
rosdep update --rosdistro=$ROS_DISTRO
sudo apt-get update
rosdep install --from-paths src --ignore-src -r -y
source /opt/ros/$ROS_DISTRO/setup.sh
colcon build --symlink-install --executor sequential

# Download and build Gazebo ROS2 Control
echo "Downloading and building Gazebo ROS2 Control..."
cd ~/ros2_ws/src
git clone -b humble https://github.com/ros-simulation/gazebo_ros2_control.git
cd ..
colcon build --symlink-install

# Clone and build the Robotic Arm Repository
echo "Cloning and building the Robotic Arm Repository..."
cd ~/ros2_ws/src
git clone https://github.com/nandu-k01/robotic_arm_environment.git
cd ..
colcon build --symlink-install

echo "Setup complete. Your workspace is ready and the robotic arm repository has been built."
