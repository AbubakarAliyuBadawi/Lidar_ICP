cmake_minimum_required(VERSION 3.0.2)
project(lidar_icp)

find_package(catkin REQUIRED COMPONENTS
  pcl_msgs
  pcl_ros
  rospy
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS pcl_msgs pcl_ros rospy sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/lidar_icp_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
