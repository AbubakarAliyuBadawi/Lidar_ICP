# LiDAR ICP Node

A ROS node for real-time point cloud registration and mapping using the Iterative Closest Point (ICP) algorithm. This node performs continuous alignment of incoming LiDAR scans, builds an accumulated point cloud map, and provides transform publications for localization.

## Overview

The LiDAR ICP Node subscribes to point cloud data from a LiDAR sensor, processes each incoming frame using the ICP algorithm to align it with previous scans, and maintains a cumulative map of the environment. It includes robust error checking, point cloud filtering, and transform validation to ensure reliable operation.

## Key Features

- Real-time point cloud registration using ICP
- Robust point cloud filtering including:
 - Voxel grid downsampling
 - Statistical outlier removal
- Transform validation with configurable limits
- Accumulated point cloud mapping
- Real-time TF tree updates
- Comprehensive error checking and handling
- Configurable parameters for fine-tuning performance

## Prerequisites

### Required Software
- ROS Noetic
- Python 3.x
- Point Cloud Library (PCL)
- NumPy
- TF2

### Installation

1. Install ROS dependencies:
```bash
sudo apt-get install ros-noetic-pcl-ros python3-pcl python3-tf2-ros
