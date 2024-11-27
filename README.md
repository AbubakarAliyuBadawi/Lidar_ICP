# Lidar ICP Node

The `LidarICPNode` is a ROS 2 package designed to align LiDAR point clouds using the Iterative Closest Point (ICP) algorithm. The node subscribes to a LiDAR point cloud topic, performs point cloud alignment, and publishes the aligned point cloud along with the transformation between the reference and fixed frames. It also features point cloud filtering to improve the quality of the alignment.

## Overview

This ROS 2 node implements the following key functionality:

- **ICP Algorithm**: The node applies the Iterative Closest Point (ICP) algorithm to align the incoming point clouds to a reference point cloud.
- **Point Cloud Filtering**: 
  - **Voxel Grid Filtering**: Used for downsampling the point cloud to reduce computation time.
  - **Statistical Outlier Removal**: Used to remove noisy points in the cloud before processing.
- **Transform Broadcasting**: The node broadcasts the transformation between the reference cloud and the fixed frame for use in localization and mapping.
- **Point Cloud Accumulation**: The node accumulates all aligned point clouds and publishes them for further use or visualization.

## Repository Structure

Here's an overview of the key files in the repository:

### 1. `src/lidar_icp_node.cpp`

This is the main implementation file for the `LidarICPNode` class. The file contains the core logic of the ICP-based point cloud alignment and the associated transformations. It handles the following:

- **Point Cloud Callback**: This method processes incoming point cloud messages, applies filtering, and performs the ICP alignment.
- **ICP Algorithm**: The ICP algorithm is applied to align the source point cloud to the reference.
- **Transformation Check**: The method `checkTransformValidity()` ensures that the generated transformation is valid based on translation and rotation thresholds.
- **Publishers and Subscribers**: The node subscribes to the LiDAR point cloud topic and publishes the aligned point cloud and transformations.

### 2. `include/lidar_icp/lidar_icp_node.hpp`

This is the header file for the `LidarICPNode` class. It declares the class and its member functions, including the ICP-specific parameters, state variables, and helper methods for filtering, transforming, and publishing data. This file also includes necessary ROS 2, PCL, and Eigen dependencies.

### 3. `launch/lidar_icp_launch.py`

This is the launch file for starting the `LidarICPNode`. The launch file configures the node, passes parameters, and remaps topics. Parameters such as the maximum correspondence distance, maximum iterations for ICP, and the fixed frame are set here. It also configures the topic remappings for the point cloud data and aligned output.

### 4. `CMakeLists.txt`

This file contains the build instructions for the package. It specifies dependencies (e.g., PCL, ROS 2) and includes instructions on how to build the project. The CMake file ensures that the necessary libraries are linked and compiled correctly.

### 5. `package.xml`

The `package.xml` file defines metadata for the ROS 2 package, including dependencies and version information. It ensures that the package is correctly recognized by ROS 2 and links it to the necessary dependencies.

### 6. `README.md`

This file provides an overview of the repository, installation instructions, configuration details, and usage. It is designed to help users understand the purpose of the project and how to use it.

## How It Works

The `LidarICPNode` node listens for incoming LiDAR point cloud messages from a specified topic, applies point cloud filtering, and performs the ICP alignment with a reference cloud. The node then publishes the aligned point cloud and broadcasts the transformation between the reference and fixed frame.

### Main Workflow

1. **Subscription**: The node subscribes to the LiDAR point cloud topic (e.g., `/os_cloud_node/points`).
2. **Filtering**: The incoming point cloud is filtered using a voxel grid filter and statistical outlier removal.
3. **ICP Alignment**: The filtered point cloud is aligned to the reference cloud using the ICP algorithm. The algorithm iteratively refines the transformation.
4. **Transformation Validation**: The node checks the validity of the transformation by comparing the translation and rotation against predefined thresholds.
5. **Publishing**: The aligned point cloud and the transformation are published to respective topics (`/accumulated_points`, `/current_aligned`), and the transformation is broadcasted.

### ICP Algorithm

The Iterative Closest Point (ICP) algorithm is used to align the incoming LiDAR point cloud with a reference point cloud. ICP finds the optimal rigid transformation (translation and rotation) that minimizes the distance between corresponding points in the source and target clouds.

The algorithm proceeds through several iterations, each time refining the transformation until the alignment converges (or the maximum iterations are reached).

### Transform Broadcasting

The node uses ROS 2’s `tf2` library to broadcast the transformation between the reference cloud and the fixed frame (usually `map`). This is useful for localization, mapping, and visualizing the alignment in a broader context (e.g., in RViz).

## Parameters

The node is configurable through the launch file or a parameter YAML file. Key parameters include:

- **max_correspondence_distance**: Maximum distance between points to consider as correspondences.
- **max_iterations**: Maximum number of iterations for the ICP algorithm.
- **fitness_epsilon**: Convergence threshold for the ICP algorithm.
- **leaf_size**: Size of the voxel grid for downsampling.
- **min_fitness_score**: Minimum fitness score required for a valid alignment.
- **max_translation**: Maximum allowed translation in the alignment.
- **max_rotation**: Maximum allowed rotation in the alignment.
- **fixed_frame**: The fixed reference frame for broadcasting transforms.

## How to Run

1. **Build the package**:
    ```bash
    colcon build --packages-select lidar_icp
    ```

2. **Source the workspace**:
    ```bash
    source ~/ros2_ws/install/setup.bash
    ```

3. **Launch the node**:
    Use the following command to launch the node:
    ```bash
    ros2 launch lidar_icp lidar_icp_launch.py
    ```

4. **Visualize the result**:
    You can visualize the point clouds and transformations in RViz by subscribing to the `/accumulated_points` and `/current_aligned` topics.

## Topics

- **Input**: `/os_cloud_node/points` (sensor_msgs/PointCloud2) - LiDAR point cloud to be aligned.
- **Output**:
  - `/accumulated_points` (sensor_msgs/PointCloud2) - Accumulated aligned point cloud.
  - `/current_aligned` (sensor_msgs/PointCloud2) - Current aligned point cloud.
- **Transforms**: Transformation from reference frame to fixed frame (e.g., `map`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
