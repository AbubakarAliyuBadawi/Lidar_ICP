# LiDAR Point Cloud Registration using ICP

## Overview
This ROS node implements real-time point cloud registration using the Iterative Closest Point (ICP) algorithm. It processes incoming LiDAR point cloud data, aligns consecutive frames, and maintains a cumulative transformation to track the sensor's movement over time. The node includes robust error checking and filtering to ensure reliable operation.

## Features
- Real-time point cloud alignment using ICP
- Robust point cloud filtering and downsampling
- Transform validation to detect unrealistic movements
- Accumulated point cloud visualization
- TF tree integration for coordinate frame management

## Dependencies
- ROS (tested on ROS Noetic)
- Python 3
- Point Cloud Library (PCL)
- NumPy
- TF2

## Installation
1. Clone this repository into your ROS workspace's `src` directory:
```bash
cd ~/catkin_ws/src
git clone [repository-url]
```

2. Install the required dependencies:
```bash
sudo apt-get install ros-noetic-pcl-ros python3-pcl
pip3 install numpy
```

3. Build your workspace:
```bash
cd ~/catkin_ws
catkin_make
```

## Node Details

### Subscribed Topics
- `/os_cloud_node/points` (sensor_msgs/PointCloud2): Input point cloud data from LiDAR sensor

### Published Topics
- `/aligned_points` (sensor_msgs/PointCloud2): Accumulated aligned point cloud in the fixed frame
- `/current_aligned` (sensor_msgs/PointCloud2): Current frame aligned to the fixed frame
- `/tf` (tf2_msgs/TFMessage): Transform broadcasts

### Parameters
- ICP Parameters:
  - `max_correspondence_distance`: 0.5 meters
  - `max_iterations`: 50
  - `fitness_epsilon`: 1e-6

- Filtering Parameters:
  - `leaf_size`: 0.2 meters (voxel grid filter)
  - `min_fitness_score`: 0.01

- Motion Limits:
  - `max_translation`: 5.0 meters
  - `max_rotation`: 1.0 radians (~57 degrees)

## Implementation Details

### Point Cloud Processing Pipeline
1. **Preprocessing**
   - Conversion from ROS message to PCL format
   - Voxel grid downsampling
   - Statistical outlier removal

2. **ICP Registration**
   - Frame-to-frame alignment
   - Convergence checking
   - Fitness score validation

3. **Transform Validation**
   - Translation limit checking
   - Rotation limit checking
   - Cumulative transform maintenance

4. **Publishing**
   - Aligned point clouds
   - Transform broadcasts
   - Debug information

### Key Features Explained

#### Point Cloud Filtering
The node implements two-stage filtering:
```python
def filter_cloud(self, cloud_pcl):
    # Voxel Grid filter for downsampling
    vg = cloud_pcl.make_voxel_grid_filter()
    vg.set_leaf_size(self.leaf_size, self.leaf_size, self.leaf_size)
    filtered = vg.filter()
    
    # Statistical Outlier Removal
    sor = filtered.make_statistical_outlier_filter()
    sor.set_mean_k(30)
    sor.set_std_dev_mul_thresh(2.0)
    filtered = sor.filter()
    
    return filtered
```

#### Transform Validation
Ensures physically realistic movements:
```python
def check_transform_validity(self, transform):
    # Check translation magnitude
    translation = transform[:3, 3]
    translation_magnitude = np.linalg.norm(translation)
    if translation_magnitude > self.max_translation:
        return False, "Excessive translation detected"
    
    # Check rotation magnitude
    rotation = transform[:3, :3]
    euler_angles = tf.transformations.euler_from_matrix(rotation)
    max_rotation = max(abs(angle) for angle in euler_angles)
    if max_rotation > self.max_rotation:
        return False, "Excessive rotation detected"
```

## Usage

1. Launch your LiDAR sensor's ROS driver.

2. Run the ICP node:
```bash
rosrun [package_name] lidar_icp_node.py
```

3. Visualize the results in RViz:
   - Add a PointCloud2 display for `/aligned_points`
   - Add a PointCloud2 display for `/current_aligned`
   - Enable TF visualization

## Performance Considerations

- The node includes various parameters that can be tuned for different scenarios:
  - Decrease `leaf_size` for more precise alignment at the cost of computational speed
  - Adjust `max_correspondence_distance` based on expected movement between frames
  - Modify `max_translation` and `max_rotation` based on your system's movement characteristics

- Memory usage increases over time as points are accumulated. Consider implementing a sliding window or octree-based point cloud management for long-term operation.

## Debug Information
The node provides detailed logging information:
- Frame processing status
- ICP convergence and fitness scores
- Transform validation results
- Point cloud statistics

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License
[Your License Here]

## Acknowledgments
- Point Cloud Library (PCL) for point cloud processing capabilities
- ROS community for the robust robotics framework
