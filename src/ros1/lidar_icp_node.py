#!/usr/bin/env python3

"""
ROS node for performing Iterative Closest Point (ICP) alignment on LiDAR point clouds.
This node subscribes to incoming point cloud data, performs ICP registration with previous frames,
and publishes the aligned point clouds along with their transformations.

The node includes functionality for:
- Point cloud filtering and downsampling
- ICP-based registration between consecutive frames
- Transform validation to detect unrealistic movements
- Publishing aligned point clouds and transforms to the ROS ecosystem
"""

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pcl
import tf
import tf2_ros
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped

class LidarICPNode:
    """
    A ROS node that performs real-time point cloud alignment using ICP algorithm.
    
    This class handles the continuous processing of incoming LiDAR point clouds,
    maintaining their spatial relationships, and publishing the results to the ROS network.
    It includes various parameters for tuning the ICP process and filtering steps.
    """

    def __init__(self):
        """
        Initialize the LidarICPNode with configuration parameters and ROS setup.
        Sets up subscribers, publishers, and transform broadcasters.
        """
        rospy.init_node('LidarICPNode', anonymous=True)
        
        # ICP (Iterative Closest Point) Parameters
        self.max_correspondence_distance = 0.5  # Maximum distance (in meters) between points to be considered a match
        self.max_iterations = 50                # Number of ICP iterations before giving up
        self.fitness_epsilon = 1e-6             # Stop iterating if improvement is less than this value

        # Point Cloud Processing Parameters
        self.leaf_size = 0.2                    # Size (in meters) of voxel grid cells for downsampling point cloud
        self.min_fitness_score = 0.01           # Maximum acceptable average distance between matched points (lower is better)

        # Motion Limit Parameters
        self.max_translation = 5.0              # Maximum allowed movement (in meters) between consecutive frames
        self.max_rotation = 1.0                 # Maximum allowed rotation (in radians) between consecutive frames (~57 degrees)
        
        # Subscribers and Publishers
        self.subscriber = rospy.Subscriber('/os_cloud_node/points', PointCloud2, self.callback)
        self.pub_aligned = rospy.Publisher('/aligned_points', PointCloud2, queue_size=1)
        self.pub_current = rospy.Publisher('/current_aligned', PointCloud2, queue_size=1)
        
        # TF handling
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.fixed_frame = "map"  # Global fixed frame for alignment
        
        # State variables
        self.previous_cloud = None              # Store previous frame for ICP alignment
        self.cumulative_transform = np.identity(4)  # Accumulated transform from initial frame
        self.accumulated_points = []            # List of all aligned points in fixed frame
        self.callback_count = 0                 # Counter for processed frames

    def filter_cloud(self, cloud_pcl):
        """
        Apply filtering operations to input point cloud to improve quality and reduce computation time.
        
        Args:
            cloud_pcl (pcl.PointCloud): Input point cloud to be filtered
            
        Returns:
            pcl.PointCloud: Filtered point cloud
        """
        # Voxel Grid filter for downsampling
        vg = cloud_pcl.make_voxel_grid_filter()
        vg.set_leaf_size(self.leaf_size, self.leaf_size, self.leaf_size)
        filtered = vg.filter()
        
        # Statistical Outlier Removal with adjusted parameters
        sor = filtered.make_statistical_outlier_filter()
        sor.set_mean_k(30)  # Number of neighbors to analyze for each point
        sor.set_std_dev_mul_thresh(2.0)  # Standard deviation threshold for outlier removal
        filtered = sor.filter()
        
        return filtered

    def check_transform_validity(self, transform):
        """
        Verify if the calculated transformation is physically realistic.
        
        Args:
            transform (np.array): 4x4 transformation matrix to be validated
            
        Returns:
            tuple: (bool, str) indicating if transform is valid and explanation message
        """
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
        
        return True, "Transform within limits"

    def perform_icp(self, source_cloud, target_cloud):
        """
        Execute ICP alignment between two point clouds with error checking.
        
        Args:
            source_cloud (pcl.PointCloud): Point cloud to be aligned
            target_cloud (pcl.PointCloud): Reference point cloud
            
        Returns:
            tuple: (success_bool, transformation_matrix, status_message)
        """
        # Create ICP object
        icp = source_cloud.make_IterativeClosestPoint()
        
        # Perform ICP
        converged, transf, estimate, fitness = icp.icp(source_cloud, target_cloud)
        
        rospy.loginfo(f"ICP Fitness Score: {fitness}")
        rospy.loginfo(f"Convergence Status: {converged}")
        
        if not converged:
            return False, None, "ICP failed to converge"
        
        # Only check fitness if convergence succeeded
        if fitness < self.min_fitness_score:
            return False, None, f"Poor alignment score: {fitness}"
        
        # Check if transform is reasonable
        valid_transform, message = self.check_transform_validity(transf)
        if not valid_transform:
            # Log the actual translation for debugging
            translation = transf[:3, 3]
            translation_magnitude = np.linalg.norm(translation)
            rospy.loginfo(f"Translation magnitude: {translation_magnitude}")
            return False, None, message
        
        return True, transf, f"Success with fitness: {fitness}"

    def transform_cloud(self, cloud_pcl, transform):
        """
        Apply transformation matrix to point cloud.
        
        Args:
            cloud_pcl (pcl.PointCloud): Point cloud to transform
            transform (np.array): 4x4 transformation matrix
            
        Returns:
            pcl.PointCloud: Transformed point cloud
        """
        points = np.asarray(cloud_pcl)
        transformed_points = []
        
        for point in points:
            p = np.array([point[0], point[1], point[2], 1.0])
            transformed = np.dot(transform, p)
            transformed_points.append(transformed[:3])
        
        transformed_cloud = pcl.PointCloud()
        transformed_cloud.from_array(np.array(transformed_points, dtype=np.float32))
        return transformed_cloud

    def publish_transform(self, transform, timestamp, child_frame):
        """
        Publish transformation to ROS TF tree.
        
        Args:
            transform (np.array): 4x4 transformation matrix
            timestamp (rospy.Time): Timestamp for the transform
            child_frame (str): Name of the child frame in the transform
        """
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.fixed_frame
        t.child_frame_id = child_frame
        
        t.transform.translation.x = transform[0, 3]
        t.transform.translation.y = transform[1, 3]
        t.transform.translation.z = transform[2, 3]
        
        quat = tf.transformations.quaternion_from_matrix(transform)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    def callback(self, data):
        """
        Main callback function for processing incoming point cloud messages.
        
        This function handles the core logic of the node:
        1. Converts ROS message to PCL format
        2. Filters and preprocesses the point cloud
        3. Performs ICP alignment with previous frame
        4. Updates and publishes transforms and aligned point clouds
        
        Args:
            data (sensor_msgs.msg.PointCloud2): Incoming point cloud message
        """
        self.callback_count += 1
        rospy.loginfo(f"\n------ Processing frame {self.callback_count} ------")
        
        # Convert ROS message to PCL
        points_list = []
        for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        
        if not points_list:
            rospy.logwarn("Empty point cloud received")
            return
        
        current_cloud = pcl.PointCloud()
        current_cloud.from_list(points_list)
        
        # Filter current cloud
        current_filtered = self.filter_cloud(current_cloud)
        rospy.loginfo(f"Filtered cloud size: {current_filtered.size}")
        
        # Handle first frame
        if self.previous_cloud is None:
            self.previous_cloud = current_filtered
            self.accumulated_points = points_list
            
            # Publish initial cloud in fixed frame
            header = Header()
            header.stamp = data.header.stamp
            header.frame_id = self.fixed_frame
            cloud_msg = pc2.create_cloud_xyz32(header, points_list)
            self.pub_aligned.publish(cloud_msg)
            
            # Publish initial transform
            self.publish_transform(self.cumulative_transform, data.header.stamp, data.header.frame_id)
            return
        
        # Perform ICP with previous frame
        success, transform, message = self.perform_icp(current_filtered, self.previous_cloud)
        
        if not success:
            rospy.logwarn(f"ICP failed: {message}")
            return
        
        # Update transforms and points
        self.cumulative_transform = np.dot(self.cumulative_transform, transform)
        
        # Transform current points to fixed frame
        transformed_cloud = self.transform_cloud(current_filtered, self.cumulative_transform)
        transformed_points = transformed_cloud.to_array().tolist()
        
        # Add to accumulated points (with optional downsampling)
        self.accumulated_points.extend(transformed_points)
        
        # Publish results
        header = Header()
        header.stamp = data.header.stamp
        header.frame_id = self.fixed_frame
        
        # Publish accumulated cloud
        accumulated_msg = pc2.create_cloud_xyz32(header, self.accumulated_points)
        self.pub_aligned.publish(accumulated_msg)
        
        # Publish current aligned cloud
        current_msg = pc2.create_cloud_xyz32(header, transformed_points)
        self.pub_current.publish(current_msg)
        
        # Publish transform
        self.publish_transform(self.cumulative_transform, data.header.stamp, data.header.frame_id)
        
        # Update previous cloud for next iteration
        self.previous_cloud = current_filtered
        
        rospy.loginfo(f"Successfully processed frame {self.callback_count}")
        rospy.loginfo(f"Total accumulated points: {len(self.accumulated_points)}")

if __name__ == '__main__':
    try:
        node = LidarICPNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
