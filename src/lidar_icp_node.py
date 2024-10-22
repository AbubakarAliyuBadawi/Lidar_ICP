#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pcl
import tf
import tf2_ros
import numpy as np
from std_msgs.msg import Header

class LidarICPNode:
    def __init__(self):
        rospy.init_node('lidar_icp_node', anonymous=True)
        self.subscriber = rospy.Subscriber('/os_cloud_node/points', PointCloud2, self.callback)
        self.pub_aligned = rospy.Publisher('/aligned_points', PointCloud2, queue_size=1)
        self.t0_cloud_ros = None
        self.t0_cloud_pcl = None
        self.static_frame = "t0_static_frame"
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.cumulative_transform = np.identity(4)
        self.accumulated_points = []
        self.callback_count = 0
        rospy.loginfo("Lidar ICP Node Initialized - Publishing reverse transform (os_sensor -> t0_static_frame)")

    def print_transform_info(self, transform_matrix, description):
        """Print detailed information about a transformation matrix."""
        rospy.loginfo(f"\n{description}:")
        rospy.loginfo("Transformation Matrix:")
        for row in transform_matrix:
            rospy.loginfo(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}]")
        
        translation = transform_matrix[:3, 3]
        rospy.loginfo(f"Translation [x, y, z]: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]")
        
        euler_angles = tf.transformations.euler_from_matrix(transform_matrix)
        rospy.loginfo(f"Rotation [roll, pitch, yaw] (degrees): [{np.degrees(euler_angles[0]):.6f}, {np.degrees(euler_angles[1]):.6f}, {np.degrees(euler_angles[2]):.6f}]")

    def transform_cloud(self, cloud_pcl, transform):
        """Transform PCL point cloud using transformation matrix."""
        points = np.asarray(cloud_pcl)
        transformed_points = []
        
        for point in points:
            p = np.array([point[0], point[1], point[2], 1.0])
            transformed = np.dot(transform, p)
            transformed_points.append(transformed[:3])
            
        transformed_cloud = pcl.PointCloud()
        transformed_cloud.from_array(np.array(transformed_points, dtype=np.float32))
        return transformed_cloud

    def callback(self, data):
        self.callback_count += 1
        rospy.loginfo(f"\n------ Callback {self.callback_count} started ------")
        rospy.loginfo(f"Received PointCloud2 message with frame_id: {data.header.frame_id}")

        points_list = []
        for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        
        if not points_list:
            rospy.logwarn("No valid points in the point cloud. Skipping this frame.")
            return

        rospy.loginfo(f"Number of points in input cloud: {len(points_list)}")
        
        current_cloud_pcl = pcl.PointCloud()
        current_cloud_pcl.from_list(points_list)

        # Downsample and filter
        vg = current_cloud_pcl.make_voxel_grid_filter()
        leaf_size = 0.1
        vg.set_leaf_size(leaf_size, leaf_size, leaf_size)
        current_cloud_filtered = vg.filter()
        rospy.loginfo(f"Points after voxel grid filter: {current_cloud_filtered.size}")

        sor = current_cloud_filtered.make_statistical_outlier_filter()
        sor.set_mean_k(50)
        sor.set_std_dev_mul_thresh(1.0)
        current_cloud_filtered = sor.filter()
        rospy.loginfo(f"Points after statistical outlier removal: {current_cloud_filtered.size}")

        if self.t0_cloud_pcl is None:
            self.t0_cloud_pcl = current_cloud_filtered
            self.t0_cloud_ros = data
            
            rospy.loginfo("\nInitializing t0 reference frame:")
            rospy.loginfo(f"t0 cloud size: {self.t0_cloud_pcl.size} points")
            self.print_transform_info(self.cumulative_transform, "Initial cumulative transform (Identity)")
            
            self.accumulated_points = points_list.copy()
            
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.static_frame
            t0_cloud_msg = pc2.create_cloud_xyz32(header, self.accumulated_points)
            self.pub_aligned.publish(t0_cloud_msg)
            
            # Publishing initial transform in reverse direction
            t = tf2_ros.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = data.header.frame_id  # os_sensor is now the parent frame
            t.child_frame_id = self.static_frame      # t0_static_frame is now the child frame
            
            # Identity transform
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.w = 1.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            
            self.tf_broadcaster.sendTransform(t)
            rospy.loginfo(f"Stored t0 cloud and published initial TF: {data.header.frame_id} -> {self.static_frame}")
            return

        # Perform ICP
        icp = current_cloud_filtered.make_IterativeClosestPoint()
        converged, transf, estimate, fitness = icp.icp(current_cloud_filtered, self.t0_cloud_pcl)

        if converged:
            rospy.loginfo("\nICP Results:")
            rospy.loginfo(f"Converged: {converged}")
            rospy.loginfo(f"Fitness score: {fitness}")
            
            self.print_transform_info(transf, "ICP Transform (current -> t0)")
            self.print_transform_info(self.cumulative_transform, "Previous Cumulative Transform")
            
            # Update cumulative transform
            self.cumulative_transform = np.dot(self.cumulative_transform, transf)
            
            self.print_transform_info(self.cumulative_transform, "Updated Cumulative Transform")
            
            # Transform current cloud
            transformed_cloud = self.transform_cloud(current_cloud_filtered, self.cumulative_transform)
            transformed_points = transformed_cloud.to_array().tolist()
            
            # Add transformed points
            self.accumulated_points.extend(transformed_points)
            rospy.loginfo(f"Added {len(transformed_points)} new points to accumulated cloud")
            
            # Publish accumulated cloud
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.static_frame
            accumulated_cloud_msg = pc2.create_cloud_xyz32(header, self.accumulated_points)
            self.pub_aligned.publish(accumulated_cloud_msg)
            
            # Publish TF in reverse direction
            t = tf2_ros.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = data.header.frame_id  # os_sensor is now the parent frame
            t.child_frame_id = self.static_frame      # t0_static_frame is now the child frame
            
            # Use cumulative transform directly (no inversion needed)
            t.transform.translation.x = self.cumulative_transform[0, 3]
            t.transform.translation.y = self.cumulative_transform[1, 3]
            t.transform.translation.z = self.cumulative_transform[2, 3]
            
            quat = tf.transformations.quaternion_from_matrix(self.cumulative_transform)
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            
            self.tf_broadcaster.sendTransform(t)
            rospy.loginfo("\nTransform Broadcasting:")
            rospy.loginfo(f"Published TF: {data.header.frame_id} -> {self.static_frame}")
            rospy.loginfo(f"Translation: [{t.transform.translation.x:.6f}, {t.transform.translation.y:.6f}, {t.transform.translation.z:.6f}]")
            rospy.loginfo(f"Rotation (quaternion): [{t.transform.rotation.x:.6f}, {t.transform.rotation.y:.6f}, {t.transform.rotation.z:.6f}, {t.transform.rotation.w:.6f}]")
            rospy.loginfo(f"Total accumulated points: {len(self.accumulated_points)}")
        else:
            rospy.logwarn("ICP did not converge. Skipping this frame.")

        rospy.loginfo(f"\n------ Callback {self.callback_count} completed ------")

if __name__ == '__main__':
    try:
        node = LidarICPNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
