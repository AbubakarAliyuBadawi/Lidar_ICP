#ifndef LIDAR_ICP_NODE_HPP_
#define LIDAR_ICP_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <Eigen/Core>

class LidarICPNode : public rclcpp::Node {
public:
    explicit LidarICPNode();

private:
    // ICP and filtering parameters
    double max_correspondence_distance_;
    int max_iterations_;
    double fitness_epsilon_;
    double leaf_size_;
    double min_fitness_score_;
    double max_translation_;
    double max_rotation_;
    std::string fixed_frame_;

    // ROS publishers and subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Add these to private section of LidarICPNode class
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_cloud_;  // Store all aligned points
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_current_;  // Publisher for current cloud

    // State variables
    pcl::PointCloud<pcl::PointXYZ>::Ptr reference_cloud_;
    Eigen::Matrix4f reference_to_map_;
    int callback_count_;

    // Member functions
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    bool checkTransformValidity(const Eigen::Matrix4f& transform);
    std::tuple<bool, Eigen::Matrix4f, std::string> performICP(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target);
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const Eigen::Matrix4f& transform);
    void publishTransform(const Eigen::Matrix4f& transform,
                         const rclcpp::Time& timestamp,
                         const std::string& child_frame);
};

#endif
