#include "lidar_icp/lidar_icp_node.hpp"
#include <pcl_conversions/pcl_conversions.h>

LidarICPNode::LidarICPNode() : Node("lidar_icp_node"), callback_count_(0) {
    // Initialize parameters
    declare_parameter("max_correspondence_distance", 0.5);
    declare_parameter("max_iterations", 50);
    declare_parameter("fitness_epsilon", 1e-6);
    declare_parameter("leaf_size", 0.2);
    declare_parameter("min_fitness_score", 5.0);
    declare_parameter("max_translation", 5.0);
    declare_parameter("max_rotation", 3.15);
    declare_parameter("fixed_frame", "map");

    // Get parameters
    max_correspondence_distance_ = get_parameter("max_correspondence_distance").as_double();
    max_iterations_ = get_parameter("max_iterations").as_int();
    fitness_epsilon_ = get_parameter("fitness_epsilon").as_double();
    leaf_size_ = get_parameter("leaf_size").as_double();
    min_fitness_score_ = get_parameter("min_fitness_score").as_double();
    max_translation_ = get_parameter("max_translation").as_double();
    max_rotation_ = get_parameter("max_rotation").as_double();
    fixed_frame_ = get_parameter("fixed_frame").as_string();

    // Initialize subscribers and publishers
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "points", 10, std::bind(&LidarICPNode::pointCloudCallback, this, std::placeholders::_1));
    
    // Initialize publishers with correct topics
    pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("accumulated_points", 10);
    pub_current_ = create_publisher<sensor_msgs::msg::PointCloud2>("current_aligned", 10);

    // Initialize TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Initialize point clouds
    reference_cloud_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    accumulated_cloud_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    reference_to_map_ = Eigen::Matrix4f::Identity();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr LidarICPNode::transformCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const Eigen::Matrix4f& transform) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform);
    return transformed_cloud;
}

std::tuple<bool, Eigen::Matrix4f, std::string> LidarICPNode::performICP(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target) {
    
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    
    // More robust ICP settings
    icp.setMaxCorrespondenceDistance(max_correspondence_distance_);
    icp.setMaximumIterations(max_iterations_);
    icp.setTransformationEpsilon(fitness_epsilon_);
    icp.setEuclideanFitnessEpsilon(0.01);
    icp.setRANSACOutlierRejectionThreshold(0.05);
    icp.setInputSource(source);
    icp.setInputTarget(target);
    
    pcl::PointCloud<pcl::PointXYZ> aligned;
    icp.align(aligned);
    
    double fitness_score = icp.getFitnessScore();
    
    // Detailed logging
    RCLCPP_INFO(get_logger(), "ICP Details:");
    RCLCPP_INFO(get_logger(), "  Converged: %s", icp.hasConverged() ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  Fitness Score: %f", fitness_score);
    
    // Get transformation and log its details
    Eigen::Matrix4f transformation = icp.getFinalTransformation();
    Eigen::Vector3f translation = transformation.block<3, 1>(0, 3);
    Eigen::Vector3f euler = transformation.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
    
    RCLCPP_INFO(get_logger(), "Transform Details:");
    RCLCPP_INFO(get_logger(), "  Translation: [%.3f, %.3f, %.3f] (norm: %.3f)",
                translation.x(), translation.y(), translation.z(), translation.norm());
    RCLCPP_INFO(get_logger(), "  Rotation (rad): [%.3f, %.3f, %.3f] (max: %.3f)",
                euler.x(), euler.y(), euler.z(), euler.array().abs().maxCoeff());

    if (!icp.hasConverged()) {
        return std::make_tuple(false, Eigen::Matrix4f::Identity(), "ICP failed to converge");
    }
    
    if (fitness_score > min_fitness_score_) {
        return std::make_tuple(false, Eigen::Matrix4f::Identity(), 
                             "Poor alignment score: " + std::to_string(fitness_score));
    }
    
    if (!checkTransformValidity(transformation)) {
        return std::make_tuple(false, Eigen::Matrix4f::Identity(), "Invalid transform detected");
    }
    
    return std::make_tuple(true, transformation, 
                          "Success with fitness: " + std::to_string(fitness_score));
}

pcl::PointCloud<pcl::PointXYZ>::Ptr LidarICPNode::filterCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    
    auto filtered = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Voxel grid filter with smaller leaf size for more detail
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    vg.filter(*filtered);
    
    // More aggressive outlier removal
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(filtered);
    sor.setMeanK(50);  // Increased from 30
    sor.setStddevMulThresh(1.0);  // Decreased from 2.0 for stricter filtering
    sor.filter(*filtered);
    
    return filtered;
}

void LidarICPNode::publishTransform(
    const Eigen::Matrix4f& transform,
    const rclcpp::Time& timestamp,
    const std::string& child_frame) {
    
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = timestamp;
    t.header.frame_id = fixed_frame_;
    t.child_frame_id = child_frame;
    
    t.transform.translation.x = transform(0, 3);
    t.transform.translation.y = transform(1, 3);
    t.transform.translation.z = transform(2, 3);
    
    Eigen::Quaternionf q(transform.block<3, 3>(0, 0));
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();
    
    tf_broadcaster_->sendTransform(t);
}

void LidarICPNode::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    callback_count_++;
    RCLCPP_INFO(get_logger(), "\n------ Processing frame %d ------", callback_count_);
    
    // Convert ROS message to PCL cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *current_cloud);
    
    if (current_cloud->empty()) {
        RCLCPP_WARN(get_logger(), "Empty point cloud received");
        return;
    }
    
    // Filter current cloud
    auto current_filtered = filterCloud(current_cloud);
    RCLCPP_INFO(get_logger(), "Filtered cloud size: %lu", current_filtered->size());
    
    // Handle first frame
    if (reference_cloud_->empty()) {
        *reference_cloud_ = *current_filtered;
        *accumulated_cloud_ = *current_filtered;  // Initialize accumulated cloud
        
        // Publish initial clouds and transform
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*accumulated_cloud_, cloud_msg);
        cloud_msg.header = msg->header;
        cloud_msg.header.frame_id = fixed_frame_;
        
        pub_cloud_->publish(cloud_msg);  // Publish accumulated
        pub_current_->publish(cloud_msg); // Publish current
        publishTransform(reference_to_map_, msg->header.stamp, msg->header.frame_id);
        return;
    }
    
    // Perform ICP with reference frame
    auto [success, transform, message] = performICP(current_filtered, reference_cloud_);
    
    if (!success) {
        RCLCPP_WARN(get_logger(), "ICP failed: %s", message.c_str());
        return;
    }
    
    // Update cumulative transform and transform current cloud
    reference_to_map_ = reference_to_map_ * transform.inverse();  // Update cumulative transform
    auto transformed_cloud = transformCloud(current_filtered, reference_to_map_);
    
    // Add transformed points to accumulated cloud
    *accumulated_cloud_ += *transformed_cloud;
    
    // Optional: Downsample accumulated cloud if it gets too large
    if (accumulated_cloud_->size() > 100000) {  // Adjust threshold as needed
        accumulated_cloud_ = filterCloud(accumulated_cloud_);
    }
    
    // Publish accumulated cloud
    sensor_msgs::msg::PointCloud2 accumulated_msg;
    pcl::toROSMsg(*accumulated_cloud_, accumulated_msg);
    accumulated_msg.header = msg->header;
    accumulated_msg.header.frame_id = fixed_frame_;
    pub_cloud_->publish(accumulated_msg);
    
    // Publish current aligned cloud
    sensor_msgs::msg::PointCloud2 current_msg;
    pcl::toROSMsg(*transformed_cloud, current_msg);
    current_msg.header = msg->header;
    current_msg.header.frame_id = fixed_frame_;
    pub_current_->publish(current_msg);
    
    // Publish transform
    publishTransform(reference_to_map_, this->now(), msg->header.frame_id);
    
    // Update reference cloud for next iteration
    *reference_cloud_ = *current_filtered;
    
    RCLCPP_INFO(get_logger(), "Successfully processed frame %d", callback_count_);
    RCLCPP_INFO(get_logger(), "Total accumulated points: %lu", accumulated_cloud_->size());
}

bool LidarICPNode::checkTransformValidity(const Eigen::Matrix4f& transform) {
    Eigen::Vector3f translation = transform.block<3, 1>(0, 3);
    if (translation.norm() > max_translation_) {
        RCLCPP_WARN(get_logger(), "Excessive translation detected: %f", translation.norm());
        return false;
    }
    
    Eigen::Vector3f euler = transform.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
    if (euler.array().abs().maxCoeff() > max_rotation_) {
        RCLCPP_WARN(get_logger(), "Excessive rotation detected: %f", euler.array().abs().maxCoeff());
        return false;
    }
    
    return true;
}
