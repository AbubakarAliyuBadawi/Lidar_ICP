#include "lidar_icp/lidar_icp_node.hpp"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarICPNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
