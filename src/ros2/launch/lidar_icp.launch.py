from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lidar_icp',
            executable='lidar_icp_node',
            name='lidar_icp_node',
            parameters=[{
                'max_correspondence_distance': 10.0, 
                'max_iterations': 100,              
                'fitness_epsilon': 1e-6,          
                'leaf_size': 0.2,                  
                'min_fitness_score': 20.0,          
                'max_translation': 30.0,           
                'max_rotation': 3.15,              
                'fixed_frame': 'map'
            }],
            remappings=[
                ('points', '/os_cloud_node/points'),
                ('accumulated_points', '/accumulated_points'), 
                ('current_aligned', '/current_aligned')       
            ],
            output='screen'
        )
    ])
