from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('autodrive_f1tenth')
    
    # Default model path
    default_model_path = os.path.join(pkg_dir, 'autodrive_f1tenth/model1004.zip')
    
    # Create a launch argument for the model path
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model_path,
        description='Path to the pre-trained model file'
    )
    
    # Launch the RL evaluation node - fixed executable name
    rl_eval_node = Node(
        package='autodrive_f1tenth',
        executable='rl_eval_node',  # Changed from rl_eval_ros2_node to rl_eval_node
        name='rl_eval',
        output='screen',
        parameters=[{'model_path': LaunchConfiguration('model_path')}],
        remappings=[
            ('odom', '/ego_racecar/odom'),
            ('scan', '/scan'),
            ('cmd_vel', '/ego_racecar/drive')
        ]
    )
    
    return LaunchDescription([
        model_path_arg,
        rl_eval_node
    ])