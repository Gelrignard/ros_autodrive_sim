from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the source directory for the model
    src_dir = '/workspace/src/autodrive_f1tenth'
    model_path = os.path.join(src_dir, 'autodrive_f1tenth/model.zip')
    
    # Create a launch argument for the model path
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=model_path,
        description='Path to the pre-trained model file'
    )
    
    # Launch the RL evaluation node
    rl_eval_node = Node(
        package='autodrive_f1tenth',
        executable='autodrive_rl_node',
        name='autodrive_rl',
        output='screen',
        parameters=[{'model_path': LaunchConfiguration('model_path')}]
    )
    
    return LaunchDescription([
        model_path_arg,
        rl_eval_node
    ])