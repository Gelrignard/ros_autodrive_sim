#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import os
import sys
import numpy as np
print(f"Running with NumPy version: {np.__version__}")
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
import math
from tf_transformations import euler_from_quaternion

class RLEvalNode(Node):
    def __init__(self):
        super().__init__('rl_eval_node')
        
        # First try direct path to the model in the source directory
        source_model_path = os.path.join(os.path.dirname(__file__), 'model1004.zip')
        
        if os.path.exists(source_model_path):
            model_path = source_model_path
            self.get_logger().info(f"Found model in source directory: {model_path}")
        else:
            # Get the package directory to locate the model file
            pkg_dir = get_package_share_directory('autodrive_f1tenth')
            model_path = os.path.join(pkg_dir, 'autodrive_f1tenth/model1004.zip')
            self.get_logger().info(f"Looking for model in install directory: {model_path}")
        
        # Declare parameters
        self.declare_parameter('model_path', model_path)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.get_logger().info(f"Loading model from: {self.model_path}")
        
        # Initialize message storage
        self.latest_odom = None
        self.latest_scan = None
        self.steering_delta = 0.0
        
        try:
            # Try importing stable_baselines3 here to catch import errors
            try:
                from stable_baselines3 import PPO
                # Load the pre-trained model - be explicit about file existence
                if os.path.exists(self.model_path):
                    self.model = PPO.load(self.model_path)
                    self.get_logger().info("Model loaded successfully!")
                elif os.path.exists(self.model_path + ".zip"):
                    # Try with .zip extension if the file wasn't found
                    self.model = PPO.load(self.model_path + ".zip")
                    self.get_logger().info("Model loaded successfully with .zip extension!")
                else:
                    self.get_logger().error(f"Model file not found: {self.model_path}")
                    return
            except ImportError as e:
                self.get_logger().error(f"Error importing stable_baselines3: {str(e)}")
                self.get_logger().error("Try downgrading NumPy: pip install numpy==1.23.5")
                return
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            return
            
        # Subscribe to standard ROS topics
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
            
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10)
        
        # Publisher for the actions
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10)
            
        # Create a timer for prediction at a fixed rate
        self.timer = self.create_timer(0.05, self.prediction_timer_callback)  # 20Hz
            
        self.get_logger().info("RL Evaluation Node is ready")


    def odom_callback(self, msg):
        self.latest_odom = msg
        # Extract steering angle from odometry if available
        # In a real system, you might need to get this from another source
        
    def scan_callback(self, msg):
        self.latest_scan = msg

    def prediction_timer_callback(self):
        if not self.latest_odom or not self.latest_scan:
            self.get_logger().info("Waiting for odometry and scan data...")
            return
            
        try:
            # Create observation from ROS messages
            observation = self.create_observation_from_ros_msgs()
            
            # Predict the action using the loaded model
            action, _states = self.model.predict(observation, deterministic=True)
            
            # Convert action to ROS Twist message
            twist_msg = Twist()
            twist_msg.linear.x = action[1]  # Speed
            # Convert steering angle to angular velocity
            # This is a simplified conversion and might need adjustment for your vehicle
            twist_msg.angular.z = action[0] * 2.0  # Steering angle to angular velocity
            
            # Publish the action
            self.cmd_vel_pub.publish(twist_msg)
            
            # Update steering delta for next observation
            self.steering_delta = action[0]
            
            self.get_logger().debug(f"Published action: steering={action[0]}, speed={action[1]}")
            
        except Exception as e:
            self.get_logger().error(f"Error in prediction: {str(e)}")

    def create_observation_from_ros_msgs(self):
        """
        Create observation vector from ROS messages
        """
        # Extract pose data from odometry
        pose = self.latest_odom.pose.pose
        twist = self.latest_odom.twist.twist
        
        # Convert quaternion to euler angles
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        )
        _, _, yaw = euler_from_quaternion(quaternion)
        
        # Extract scan data and ensure it has 36 points
        # You may need to downsample or interpolate the scan data
        scan_ranges = np.array(self.latest_scan.ranges)
        num_beams = len(scan_ranges)
        
        if num_beams != 36:
            # Downsample or interpolate to get 36 beams
            indices = np.linspace(0, num_beams-1, 36).astype(int)
            scan_ranges = scan_ranges[indices]
        
        # Handle inf and nan values
        scan_ranges = np.nan_to_num(scan_ranges, nan=100.0, posinf=100.0, neginf=0.0)
        
        # Create observation vector in the format expected by the model
        observation = np.concatenate([
            [pose.position.x],
            [pose.position.y],
            [yaw],
            [twist.linear.x],
            [twist.linear.y],
            [twist.angular.z],
            [self.steering_delta],  # Using most recent steering command
            scan_ranges
        ])
        
        return observation

def main(args=None):
    rclpy.init(args=args)
    rl_eval_node = RLEvalNode()
    
    try:
        rclpy.spin(rl_eval_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        rl_eval_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()