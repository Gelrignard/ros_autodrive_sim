#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import zipfile
import io
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from sensor_msgs.msg import Imu, LaserScan, JointState
import os
import math
from tf_transformations import euler_from_quaternion

class SB3PolicyNetwork(nn.Module):
    """A neural network matching the Stable Baselines 3 PPO structure"""
    def __init__(self, state_dict, obs_dim=46, hidden_dim=64, action_dim=2):  # Changed from 43 to 46
        super(SB3PolicyNetwork, self).__init__()
        
        # Create architecture matching SB3 structure
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.action_net = nn.Linear(hidden_dim, action_dim)
        
        # Load the state dict (just the policy parts)
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if 'value_net' not in key:  # Skip value network components
                filtered_state_dict[key] = value
        
        # We need to map some keys to match our simplified architecture
        new_state_dict = {}
        key_mapping = {
            'mlp_extractor.policy_net.0.weight': 'policy_net.0.weight',
            'mlp_extractor.policy_net.0.bias': 'policy_net.0.bias',
            'mlp_extractor.policy_net.2.weight': 'policy_net.2.weight',
            'mlp_extractor.policy_net.2.bias': 'policy_net.2.bias',
            'action_net.weight': 'action_net.weight',
            'action_net.bias': 'action_net.bias'
        }
        
        for old_key, new_key in key_mapping.items():
            if old_key in state_dict:
                new_state_dict[new_key] = state_dict[old_key]
        
        # Load the mapped state dict
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
    def forward(self, x):
        features = self.policy_net(x)
        actions = self.action_net(features)
        return actions, None

class CustomPPO:
    """Simple wrapper to mimic Stable-Baselines3 PPO without importing it"""
    def __init__(self, policy_state_dict=None, policy_network=None):
        if policy_network is not None:
            # Use the provided policy network
            self.policy = policy_network
        else:
            # Create a policy network from state dict
            self.policy = SB3PolicyNetwork(policy_state_dict)
            # Set to evaluation mode
            self.policy.eval()
        
    def predict(self, observation, deterministic=True):
        # Convert to tensor
        obs = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            # Forward pass through policy network
            action, _ = self.policy(obs)
        return action.squeeze().numpy(), None
    
    @classmethod
    def load(cls, path, device='cpu'):
        # Handle path with or without .zip extension
        if not path.endswith('.zip'):
            path += '.zip'
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Extract policy network from zip
        with zipfile.ZipFile(path, 'r') as z:
            with z.open('policy.pth') as f:
                buffer = io.BytesIO(f.read())
                policy_state_dict = torch.load(buffer, map_location=device)
        
        return cls(policy_state_dict=policy_state_dict)

class AutodriveRLNode(Node):
    def __init__(self):
        super().__init__('autodrive_rl_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '')
        
        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if not model_path:
            model_path = '/workspace/src/autodrive_f1tenth/autodrive_f1tenth/model.zip'
            
        self.get_logger().info(f"Loading model from: {model_path}")
        
        # Initialize message storage
        self.position = None
        self.orientation = None
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.lidar_ranges = None
        self.steering_delta = 0.0
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0
        
        try:
            # Load the model
            self.model = CustomPPO.load(model_path)
            self.get_logger().info("Model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return
        
        # Subscribe to AutoDRIVE topics
        self.ips_sub = self.create_subscription(
            Point,
            '/autodrive/f1tenth_1/ips',
            self.ips_callback,
            10)
            
        self.imu_sub = self.create_subscription(
            Imu,
            '/autodrive/f1tenth_1/imu',
            self.imu_callback,
            10)
            
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/autodrive/f1tenth_1/lidar',
            self.lidar_callback,
            10)
        
        self.left_encoder_sub = self.create_subscription(
            JointState, 
            '/autodrive/f1tenth_1/left_encoder', 
            self.left_encoder_callback, 
            10
        )
        
        self.right_encoder_sub = self.create_subscription(
            JointState, 
            '/autodrive/f1tenth_1/right_encoder', 
            self.right_encoder_callback, 
            10
        )
        
        # Publishers for the actions
        self.throttle_pub = self.create_publisher(
            Float32,
            '/autodrive/f1tenth_1/throttle_command',
            10)
            
        self.steering_pub = self.create_publisher(
            Float32,
            '/autodrive/f1tenth_1/steering_command',
            10)
            
        # Create a timer for prediction at a fixed rate
        self.timer = self.create_timer(0.1, self.prediction_timer_callback)  # 10Hz
        
        # Data received flags
        self.received_ips = False
        self.received_imu = False
        self.received_lidar = False
        self.received_encoders = False
        
        self.get_logger().info("AutoDRIVE RL Evaluation Node is ready")

    def ips_callback(self, msg):
        self.position = [msg.x, msg.y, msg.z]
        if not self.received_ips:
            self.get_logger().info("✓ IPS data received!")
            self.received_ips = True
        
    def imu_callback(self, msg):
        # Extract orientation and angular velocity
        self.orientation = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]
        self.angular_velocity = msg.angular_velocity.z
        
        if not self.received_imu:
            self.get_logger().info("✓ IMU data received!")
            self.received_imu = True
            
    def lidar_callback(self, msg):
        self.lidar_ranges = np.array(msg.ranges)
        
        if not self.received_lidar:
            self.get_logger().info("✓ LiDAR data received!")
            self.received_lidar = True
    
    def left_encoder_callback(self, msg):
        if len(msg.velocity) > 0:
            self.left_wheel_vel = msg.velocity[0]
            self.update_velocity()
            
        if not self.received_encoders:
            self.get_logger().info("✓ Encoder data received!")
            self.received_encoders = True
    
    def right_encoder_callback(self, msg):
        if len(msg.velocity) > 0:
            self.right_wheel_vel = msg.velocity[0]
            self.update_velocity()
    
    def update_velocity(self):
        """Calculate forward velocity from wheel encoders"""
        wheel_radius = 0.324  # meters (approximate)
        self.velocity = (self.left_wheel_vel + self.right_wheel_vel) / 2.0 * wheel_radius

    def prediction_timer_callback(self):
        # Check if all necessary data is available
        if not all([self.received_ips, self.received_imu, self.received_lidar]):
            if not self.received_ips:
                self.get_logger().info("Waiting for IPS data...")
            if not self.received_imu:
                self.get_logger().info("Waiting for IMU data...")
            if not self.received_lidar:
                self.get_logger().info("Waiting for LiDAR data...")
            return
            
        try:
            # Create observation from sensor data
            observation = self.create_observation()
            
            # Predict the action using the loaded model
            action, _states = self.model.predict(observation, deterministic=True)
            
            # Convert action to control commands
            steering_cmd = float(action[0])  # Assuming range [-1, 1]
            throttle_cmd = float(action[1])  # Assuming range [0, 1]
            
            # Scale throttle to appropriate range (optional)
            # throttle_cmd = max(0.0, min(1.0, (throttle_cmd + 1.0) / 2.0))  # Convert from [-1,1] to [0,1]
            
            # Create messages
            steering_msg = Float32()
            steering_msg.data = steering_cmd
            
            throttle_msg = Float32()
            throttle_msg.data = throttle_cmd
            
            # Publish the actions
            self.steering_pub.publish(steering_msg)
            self.throttle_pub.publish(throttle_msg)
            
            # Update steering delta for next observation
            self.steering_delta = steering_cmd
            
            self.get_logger().info(f"Published action: steering={steering_cmd:.2f}, throttle={throttle_cmd:.2f}")
            
        except Exception as e:
            self.get_logger().error(f"Error in prediction: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def create_observation(self):
        """Create observation vector from sensor data"""
        # Extract yaw from quaternion
        _, _, yaw = euler_from_quaternion(self.orientation)
        
        # Process LiDAR data - downsample to 36 beams if needed
        scan_ranges = self.lidar_ranges
        num_beams = len(scan_ranges)
        
        if num_beams != 36:
            # Downsample to get 36 beams
            indices = np.linspace(0, num_beams-1, 36).astype(int)
            scan_ranges = scan_ranges[indices]
        
        # Handle inf and nan values
        scan_ranges = np.where(np.isnan(scan_ranges), 100.0, scan_ranges)
        scan_ranges = np.where(np.isinf(scan_ranges) & (scan_ranges > 0), 100.0, scan_ranges)
        scan_ranges = np.where(np.isinf(scan_ranges) & (scan_ranges < 0), 0.0, scan_ranges)


        # bicycle model: X_dot = v * cos(theta), Y_dot = v * sin(theta), theta_dot = (v / L) * tan(delta)
        L = 0.33  # wheelbase in meters
        linear_vel_x = self.velocity * math.cos(yaw)
        linear_vel_y = self.velocity * math.sin(yaw)
        
        # Create observation vector in the format expected by the model (46 dimensions)
        observation = np.concatenate([
            [0],                         # ego_idx (always 0 for single agent)
            scan_ranges,                 # 36 LiDAR readings
            [self.position[0]],          # poses_x
            [self.position[1]],          # poses_y
            [yaw],                       # poses_theta
            [linear_vel_x],              # vels_x
            [linear_vel_y],              # vels_y
            [self.angular_velocity],     # ang_vels_z
            [0.0],                       # collisions (assume no collision)
            [0.0],                       # lap_times (not tracking)
            [0.0]                        # lap_counts (not tracking)
        ])
        
        return observation

def main(args=None):
    rclpy.init(args=args)
    autodrive_rl_node = AutodriveRLNode()
    
    try:
        rclpy.spin(autodrive_rl_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        autodrive_rl_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()