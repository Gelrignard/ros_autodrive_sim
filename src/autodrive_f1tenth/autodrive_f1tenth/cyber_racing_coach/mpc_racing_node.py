#!/usr/bin/env python3
"""
MPC Racing Node for AutoDRIVE F1TENTH
Uses actual AutoDRIVE simulator topics
"""

import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation

# Standard ROS2 messages
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

from autodrive_f1tenth.cyber_racing_coach.mpc_controller import PathFreeMPC, MPCParameters
from autodrive_f1tenth.cyber_racing_coach.vehicle_model import VehicleParameters
from autodrive_f1tenth.cyber_racing_coach.track_boundary import TrackBoundary


class MPCRacingNode(Node):
    def __init__(self):
        super().__init__('mpc_racing_node')
        
        self.get_logger().info("Initializing MPC Racing Controller...")
        
        # Vehicle namespace
        self.vehicle_ns = '/autodrive/f1tenth_1'
        
        # Load parameters
        vehicle_params = VehicleParameters.get_f1tenth_params()
        mpc_params = MPCParameters.get_default_params()
        
        # Create simple test track
        track = TrackBoundary.create_simple_oval_track()
        
        # Initialize MPC controller
        self.mpc = PathFreeMPC(vehicle_params, mpc_params, track)
        
        # State variables
        self.position = np.zeros(3)  # [x, y, z]
        self.orientation = np.zeros(3)  # [roll, pitch, yaw]
        self.angular_velocity = np.zeros(3)  # [wx, wy, wz]
        self.linear_velocity = 0.0  # Computed from wheel encoders
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0

        # Previous encoder positions and timestamps for differentiation
        self.left_encoder_pos_prev = None
        self.left_encoder_time_prev = None
        self.right_encoder_pos_prev = None
        self.right_encoder_time_prev = None
        
        # Data received flags
        self.received_ips = False
        self.received_imu = False
        self.received_encoders = False
        
        # ROS2 Subscribers
        self.ips_sub = self.create_subscription(
            Point,
            f'{self.vehicle_ns}/ips',
            self.ips_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            f'{self.vehicle_ns}/imu',
            self.imu_callback,
            10
        )
        
        self.left_encoder_sub = self.create_subscription(
            JointState,
            f'{self.vehicle_ns}/left_encoder',
            self.left_encoder_callback,
            10
        )
        
        self.right_encoder_sub = self.create_subscription(
            JointState,
            f'{self.vehicle_ns}/right_encoder',
            self.right_encoder_callback,
            10
        )
        
        # ROS2 Publishers
        self.steering_pub = self.create_publisher(
            Float32,
            f'{self.vehicle_ns}/steering_command',
            10
        )
        
        self.throttle_pub = self.create_publisher(
            Float32,
            f'{self.vehicle_ns}/throttle_command',
            10
        )
        
        # Control loop timer (20Hz to match MPC dt=0.05s)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        # Data received flags
        self.received_ips = False
        self.received_imu = False
        self.received_encoders = False
        
        self.get_logger().info("MPC Racing Controller initialized!")
        self.get_logger().info(f"Listening to: {self.vehicle_ns}")
        
    def ips_callback(self, msg):
        """Indoor Positioning System - provides x, y, z position."""
        self.position = np.array([msg.x, msg.y, msg.z])
        self.received_ips = True
        
    def imu_callback(self, msg):
        """IMU - provides orientation and angular velocity."""
        # Extract orientation (quaternion to euler)
        quat = msg.orientation
        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        self.orientation = rot.as_euler('xyz')  # [roll, pitch, yaw]
        
        # Extract angular velocity
        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        self.received_imu = True
        
    def left_encoder_callback(self, msg):
        """Left wheel encoder - calculate velocity from position."""
        if len(msg.position) == 0:
            return
            
        current_pos = msg.position[0]
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Calculate velocity by differentiating position
        if self.left_encoder_pos_prev is not None and self.left_encoder_time_prev is not None:
            dt = current_time - self.left_encoder_time_prev
            if dt > 0:
                dpos = current_pos - self.left_encoder_pos_prev
                self.left_wheel_vel = dpos / dt
        
        # Store for next iteration
        self.left_encoder_pos_prev = current_pos
        self.left_encoder_time_prev = current_time
        
        # Update vehicle velocity
        self._update_vehicle_velocity()
        
        # Mark as received
        if not self.received_encoders:
            self.get_logger().info("✓ Left encoder data received!")
            self.received_encoders = True
            
    def right_encoder_callback(self, msg):
        """Right wheel encoder - calculate velocity from position."""
        if len(msg.position) == 0:
            return
            
        current_pos = msg.position[0]
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Calculate velocity by differentiating position
        if self.right_encoder_pos_prev is not None and self.right_encoder_time_prev is not None:
            dt = current_time - self.right_encoder_time_prev
            if dt > 0:
                dpos = current_pos - self.right_encoder_pos_prev
                self.right_wheel_vel = dpos / dt
        
        # Store for next iteration
        self.right_encoder_pos_prev = current_pos
        self.right_encoder_time_prev = current_time
        
        # Update vehicle velocity
        self._update_vehicle_velocity()
        
        # Mark as received
        if not self.received_encoders:
            self.get_logger().info("✓ Right encoder data received!")
            self.received_encoders = True
    
    def _update_vehicle_velocity(self):
        """
        Compute vehicle longitudinal velocity from wheel encoders.
        v = (v_left + v_right) / 2 * wheel_radius
        """
        # F1TENTH wheel radius (approximate)
        wheel_radius = 0.05  # meters (TODO: verify this value)
        
        # Average wheel angular velocity to linear velocity
        avg_wheel_vel = (self.left_wheel_vel + self.right_wheel_vel) / 2.0
        self.linear_velocity = avg_wheel_vel * wheel_radius
    
    def get_current_state(self):
        """
        Construct state vector for MPC: [x, y, psi, u, v, r]
        - x, y: position (from IPS)
        - psi: yaw angle (from IMU)
        - u: longitudinal velocity (from encoders)
        - v: lateral velocity (assumed 0 for now)
        - r: yaw rate (from IMU)
        """
        x = self.position[0]
        y = self.position[1]
        psi = self.orientation[2]  # yaw
        u = self.linear_velocity
        v = 0.0  # Lateral velocity (not directly measured, assume 0)
        r = self.angular_velocity[2]  # yaw rate
        
        return np.array([x, y, psi, u, v, r])
    
    def control_loop(self):
        """Main control loop - runs at 20Hz."""
        # Check if we have all necessary data
        if not (self.received_ips and self.received_imu and self.received_encoders):
            missing = []
            if not self.received_ips:
                missing.append("IPS")
            if not self.received_imu:
                missing.append("IMU")
            if not self.received_encoders:
                missing.append("Encoders")
            
            self.get_logger().warn(
                f"Waiting for sensor data: {', '.join(missing)}", 
                throttle_duration_sec=2.0
            )
            return
        
        # Get current state
        current_state = self.get_current_state()
        
        # Solve MPC
        try:
            optimal_control, predicted_traj = self.mpc.solve(current_state)
            
            if optimal_control is not None:
                # Extract control commands
                delta = float(optimal_control[0])  # Steering angle (rad)
                ax = float(optimal_control[1])     # Longitudinal acceleration (m/s^2)
                
                # Publish control commands
                self._publish_control(delta, ax)
                
                # Log performance
                self.get_logger().info(
                    f"State: x={current_state[0]:.2f}, y={current_state[1]:.2f}, "
                    f"ψ={np.rad2deg(current_state[2]):.1f}°, v={current_state[3]:.2f} m/s | "
                    f"Control: δ={np.rad2deg(delta):.1f}°, ax={ax:.2f} m/s²",
                    throttle_duration_sec=0.5
                )
            else:
                self.get_logger().warn("MPC returned no solution, using safe fallback")
                self._publish_control(0.0, -2.0)  # Straight + brake
                
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            self._publish_control(0.0, -2.0)  # Safe fallback
    
    def _publish_control(self, delta, ax):
        """
        Publish control commands to AutoDRIVE simulator.
        
        Args:
            delta: steering angle (rad) - normalized to [-1, 1]
            ax: longitudinal acceleration (m/s^2) - converted to throttle [-1, 1]
        """
        # Normalize steering angle to [-1, 1]
        # Assuming max steering angle is ~30 degrees (0.52 rad)
        max_steering_angle = 0.52  # rad (TODO: verify this value)
        steering_normalized = np.clip(delta / max_steering_angle, -1.0, 1.0)
        
        steering_msg = Float32()
        steering_msg.data = float(steering_normalized)
        self.steering_pub.publish(steering_msg)
        
        # Convert acceleration to throttle [-1, 1]
        throttle_msg = Float32()
        throttle_msg.data = self._accel_to_throttle(ax)
        self.throttle_pub.publish(throttle_msg)
    
    def _accel_to_throttle(self, ax):
        """
        Convert acceleration command to throttle/brake command.
        Throttle range: [-1 (full brake), 1 (full throttle)]
        
        TODO: Calibrate this mapping for AutoDRIVE F1TENTH
        """
        if ax >= 0:
            # Positive acceleration -> throttle
            # Assume max acceleration ~4 m/s^2
            return float(np.clip(ax / 4.0, 0.0, 1.0))
        else:
            # Negative acceleration -> brake
            # Assume max braking ~8 m/s^2
            return float(np.clip(ax / 8.0, -1.0, 0.0))


def main(args=None):
    rclpy.init(args=args)
    node = MPCRacingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down MPC Racing Controller...")
    finally:
        # Send zero commands before shutdown
        zero_msg = Float32()
        zero_msg.data = 0.0
        node.steering_pub.publish(zero_msg)
        node.throttle_pub.publish(zero_msg)
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
