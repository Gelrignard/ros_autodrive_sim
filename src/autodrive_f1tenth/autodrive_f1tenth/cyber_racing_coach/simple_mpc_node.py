#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as GeomPoint

import numpy as np
import math
import sys
import os

# Add the cyberracingcoach directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .kinematic_mpc import KMPCPlanner, mpc_config, State


class SimpleMPCNode(Node):
    def __init__(self):
        super().__init__('simple_mpc_node')
        
        # ========== Vehicle Parameters ==========
        self.wheel_radius = 0.04  # meters
        
        # ========== State Variables ==========
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.vx = 0.0
        self.yaw_rate = 0.0
        self.steering_angle = 0.0
        self.beta = 0.0  # Slip angle
        
        # ========== Sensor Data Flags ==========
        self.received_ips = False
        self.received_imu = False
        self.received_encoder = False
        
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0
        
        # ========== MPC Configuration ==========
        # Using default parameters from kinematic_mpc.py
        self.mpc_config = mpc_config()
        
        # ========== Create Waypoints ==========
        self.waypoints = self.create_circle_waypoints(
            radius=3.0, 
            num_points=100, 
            target_speed=2.0
        )
        
        # ========== Initialize MPC Planner ==========
        self.get_logger().info("🔧 Initializing MPC planner...")
        self.mpc_planner = KMPCPlanner(
            waypoints=self.waypoints,
            config=self.mpc_config
        )
        self.get_logger().info("✓ MPC planner initialized!")
        
        # ========== ROS2 Subscribers ==========
        self.ips_sub = self.create_subscription(
            Point, 
            '/autodrive/f1tenth_1/ips', 
            self.ips_callback, 
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu, 
            '/autodrive/f1tenth_1/imu', 
            self.imu_callback, 
            10
        )
        
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
        
        # ========== ROS2 Publishers ==========
        self.throttle_pub = self.create_publisher(
            Float32, 
            '/autodrive/f1tenth_1/throttle_command', 
            10
        )
        
        self.steering_pub = self.create_publisher(
            Float32, 
            '/autodrive/f1tenth_1/steering_command', 
            10
        )
        
        # ========== Control Loop Timer ==========
        # 10Hz to match MPC DTK=0.1s
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        # ========== Logging ==========
        self.iteration = 0
        self.get_logger().info("=" * 60)
        self.get_logger().info("🚗 Simple MPC Node Started!")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"📊 MPC Configuration:")
        self.get_logger().info(f"   - Horizon: {self.mpc_config.TK} steps")
        self.get_logger().info(f"   - Time step: {self.mpc_config.DTK} s")
        self.get_logger().info(f"   - Max speed: {self.mpc_config.MAX_SPEED} m/s")
        self.get_logger().info(f"   - Max steering: ±{math.degrees(self.mpc_config.MAX_STEER):.1f}°")
        self.get_logger().info(f"   - Wheelbase: {self.mpc_config.WB} m")
        self.get_logger().info("=" * 60)
        
    def create_circle_waypoints(self, radius=3.0, num_points=100, target_speed=2.0):
        """
        Create circular waypoint trajectory
        
        Args:
            radius: Circle radius in meters
            num_points: Number of waypoints
            target_speed: Target velocity in m/s
            
        Returns:
            numpy.ndarray [4 x N]: [x, y, yaw, speed]
        """
        theta = np.linspace(0, 2*np.pi, num_points)
        cx = radius * np.cos(theta)
        cy = radius * np.sin(theta)
        cyaw = theta + np.pi/2  # Tangent to circle
        cyaw = np.arctan2(np.sin(cyaw), np.cos(cyaw))  # Normalize to [-pi, pi]
        sp = np.ones(num_points) * target_speed
        
        self.get_logger().info(f"📍 Created circular waypoints:")
        self.get_logger().info(f"   - Radius: {radius} m")
        self.get_logger().info(f"   - Points: {num_points}")
        self.get_logger().info(f"   - Target speed: {target_speed} m/s")
        
        return np.array([cx, cy, cyaw, sp])
    
    def load_custom_waypoints(self, filepath):
        """
        Load waypoints from CSV file
        
        Expected format: x, y, yaw, speed (one per line)
        """
        try:
            data = np.loadtxt(filepath, delimiter=',')
            cx = data[:, 0]
            cy = data[:, 1]
            cyaw = data[:, 2]
            sp = data[:, 3]
            
            self.waypoints = np.array([cx, cy, cyaw, sp])
            self.get_logger().info(f"✓ Loaded {len(cx)} waypoints from {filepath}")
            return True
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load waypoints: {str(e)}")
            return False
    
    # ========== Sensor Callbacks ==========
    
    def ips_callback(self, msg):
        """Indoor Positioning System callback"""
        self.x = msg.x
        self.y = msg.y
        
        if not self.received_ips:
            self.get_logger().info("✓ IPS data received!")
            self.received_ips = True
    
    def imu_callback(self, msg):
        """IMU callback"""
        self.yaw_rate = msg.angular_velocity.z
        
        # Dead reckoning for heading (integrate yaw rate)
        self.heading += self.yaw_rate * 0.1  # Assuming 10Hz
        
        # Normalize heading to [-pi, pi]
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        
        if not self.received_imu:
            self.get_logger().info("✓ IMU data received!")
            self.received_imu = True
    
    def left_encoder_callback(self, msg):
        """Left wheel encoder callback"""
        if len(msg.velocity) > 0:
            self.left_wheel_vel = msg.velocity[0]
            self.update_velocity()
            
        if not self.received_encoder:
            self.get_logger().info("✓ Encoder data received!")
            self.received_encoder = True
    
    def right_encoder_callback(self, msg):
        """Right wheel encoder callback"""
        if len(msg.velocity) > 0:
            self.right_wheel_vel = msg.velocity[0]
            self.update_velocity()
    
    def update_velocity(self):
        """Calculate forward velocity from wheel encoders"""
        avg_wheel_vel = (self.left_wheel_vel + self.right_wheel_vel) / 2.0
        self.vx = avg_wheel_vel * self.wheel_radius
    
    # ========== Main Control Loop ==========
    
    def control_loop(self):
        """Main control loop called at 10Hz"""
        self.iteration += 1
        
        # Check if all sensors are ready
        all_sensors_ready = (
            self.received_ips and 
            self.received_imu and 
            self.received_encoder
        )
        
        if not all_sensors_ready:
            if self.iteration % 10 == 0:  # Log every second
                missing = []
                if not self.received_ips: missing.append("IPS")
                if not self.received_imu: missing.append("IMU")
                if not self.received_encoder: missing.append("Encoder")
                self.get_logger().warn(f"⏳ Waiting for sensors: {', '.join(missing)}")
            
            # Send zero commands while waiting
            self.publish_control(0.0, 0.0)
            return
        
        # ========== Create Vehicle State ==========
        vehicle_state = State(
            x=self.x,
            y=self.y,
            delta=self.steering_angle,
            v=self.vx,
            yaw=self.heading,
            yawrate=self.yaw_rate,
            beta=self.beta
        )
        
        # ========== Run MPC Planner ==========
        try:
            # Call the MPC planner
            steering_cmd, speed_cmd = self.mpc_planner.plan(
                states=[
                    vehicle_state.x,
                    vehicle_state.y,
                    vehicle_state.delta,
                    vehicle_state.v,
                    vehicle_state.yaw,
                    vehicle_state.yawrate,
                    vehicle_state.beta
                ],
                waypoints=self.waypoints
            )
            
            # ========== Convert Speed to Throttle ==========
            # Simple proportional controller
            speed_error = speed_cmd - self.vx
            throttle_cmd = np.clip(speed_error * 0.5, -1.0, 1.0)
            
            # ========== Clip Steering Command ==========
            steering_cmd = np.clip(
                steering_cmd, 
                self.mpc_config.MIN_STEER, 
                self.mpc_config.MAX_STEER
            )
            
            # ========== Update Steering Angle Estimate ==========
            self.steering_angle = steering_cmd
            
            # ========== Publish Control Commands ==========
            self.publish_control(steering_cmd, throttle_cmd)
            
            # ========== Logging ==========
            if self.iteration % 10 == 0:  # Log every second
                self.get_logger().info(
                    f"📍 State: x={self.x:.2f}m, y={self.y:.2f}m, "
                    f"v={self.vx:.2f}m/s, ψ={math.degrees(self.heading):.1f}° | "
                    f"🎮 Control: δ={math.degrees(steering_cmd):.1f}°, "
                    f"throttle={throttle_cmd:.2f}"
                )
                
        except Exception as e:
            self.get_logger().error(f"❌ MPC error: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            # Send zero commands on error
            self.publish_control(0.0, 0.0)
    
    def publish_control(self, steering, throttle):
        """
        Publish control commands to vehicle
        
        Args:
            steering: Steering angle in radians
            throttle: Throttle command [-1, 1]
        """
        # Publish steering
        steering_msg = Float32()
        steering_msg.data = float(steering)
        self.steering_pub.publish(steering_msg)
        
        # Publish throttle
        throttle_msg = Float32()
        throttle_msg.data = float(throttle)
        self.throttle_pub.publish(throttle_msg)
    
    def visualize_mpc(self, ref_x, ref_y, pred_x, pred_y, ox, oy):
        """Visualize MPC reference and predicted trajectories"""
        marker_array = MarkerArray()
        
        # Reference trajectory (purple)
        ref_marker = Marker()
        ref_marker.header.frame_id = "map"
        ref_marker.header.stamp = self.get_clock().now().to_msg()
        ref_marker.ns = "reference"
        ref_marker.id = 0
        ref_marker.type = Marker.LINE_STRIP
        ref_marker.action = Marker.ADD
        ref_marker.scale.x = 0.05
        ref_marker.color.r = 0.5
        ref_marker.color.g = 0.0
        ref_marker.color.b = 0.5
        ref_marker.color.a = 1.0
        
        for x, y in zip(ref_x, ref_y):
            p = GeomPoint()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            ref_marker.points.append(p)
        
        marker_array.markers.append(ref_marker)
        
        # Predicted trajectory (green)
        pred_marker = Marker()
        pred_marker.header.frame_id = "map"
        pred_marker.header.stamp = self.get_clock().now().to_msg()
        pred_marker.ns = "predicted"
        pred_marker.id = 1
        pred_marker.type = Marker.LINE_STRIP
        pred_marker.action = Marker.ADD
        pred_marker.scale.x = 0.05
        pred_marker.color.r = 0.0
        pred_marker.color.g = 1.0
        pred_marker.color.b = 0.0
        pred_marker.color.a = 1.0
        
        for x, y in zip(ox, oy):
            p = GeomPoint()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            pred_marker.points.append(p)
        
        marker_array.markers.append(pred_marker)
        
        self.viz_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = SimpleMPCNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
