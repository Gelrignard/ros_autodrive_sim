#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

import numpy as np
import cvxpy as cp
import math
import time

class CBFMPCNode(Node):
    def __init__(self):
        super().__init__('cbf_mpc_node')
        
        # Vehicle parameters
        self.lf = 0.15
        self.lr = 0.15
        self.L = self.lf + self.lr
        self.wheel_radius = 0.06
        
        # MPC parameters
        self.Ts = 0.05  # 20Hz
        
        # State
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.vx = 0.0
        self.yaw_rate = 0.0
        
        # LiDAR data
        self.lidar_ranges = None
        self.lidar_angles = None
        self.min_safe_distance = 0.4  # INCREASED from 0.6m
        self.emergency_distance = 0.35  # Emergency stop threshold
        
        # CBF parameters
        self.cbf_alpha = 10.0  # INCREASED - more conservative
        
        # Data flags
        self.received_ips = False
        self.received_imu = False
        self.received_encoder = False
        self.received_lidar = False
        
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0
        
        # Startup management
        self.start_time = None
        self.warmup_duration = 3.0
        self.is_warmed_up = False
        self.first_control_computed = False
        
        # Target velocity
        self.target_v = 0.0
        self.max_v = 2
        
        # Previous control
        self.prev_throttle = 0.0
        self.prev_steering = 0.0
        
        # Emergency stop flag
        self.emergency_stop = False
        
        # Subscribers
        self.ips_sub = self.create_subscription(
            Point, '/autodrive/f1tenth_1/ips', self.ips_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/autodrive/f1tenth_1/imu', self.imu_callback, 10)
        self.left_encoder_sub = self.create_subscription(
            JointState, '/autodrive/f1tenth_1/left_encoder', self.left_encoder_callback, 10)
        self.right_encoder_sub = self.create_subscription(
            JointState, '/autodrive/f1tenth_1/right_encoder', self.right_encoder_callback, 10)
        self.lidar_sub = self.create_subscription(
            LaserScan, '/autodrive/f1tenth_1/lidar', self.lidar_callback, 10)
        
        # Publishers
        self.throttle_pub = self.create_publisher(
            Float32, '/autodrive/f1tenth_1/throttle_command', 10)
        self.steering_pub = self.create_publisher(
            Float32, '/autodrive/f1tenth_1/steering_command', 10)
        
        # Control timer
        self.control_timer = self.create_timer(self.Ts, self.control_loop)
        
        self.iteration = 0
        self.get_logger().info("🛡️ CBF-MPC Node Started with Reactive Steering!")
        
    def ips_callback(self, msg):
        self.x = msg.x
        self.y = msg.y
        if not self.received_ips:
            self.get_logger().info("✓ IPS data received!")
            self.received_ips = True
    
    def imu_callback(self, msg):
        self.yaw_rate = msg.angular_velocity.z
        self.heading += self.yaw_rate * self.Ts
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        if not self.received_imu:
            self.get_logger().info("✓ IMU data received!")
            self.received_imu = True
    
    def left_encoder_callback(self, msg):
        if len(msg.velocity) > 0:
            self.left_wheel_vel = msg.velocity[0]
            self.update_velocity()
        if not self.received_encoder:
            self.get_logger().info("✓ Encoder data received!")
            self.received_encoder = True
    
    def right_encoder_callback(self, msg):
        if len(msg.velocity) > 0:
            self.right_wheel_vel = msg.velocity[0]
            self.update_velocity()
    
    def lidar_callback(self, msg):
        self.lidar_ranges = np.array(msg.ranges)
        self.lidar_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        
        # Filter out invalid readings
        valid_mask = (self.lidar_ranges > msg.range_min) & (self.lidar_ranges < msg.range_max)
        self.lidar_ranges = np.where(valid_mask, self.lidar_ranges, msg.range_max)
        
        if not self.received_lidar:
            self.get_logger().info(f"✓ LiDAR data received! {len(self.lidar_ranges)} points")
            self.received_lidar = True
    
    def update_velocity(self):
        avg_wheel_vel = (self.left_wheel_vel + self.right_wheel_vel) / 2.0
        self.vx = avg_wheel_vel * self.wheel_radius
    
    def compute_steering_reference(self):
        """
        Compute steering reference using potential field method.
        Steer towards open space, away from obstacles.
        """
        if self.lidar_ranges is None:
            return 0.0
        
        # Consider front 180 degrees
        front_mask = np.abs(self.lidar_angles) < math.pi / 2
        front_angles = self.lidar_angles[front_mask]
        front_ranges = self.lidar_ranges[front_mask]
        
        # Compute "attractiveness" of each direction
        # Higher distance = more attractive
        # Weight by inverse distance to prioritize avoiding close obstacles
        
        weights = np.zeros_like(front_angles)
        
        for i, (angle, distance) in enumerate(zip(front_angles, front_ranges)):
            if distance < 0.5:
                # Very close obstacle - strong repulsion
                weights[i] = -10.0 / (distance + 0.1)
            elif distance < 1.5:
                # Medium distance - mild repulsion
                weights[i] = -2.0 / (distance + 0.1)
            else:
                # Far away - slight attraction
                weights[i] = distance / 10.0
        
        # Compute weighted average angle (desired direction)
        if np.sum(np.abs(weights)) > 0:
            desired_angle = np.average(front_angles, weights=weights)
        else:
            desired_angle = 0.0  # Go straight if no clear direction
        
        # Convert to steering command (proportional control)
        # Positive angle = turn left, negative = turn right
        steering_gain = 2.0
        steering_ref = np.clip(steering_gain * desired_angle, -0.3, 0.3)
        
        return steering_ref
    
    def get_critical_obstacles(self):
        """Get closest obstacles in key sectors."""
        if self.lidar_ranges is None:
            return []
        
        # Divide front into sectors
        sectors = {
            'front': (math.radians(-20), math.radians(20)),
            'front_left': (math.radians(20), math.radians(60)),
            'front_right': (math.radians(-60), math.radians(-20)),
        }
        
        critical_obstacles = []
        
        for sector_name, (angle_min, angle_max) in sectors.items():
            sector_distances = []
            sector_angles = []
            
            for angle, distance in zip(self.lidar_angles, self.lidar_ranges):
                if angle_min <= angle <= angle_max and distance < 5.0:
                    sector_distances.append(distance)
                    sector_angles.append(angle)
            
            if len(sector_distances) > 0:
                min_idx = np.argmin(sector_distances)
                critical_obstacles.append({
                    'angle': sector_angles[min_idx],
                    'distance': sector_distances[min_idx],
                    'sector': sector_name
                })
        
        return critical_obstacles
    
    def compute_cbf_constraint(self, x_state, obstacles):
        """Compute CBF constraint."""
        x_pos, y_pos, heading, vx = x_state
        
        A_list = []
        b_list = []
        
        for obs in obstacles:
            obs_angle = obs['angle']
            distance = obs['distance']
            
            h = distance - self.min_safe_distance
            
            if h < -0.1:
                continue
            
            if h > 2.0:
                continue
            
            approach_rate = vx * math.cos(obs_angle)
            
            if approach_rate < 0:
                continue
            
            Lf_h = -approach_rate
            
            # Control influence: both steering and throttle affect safety
            # Steering away from obstacle helps
            # Throttle increases approach rate
            
            # If obstacle is on left (angle > 0), steering right (negative) helps
            # If obstacle is on right (angle < 0), steering left (positive) helps
            Lg_h_steering = vx * math.sin(obs_angle) * 2.0  # Steering helps avoid
            Lg_h_throttle = -math.cos(obs_angle)  # Throttle increases danger
            
            A_row = np.array([Lg_h_steering, Lg_h_throttle])
            b_val = -(Lf_h + self.cbf_alpha * h)
            
            A_list.append(A_row)
            b_list.append(b_val)
        
        if len(A_list) == 0:
            return None, None
        
        return np.array(A_list), np.array(b_list)
    
    def solve_cbf_qp(self, x_state, obstacles, steering_ref):
        """Solve CBF-QP for safe control."""
        try:
            # Reference control
            v_error = self.target_v - x_state[3]
            throttle_ref = np.clip(v_error * 1.5, -0.5, 0.8)
            
            # Use computed steering reference (NOT zero!)
            u_ref = np.array([steering_ref, throttle_ref])
            
            # Decision variables
            u = cp.Variable(2)
            
            # Objective
            tracking_cost = cp.sum_squares(u - u_ref)
            
            if self.first_control_computed:
                smoothness_cost = cp.sum_squares(u - np.array([self.prev_steering, self.prev_throttle]))
                cost = tracking_cost + 0.3 * smoothness_cost  # Reduced smoothness weight
            else:
                cost = tracking_cost
            
            # Constraints
            constraints = []
            
            # Control limits
            constraints.append(u[0] >= -0.3)
            constraints.append(u[0] <= 0.3)
            constraints.append(u[1] >= -1.0)
            constraints.append(u[1] <= 1.0)
            
            # Rate limits
            if self.first_control_computed:
                constraints.append(u[0] >= self.prev_steering - 0.15)  # Allow faster steering
                constraints.append(u[0] <= self.prev_steering + 0.15)
                constraints.append(u[1] >= self.prev_throttle - 0.4)
                constraints.append(u[1] <= self.prev_throttle + 0.4)
            
            # CBF constraints
            A_cbf, b_cbf = self.compute_cbf_constraint(x_state, obstacles)
            
            cbf_active = False
            if A_cbf is not None and len(A_cbf) > 0:
                for i in range(len(A_cbf)):
                    constraints.append(A_cbf[i] @ u >= b_cbf[i])
                cbf_active = True
                
                if self.iteration % 20 == 0:
                    self.get_logger().info(f"🛡️ {len(A_cbf)} CBF constraints active")
            
            # Solve QP
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                self.get_logger().warn(f"⚠️ QP status: {problem.status}")
                return None
            
            u_opt = u.value
            
            if cbf_active and self.iteration % 20 == 0:
                deviation = np.linalg.norm(u_opt - u_ref)
                if deviation > 0.1:
                    self.get_logger().info(f"🛡️ CBF intervention! Deviation: {deviation:.3f}")
            
            return u_opt
            
        except Exception as e:
            self.get_logger().error(f"CBF-QP error: {str(e)}")
            return None
    
    def control_loop(self):
        self.iteration += 1
        
        all_sensors_ready = (self.received_ips and self.received_imu and 
                            self.received_encoder and self.received_lidar)
        
        if not all_sensors_ready:
            if self.iteration % 20 == 0:
                missing = []
                if not self.received_ips: missing.append("IPS")
                if not self.received_imu: missing.append("IMU")
                if not self.received_encoder: missing.append("Encoder")
                if not self.received_lidar: missing.append("LiDAR")
                self.get_logger().warn(f"⏳ Waiting for: {', '.join(missing)}")
            
            self.publish_control(0.0, 0.0)
            return
        
        # Initialize start time
        if self.start_time is None:
            self.start_time = time.time()
            self.get_logger().info("🚀 All sensors ready! Starting warmup...")
        
        # Get critical obstacles
        obstacles = self.get_critical_obstacles()
        
        # EMERGENCY STOP CHECK
        if len(obstacles) > 0:
            min_dist = min([obs['distance'] for obs in obstacles])
            
            if min_dist < self.emergency_distance:
                if not self.emergency_stop:
                    self.get_logger().error(f"🚨 EMERGENCY STOP! Distance: {min_dist:.2f}m")
                    self.emergency_stop = True
                
                self.publish_control(0.0, -1.0)
                return
            else:
                if self.emergency_stop:
                    self.get_logger().info("✅ Emergency cleared, resuming...")
                    self.emergency_stop = False
        
        # Warmup phase
        elapsed = time.time() - self.start_time
        
        if elapsed < self.warmup_duration:
            self.target_v = (elapsed / self.warmup_duration) * self.max_v
            if self.iteration % 20 == 0:
                self.get_logger().info(f"🔥 Warmup: {elapsed:.1f}s, target_v={self.target_v:.2f} m/s")
        else:
            if not self.is_warmed_up:
                self.is_warmed_up = True
                self.get_logger().info("✅ Warmup complete!")
            self.target_v = self.max_v
        
        # Current state
        x_state = np.array([self.x, self.y, self.heading, self.vx])
        
        # Compute steering reference (REACTIVE AVOIDANCE)
        steering_ref = self.compute_steering_reference()
        
        # Log obstacles and steering
        if self.iteration % 20 == 0:
            if len(obstacles) > 0:
                for obs in obstacles:
                    self.get_logger().info(
                        f"📏 {obs['sector']}: {obs['distance']:.2f}m @ {math.degrees(obs['angle']):.1f}°"
                    )
            self.get_logger().info(f"🎯 Steering reference: {steering_ref:.3f} rad ({math.degrees(steering_ref):.1f}°)")
        
        # Solve CBF-QP
        u_opt = self.solve_cbf_qp(x_state, obstacles, steering_ref)
        
        if u_opt is None:
            self.get_logger().error("❌ QP failed! Emergency stop.")
            self.publish_control(0.0, -1.0)
            return
        
        steering_cmd = float(np.clip(u_opt[0], -0.3, 0.3))
        throttle_cmd = float(np.clip(u_opt[1], -1.0, 1.0))
        
        # Update previous control
        self.prev_steering = steering_cmd
        self.prev_throttle = throttle_cmd
        
        if not self.first_control_computed:
            self.first_control_computed = True
            self.get_logger().info("✅ First control computed!")
        
        # Publish
        self.publish_control(steering_cmd, throttle_cmd)
        
        if self.iteration % 20 == 0:
            self.get_logger().info(
                f"State: x={self.x:.2f}, y={self.y:.2f}, v={self.vx:.2f} (target={self.target_v:.2f}) | "
                f"Control: δ={steering_cmd:.3f}, a={throttle_cmd:.2f}"
            )
    
    def publish_control(self, steering, throttle):
        """Publish control commands."""
        steering_msg = Float32()
        steering_msg.data = steering
        self.steering_pub.publish(steering_msg)
        
        throttle_msg = Float32()
        throttle_msg.data = throttle
        self.throttle_pub.publish(throttle_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CBFMPCNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
