#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import block_diag
import math

class SimpleMPCNode(Node):
    def __init__(self):
        super().__init__('simple_mpc_node')
        
        # Vehicle parameters
        self.m = 3.47
        self.lf = 0.15875
        self.lr = 0.17145
        self.L = self.lf + self.lr
        self.Iz = 0.04712
        self.wheel_radius = 0.04
        
        # MPC parameters
        self.N = 10  # Longer horizon
        self.Ts = 0.05
        self.nx = 6
        self.nu = 2
        
        # Cost matrices - ADJUSTED
        self.Q = np.diag([50.0, 50.0, 5.0, 20.0, 1.0, 1.0])  # Higher tracking cost
        self.QN = self.Q * 2  # Terminal cost higher
        self.R = np.diag([5.0, 0.5])  # LOWER control cost
        
        # Control limits
        self.steering_min = -0.34906585
        self.steering_max = 0.34906585
        self.throttle_min = -1.5
        self.throttle_max = 5.0  # Reduced max throttle
        
        # State
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0
        
        # Previous control
        self.u_prev = np.array([0.0, 0.0])
        
        # Data flags
        self.received_ips = False
        self.received_imu = False
        self.received_left_encoder = False
        self.received_right_encoder = False
        
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0
        
        # Subscribers
        self.ips_sub = self.create_subscription(Point, '/autodrive/f1tenth_1/ips', self.ips_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/autodrive/f1tenth_1/imu', self.imu_callback, 10)
        self.left_encoder_sub = self.create_subscription(JointState, '/autodrive/f1tenth_1/left_encoder', self.left_encoder_callback, 10)
        self.right_encoder_sub = self.create_subscription(JointState, '/autodrive/f1tenth_1/right_encoder', self.right_encoder_callback, 10)
        
        # Publishers
        self.throttle_pub = self.create_publisher(Float32, '/autodrive/f1tenth_1/throttle_command', 10)
        self.steering_pub = self.create_publisher(Float32, '/autodrive/f1tenth_1/steering_command', 10)
        
        # Control timer
        self.control_timer = self.create_timer(self.Ts, self.control_loop)
        
        # Target velocity
        self.ref_v = 3.0
        
        self.iteration = 0
        self.get_logger().info("🚀 Simple MPC Node Started!")
        
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
        if not self.received_left_encoder:
            self.get_logger().info("✓ Left encoder data received!")
            self.received_left_encoder = True
    
    def right_encoder_callback(self, msg):
        if len(msg.velocity) > 0:
            self.right_wheel_vel = msg.velocity[0]
            self.update_velocity()
        if not self.received_right_encoder:
            self.get_logger().info("✓ Right encoder data received!")
            self.received_right_encoder = True
    
    def update_velocity(self):
        avg_wheel_vel = (self.left_wheel_vel + self.right_wheel_vel) / 2.0
        self.vx = avg_wheel_vel * self.wheel_radius
        self.vy = 0.0
    
    def kinematic_bicycle_model(self, x, u):
        """Kinematic bicycle model."""
        pos_x, pos_y, heading, vx, vy, yaw_rate = x
        steering, throttle = u
        
        vx_safe = max(abs(vx), 0.1)
        beta = math.atan(self.lr / self.L * math.tan(steering))
        
        x_dot = vx_safe * math.cos(heading + beta)
        y_dot = vx_safe * math.sin(heading + beta)
        heading_dot = vx_safe / self.lr * math.sin(beta)
        vx_dot = throttle
        vy_dot = 0.0
        yaw_rate_dot = heading_dot
        
        return np.array([x_dot, y_dot, heading_dot, vx_dot, vy_dot, yaw_rate_dot])
    
    def discretize_dynamics(self, x, u):
        """Euler discretization."""
        x_dot = self.kinematic_bicycle_model(x, u)
        x_next = x + self.Ts * x_dot
        return x_next
    
    def linearize_dynamics(self, x, u):
        """Numerical linearization."""
        eps = 1e-5
        
        A = np.zeros((self.nx, self.nx))
        f0 = self.kinematic_bicycle_model(x, u)
        
        for i in range(self.nx):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = self.kinematic_bicycle_model(x_plus, u)
            A[:, i] = (f_plus - f0) / eps
        
        B = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            u_plus = u.copy()
            u_plus[i] += eps
            f_plus = self.kinematic_bicycle_model(x, u_plus)
            B[:, i] = (f_plus - f0) / eps
        
        Ad = np.eye(self.nx) + self.Ts * A
        Bd = self.Ts * B
        
        return Ad, Bd
    
    def generate_reference_trajectory(self):
        """
        Generate reference trajectory AHEAD of vehicle.
        """
        x_ref = np.zeros((self.nx, self.N + 1))
        u_ref = np.zeros((self.nu, self.N))
        
        # Velocity error
        v_error = self.ref_v - self.vx
        
        for k in range(self.N + 1):
            # Project forward along current heading
            x_ref[0, k] = self.x + self.ref_v * math.cos(self.heading) * k * self.Ts
            x_ref[1, k] = self.y + self.ref_v * math.sin(self.heading) * k * self.Ts
            x_ref[2, k] = self.heading
            x_ref[3, k] = self.ref_v
            x_ref[4, k] = 0.0
            x_ref[5, k] = 0.0
        
        for k in range(self.N):
            u_ref[0, k] = 0.0  # Straight
            # Reference throttle to reach target velocity
            u_ref[1, k] = np.clip(v_error * 2.0, -1.0, 2.0)
        
        return x_ref, u_ref
    
    def solve_mpc(self, x0, x_ref, u_ref):
        """Solve MPC using OSQP."""
        try:
            # Linearize along reference
            Ad_seq = []
            Bd = None
            
            for k in range(self.N):
                Ad, Bd_k = self.linearize_dynamics(x_ref[:, k], u_ref[:, k])
                Ad_seq.append(Ad)
                if Bd is None:
                    Bd = Bd_k
            
            # Build QP - SIMPLIFIED (no slack variables for now)
            # Quadratic cost
            P = sparse.block_diag([
                sparse.kron(sparse.eye(self.N), self.Q),
                self.QN,
                sparse.kron(sparse.eye(self.N), self.R)
            ], format='csc')
            
            # Linear cost
            q_x = np.hstack([self.Q @ x_ref[:, k] for k in range(self.N)])
            q_x = np.hstack([q_x, self.QN @ x_ref[:, self.N]])
            q_u = np.hstack([self.R @ u_ref[:, k] for k in range(self.N)])
            q = np.hstack([-q_x, -q_u])
            
            # Dynamics constraints
            diag_As = block_diag(*Ad_seq)
            G = np.zeros([self.nx * (self.N + 1), self.nx * (self.N + 1)])
            G[self.nx:diag_As.shape[0] + self.nx, :diag_As.shape[1]] = diag_As
            K = np.kron(np.eye(self.N + 1), -np.eye(self.nx))
            Ax = G + K
            Ax = sparse.csc_matrix(Ax)
            
            Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), Bd)
            
            Aeq = sparse.hstack([Ax, Bu])
            
            # Initial condition + linearization error
            stacking_list = []
            for k in range(self.N):
                # Correct linearization error term
                f_nonlinear = self.discretize_dynamics(x_ref[:, k], u_ref[:, k])
                f_linear = Ad_seq[k] @ x_ref[:, k] + Bd @ u_ref[:, k]
                linearize_error = f_nonlinear - f_linear
                stacking_list.append(linearize_error)
            
            linearize_constraint = np.array(stacking_list).reshape(-1)
            
            leq = np.hstack([-x0, -linearize_constraint])
            ueq = leq
            
            # Control constraints
            Aineq = sparse.hstack([
                sparse.csc_matrix((self.N * self.nu, (self.N + 1) * self.nx)),
                sparse.eye(self.N * self.nu)
            ])
            lineq = np.tile([self.steering_min, self.throttle_min], self.N)
            uineq = np.tile([self.steering_max, self.throttle_max], self.N)
            
            # Combine
            A = sparse.vstack([Aeq, Aineq], format='csc')
            l = np.hstack([leq, lineq])
            u = np.hstack([ueq, uineq])
            
            # Solve
            prob = osqp.OSQP()
            prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
            res = prob.solve()
            
            if res.info.status != 'solved':
                self.get_logger().warn(f"OSQP failed: {res.info.status}")
                return None
            
            # Extract first control
            u_opt = res.x[(self.N + 1) * self.nx:(self.N + 1) * self.nx + self.nu]
            
            # Debug logging
            if self.iteration % 20 == 0:
                tracking_error = np.linalg.norm(x_ref[:, 0] - x0)
                self.get_logger().info(f"Tracking error: {tracking_error:.3f}, u_opt: [{u_opt[0]:.3f}, {u_opt[1]:.3f}]")
            
            return u_opt
            
        except Exception as e:
            self.get_logger().error(f"MPC error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def control_loop(self):
        self.iteration += 1
        
        all_sensors_ready = (self.received_ips and self.received_imu and 
                            self.received_left_encoder and self.received_right_encoder)
        
        if not all_sensors_ready:
            if self.iteration % 20 == 0:
                missing = []
                if not self.received_ips: missing.append("IPS")
                if not self.received_imu: missing.append("IMU")
                if not self.received_left_encoder: missing.append("Left Encoder")
                if not self.received_right_encoder: missing.append("Right Encoder")
                self.get_logger().warn(f"Waiting for: {', '.join(missing)}")
            return
        
        # Current state
        x0 = np.array([self.x, self.y, self.heading, self.vx, self.vy, self.yaw_rate])
        
        # Generate reference
        x_ref, u_ref = self.generate_reference_trajectory()
        
        # Solve MPC
        u_opt = self.solve_mpc(x0, x_ref, u_ref)
        
        if u_opt is None:
            # Fallback: simple P control
            steering_cmd = 0.0
            v_error = self.ref_v - self.vx
            throttle_cmd = np.clip(v_error * 2.0, -1.0, 2.0)
        else:
            steering_cmd = float(np.clip(u_opt[0], self.steering_min, self.steering_max))
            throttle_cmd = float(np.clip(u_opt[1], self.throttle_min, self.throttle_max))
        
        # Publish
        steering_msg = Float32()
        steering_msg.data = steering_cmd
        self.steering_pub.publish(steering_msg)
        
        throttle_msg = Float32()
        throttle_msg.data = throttle_cmd
        self.throttle_pub.publish(throttle_msg)
        
        self.u_prev = np.array([steering_cmd, throttle_cmd])
        
        if self.iteration % 20 == 0:
            self.get_logger().info(
                f"State: x={self.x:.2f}, y={self.y:.2f}, v={self.vx:.2f}, θ={self.heading:.2f} | "
                f"Ref_v={self.ref_v:.2f} | Control: δ={steering_cmd:.3f}, a={throttle_cmd:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = SimpleMPCNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
