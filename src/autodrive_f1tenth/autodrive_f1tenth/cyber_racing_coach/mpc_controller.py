import numpy as np
import casadi as ca
from .vehicle_model import BicycleModel3DOF
from .track_boundary import TrackBoundary

class PathFreeMPC:
    def __init__(self, vehicle_params, mpc_params, track_boundary):
        """
        Initialize path-free MPC controller.
        
        Args:
            vehicle_params: vehicle parameters dict
            mpc_params: MPC configuration dict
            track_boundary: TrackBoundary object
        """
        self.vehicle = BicycleModel3DOF(vehicle_params)
        self.track = track_boundary
        
        # MPC parameters
        self.N = mpc_params['horizon']           # Prediction horizon
        self.dt = mpc_params['dt']               # Time step (s)
        self.Q_progress = mpc_params['Q_progress']  # Progress weight
        self.Q_boundary = mpc_params['Q_boundary']  # Boundary weight
        self.R_delta = mpc_params['R_delta']     # Steering effort weight
        self.R_ax = mpc_params['R_ax']           # Accel effort weight
        
        # Control limits
        self.delta_max = mpc_params['delta_max']  # Max steering (rad)
        self.ax_max = mpc_params['ax_max']        # Max accel (m/s^2)
        self.ax_min = mpc_params['ax_min']        # Max braking (m/s^2)
        self.v_max = mpc_params['v_max']          # Max velocity (m/s)
        
        # Build optimization problem
        self._build_mpc_problem()
        
    def _build_mpc_problem(self):
        """
        Build the MPC optimization problem using CasADi.
        This is the core of the path-free racing controller!
        """
        # Create optimization variables
        self.opti = ca.Opti()
        
        # State trajectory: [x, y, psi, u, v, r] over horizon
        self.X = self.opti.variable(6, self.N + 1)
        
        # Control trajectory: [delta, ax] over horizon
        self.U = self.opti.variable(2, self.N)
        
        # Parameters (set at each solve)
        self.X0 = self.opti.parameter(6, 1)  # Initial state
        
        # Cost function
        cost = 0
        
        for k in range(self.N):
            # Current state and control
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            
            # 1. PROGRESS MAXIMIZATION (key for racing!)
            # Reward forward velocity
            u_vel = x_k[3]  # Longitudinal velocity
            cost -= self.Q_progress * u_vel * self.dt
            
            # 2. BOUNDARY VIOLATION COST (potential field)
            x_pos = x_k[0]
            y_pos = x_k[1]
            cost += self.Q_boundary * self.track.boundary_violation_cost(x_pos, y_pos)
            
            # 3. CONTROL EFFORT (smoothness)
            cost += self.R_delta * u_k[0]**2  # Steering effort
            cost += self.R_ax * u_k[1]**2     # Accel effort
            
            # 4. DYNAMICS CONSTRAINTS
            x_next = self.vehicle.dynamics_discrete(x_k, u_k, self.dt)
            self.opti.subject_to(self.X[:, k+1] == x_next)
        
        # Initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.X0)
        
        # Control constraints
        for k in range(self.N):
            # Steering limits
            self.opti.subject_to(self.U[0, k] <= self.delta_max)
            self.opti.subject_to(self.U[0, k] >= -self.delta_max)
            
            # Acceleration limits
            self.opti.subject_to(self.U[1, k] <= self.ax_max)
            self.opti.subject_to(self.U[1, k] >= self.ax_min)
            
            # Velocity limits
            self.opti.subject_to(self.X[3, k] <= self.v_max)
            self.opti.subject_to(self.X[3, k] >= 0.1)  # Minimum velocity
            
            # FRICTION CIRCLE CONSTRAINT (handling limits!)
            # This is crucial for racing at the limit
            u_k = self.X[3, k]
            v_k = self.X[4, k]
            r_k = self.X[5, k]
            delta_k = self.U[0, k]
            ax_k = self.U[1, k]
            
            # Lateral acceleration
            ay = v_k * r_k + u_k * r_k
            
            # Friction circle: sqrt(ax^2 + ay^2) <= mu * g
            g = 9.81
            mu = self.vehicle.mu
            self.opti.subject_to(ax_k**2 + ay**2 <= (mu * g)**2)
        
        # Set objective
        self.opti.minimize(cost)
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-4,
        }
        self.opti.solver('ipopt', opts)
        
    def solve(self, current_state):
        """
        Solve MPC optimization for current state.
        
        Args:
            current_state: [x, y, psi, u, v, r]
            
        Returns:
            optimal_control: [delta, ax] for current timestep
            predicted_trajectory: predicted state trajectory
        """
        # Set initial condition
        self.opti.set_value(self.X0, current_state)
        
        try:
            # Solve optimization
            sol = self.opti.solve()
            
            # Extract solution
            X_opt = sol.value(self.X)
            U_opt = sol.value(self.U)
            
            # Return first control action (receding horizon)
            optimal_control = U_opt[:, 0]
            
            return optimal_control, X_opt
            
        except RuntimeError as e:
            print(f"MPC solver failed: {e}")
            # Return safe fallback control
            return np.array([0.0, -2.0]), None  # Straight + brake


class MPCParameters:
    """Default MPC parameters - TUNE THESE!"""
    
    @staticmethod
    def get_default_params():
        params = {
            'horizon': 20,           # Prediction horizon steps
            'dt': 0.05,              # Time step (50ms = 20Hz)
            'Q_progress': 10.0,      # Progress reward weight
            'Q_boundary': 100.0,     # Boundary penalty weight
            'R_delta': 0.1,          # Steering effort weight
            'R_ax': 0.1,             # Accel effort weight
            'delta_max': 0.5236,     # Max steering angle (30 degrees in rad)
            'ax_max': 3.0,           # Max acceleration (m/s^2)
            'ax_min': -8.0,          # Max braking (m/s^2)
            'v_max': 5.0,            # Max velocity (m/s)
        }
        return params