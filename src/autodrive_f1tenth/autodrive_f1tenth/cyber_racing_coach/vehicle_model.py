import numpy as np
import casadi as ca

class BicycleModel3DOF:
    def __init__(self, params):
        """
        Initialize 3-DOF bicycle model with vehicle parameters.
        
        Args:
            params (dict): Vehicle parameters
                - m: mass (kg)
                - Iz: yaw moment of inertia (kg*m^2)
                - lf: distance from CG to front axle (m)
                - lr: distance from CG to rear axle (m)
                - Cf: front cornering stiffness (N/rad)
                - Cr: rear cornering stiffness (N/rad)
                - mu: friction coefficient
        """
        self.m = params['m']
        self.Iz = params['Iz']
        self.lf = params['lf']
        self.lr = params['lr']
        self.Cf = params['Cf']
        self.Cr = params['Cr']
        self.mu = params['mu']
        
    def dynamics_continuous(self, state, control):
        """
        Continuous-time dynamics: dx/dt = f(x, u)
        
        Args:
            state: [x, y, psi, u, v, r]
            control: [delta, ax] (steering angle, longitudinal accel)
            
        Returns:
            state_dot: time derivative of state
        """
        # Unpack state
        x, y, psi, u, v, r = state[0], state[1], state[2], state[3], state[4], state[5]
        
        # Unpack control
        delta = control[0]  # Front wheel steering angle (rad)
        ax = control[1]     # Longitudinal acceleration (m/s^2)
        
        # Tire slip angles
        alpha_f = delta - ca.atan2(v + self.lf * r, u)
        alpha_r = -ca.atan2(v - self.lr * r, u)
        
        # Lateral tire forces (linear tire model)
        Fyf = self.Cf * alpha_f
        Fyr = self.Cr * alpha_r
        
        # State derivatives
        x_dot = u * ca.cos(psi) - v * ca.sin(psi)
        y_dot = u * ca.sin(psi) + v * ca.cos(psi)
        psi_dot = r
        u_dot = ax + v * r  # Longitudinal dynamics
        v_dot = (Fyf + Fyr) / self.m - u * r  # Lateral dynamics
        r_dot = (self.lf * Fyf - self.lr * Fyr) / self.Iz  # Yaw dynamics
        
        return ca.vertcat(x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot)
    
    def dynamics_discrete(self, state, control, dt):
        """
        Discrete-time dynamics using RK4 integration.
        
        Args:
            state: current state [x, y, psi, u, v, r]
            control: control input [delta, ax]
            dt: time step (s)
            
        Returns:
            next_state: state at next time step
        """
        # RK4 integration
        k1 = self.dynamics_continuous(state, control)
        k2 = self.dynamics_continuous(state + dt/2 * k1, control)
        k3 = self.dynamics_continuous(state + dt/2 * k2, control)
        k4 = self.dynamics_continuous(state + dt * k3, control)
        
        next_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return next_state


class VehicleParameters:
    """Default F1TENTH parameters - ADJUST THESE based on AutoDRIVE specs"""
    
    @staticmethod
    def get_f1tenth_params():
        """
        Get F1TENTH vehicle parameters.
        TODO: Get actual values from AutoDRIVE technical guide
        """
        params = {
            'm': 3.47,          # Mass (kg) - PLACEHOLDER
            'Iz': 0.04,         # Yaw inertia (kg*m^2) - PLACEHOLDER
            'lf': 0.15875,      # Distance CG to front axle (m) - PLACEHOLDER
            'lr': 0.17145,      # Distance CG to rear axle (m) - PLACEHOLDER
            'Cf': 4.718,        # Front cornering stiffness (N/rad) - PLACEHOLDER
            'Cr': 5.4562,       # Rear cornering stiffness (N/rad) - PLACEHOLDER
            'mu': 1.0,          # Friction coefficient - PLACEHOLDER
            'width': 0.2,       # Vehicle width (m)
            'length': 0.33,     # Vehicle length (m)
        }
        return params