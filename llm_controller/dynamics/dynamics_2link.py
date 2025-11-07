"""
2-Link Manipulator Dynamics Module

This module implements the dynamics model for a 2-link robotic manipulator
including mass matrices, Coriolis/centrifugal forces, gravity, and friction.
"""

import numpy as np
import sympy as sp


class Dynamics:
    """
    Dynamics model for a 2-link planar robotic manipulator.
    
    This class computes the equations of motion for a 2-link manipulator
    including uncertainties, disturbances, and various physical effects.
    
    Attributes:
        g: Gravitational acceleration (m/s^2)
        m1, m2: Masses of link 1 and 2 (kg)
        l1, l2: Lengths of link 1 and 2 (m)
        f_v, f_v_UN: Viscous friction coefficients
        f_c, f_c_UN: Coulomb friction coefficients
        M_q: Mass/inertia matrix
        C_q: Coriolis and centrifugal matrix
        G_q: Gravity vector
        F_q, F_q_UN: Friction forces
        B_q: Input matrix
        disturb_q: External disturbances
        unmodeled_dynamics_q: Unmodeled dynamics
    """
    
    def __init__(self, current_uncertainty):
        """
        Initialize the 2-link dynamics model.
        
        Args:
            current_uncertainty: Tuple of (disturbance_expressions, unmodeled_dynamics_expressions)
                               where each is a list of SymPy expressions
        """
        # Physical parameters
        self.g = 9.81  # Gravity (m/s^2)
        self.m1 = 4.0  # Mass of link 1 (kg)
        self.m2 = 2.5  # Mass of link 2 (kg)
        self.l1 = 1.0  # Length of link 1 (m)
        self.l2 = 0.7  # Length of link 2 (m)
        
        # Friction parameters
        self.f_v = self.f_v_UN = 1.0  # Viscous friction
        self.f_c = self.f_c_UN = 2.0  # Coulomb friction
        
        # Initialize uncertainty functions
        self.update(current_uncertainty)

    def update(self, new_uncertainty):
        """
        Update uncertainty models.
        
        Args:
            new_uncertainty: Tuple of (disturbance_expressions, unmodeled_dynamics_expressions)
        """
        t = sp.Symbol('t')
        uncertainty1, uncertainty2 = new_uncertainty
        self.disturb_q_func = [sp.lambdify(t, expr) for expr in uncertainty1]
        self.unmodeled_dynamics_q_func = [sp.lambdify(t, expr) for expr in uncertainty2]

    def payload(self, load):
        """
        Add payload mass to link 2.
        
        Args:
            load: Additional mass to add (kg)
        """
        self.m2 += load

    def friction(self, f_v, f_c):
        """
        Update friction coefficients.
        
        Args:
            f_v: Viscous friction coefficient
            f_c: Coulomb friction coefficient
        """
        self.f_v_UN = f_v
        self.f_c_UN = f_c

    def telescope(self, length):
        """
        Adjust length of link 2 (telescoping link).
        
        Args:
            length: Length change (positive to shorten, negative to extend)
        """
        self.l2 -= length

    def uncertainty(self, t):
        """
        Compute uncertainties at time t.
        
        Args:
            t: Current time
        """
        self.disturb_q = np.array([expr(t) for expr in self.disturb_q_func]).reshape(2, 1)
        self.unmodeled_dynamics_q = np.array([expr(t) for expr in self.unmodeled_dynamics_q_func]).reshape(2, 1)

    def matrices(self, theta1, theta2, theta1_dot, theta2_dot):
        """
        Compute dynamics matrices in task space.
        
        Args:
            theta1, theta2: Joint angles (rad)
            theta1_dot, theta2_dot: Joint velocities (rad/s)
            
        Returns:
            Tuple of (M_x, C_x, G_x, F_x, B_x) - dynamics matrices in task space
        """
        # Trigonometric terms
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        s2 = np.sin(theta2)
        c2 = np.cos(theta2)
        c12 = np.cos(theta1 + theta2)
        
        # Mass/inertia matrix (joint space)
        self.M_q = np.array([
            [(self.m1 + self.m2) * self.l1**2 + self.m2 * self.l2**2 + 2 * self.m2 * self.l1 * self.l2 * c2,
             self.m2 * self.l2**2 + self.m2 * self.l1 * self.l2 * c2],
            [self.m2 * self.l2**2 + self.m2 * self.l1 * self.l2 * c2,
             self.m2 * self.l2**2]
        ])
        
        # Coriolis and centrifugal matrix (joint space)
        self.C_q = np.array([
            [-self.m2 * self.l1 * self.l2 * theta2_dot**2 * s2,
             -self.m2 * self.l1 * self.l2 * (theta1_dot + theta2_dot) * s2],
            [self.m2 * self.l1 * self.l2 * theta1_dot * s2,
             0]
        ])
        
        # Gravity vector (joint space)
        self.G_q = np.array([
            [(self.m1 + self.m2) * self.g * self.l1 * c1 + self.m2 * self.g * self.l2 * c12],
            [self.m2 * self.g * self.l2 * c12]
        ])
        
        # Friction forces
        q_dot = np.array([[theta1_dot], [theta2_dot]])
        self.F_q = self.f_v * q_dot + self.f_c * np.tanh(q_dot)
        self.F_q_UN = self.f_v_UN * q_dot + self.f_c_UN * np.tanh(q_dot)
        
        # Input matrix
        self.B_q = np.eye(2)
        
        # Jacobian matrix (maps joint velocities to end-effector velocities)
        jacobian = np.array([
            [-self.l1 * np.sin(theta1) - self.l2 * np.sin(theta1 + theta2),
             -self.l2 * np.sin(theta1 + theta2)],
            [self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2),
             self.l2 * np.cos(theta1 + theta2)]
        ])
        
        # Jacobian time derivative
        jacobian_dot = np.array([
            [-self.l1 * theta1_dot * np.cos(theta1) - self.l2 * (theta1_dot + theta2_dot) * np.cos(theta1 + theta2),
             -self.l2 * (theta1_dot + theta2_dot) * np.cos(theta1 + theta2)],
            [-self.l1 * theta1_dot * np.sin(theta1) - self.l2 * (theta1_dot + theta2_dot) * np.sin(theta1 + theta2),
             -self.l2 * (theta1_dot + theta2_dot) * np.sin(theta1 + theta2)]
        ])
        
        # Transform to task space
        J_inv = np.linalg.inv(jacobian)
        M_x = J_inv.T @ self.M_q @ J_inv
        C_x = J_inv.T @ (self.C_q - self.M_q @ J_inv @ jacobian_dot) @ J_inv
        G_x = J_inv.T @ self.G_q
        F_x = J_inv.T @ self.F_q_UN
        B_x = J_inv.T @ self.B_q
        
        return M_x, C_x, G_x, F_x, B_x

    def run(self, q, q_dot, taw_motor, dt):
        """
        Simulate one time step of the dynamics.
        
        Args:
            q: Joint positions (2x1 array)
            q_dot: Joint velocities (2x1 array)
            taw_motor: Motor torques (2x1 array)
            dt: Time step (s)
            
        Returns:
            Tuple of (q, q_dot, q_ddot, taw_motor, work)
        """
        # Compute joint accelerations using equations of motion
        q_ddot = np.linalg.inv(self.M_q) @ (
            self.B_q @ taw_motor + 
            self.disturb_q + 
            self.unmodeled_dynamics_q - 
            self.C_q @ q_dot - 
            self.G_q - 
            self.F_q
        )
        
        # Integrate to get velocities and positions
        q_dot += q_ddot * dt
        q += q_dot * dt
        
        # Compute work done
        work = np.sum(taw_motor.T @ (q_dot * dt))
        
        return q, q_dot, q_ddot, taw_motor, work

