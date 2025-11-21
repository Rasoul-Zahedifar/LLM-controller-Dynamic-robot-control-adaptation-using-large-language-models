"""
3-Link Manipulator Dynamics Module

This module implements the dynamics model for a 3-link robotic manipulator
including mass matrices, Coriolis/centrifugal forces, gravity, and friction.
"""

import numpy as np
import sympy as sp


class Dynamics3Link:
    """
    Dynamics model for a 3-link planar robotic manipulator.
    
    This class computes the equations of motion for a 3-link manipulator
    including uncertainties, disturbances, and various physical effects.
    
    Attributes:
        g: Gravitational acceleration (m/s^2)
        m1, m2, m3: Masses of links 1, 2, and 3 (kg)
        l1, l2, l3: Lengths of links 1, 2, and 3 (m)
        f_v: Viscous friction coefficient
        f_c: Coulomb friction coefficient
        M_q: Mass/inertia matrix
        C_q: Coriolis and centrifugal matrix
        G_q: Gravity vector
        F_q: Friction forces
        B_q: Input matrix
        disturb_q: External disturbances
        unmodeled_dynamics_q: Unmodeled dynamics
    """
    
    def __init__(self, current_uncertainty):
        """
        Initialize the 3-link dynamics model.
        
        Args:
            current_uncertainty: Tuple of (disturbance_expressions, unmodeled_dynamics_expressions)
                               where each is a list of SymPy expressions
        """
        # Physical parameters
        self.g = 9.81  # Gravity (m/s^2)
        self.m1 = 4.0  # Mass of link 1 (kg)
        self.m2 = 2.5  # Mass of link 2 (kg)
        self.m3 = 1.0  # Mass of link 3 (kg)
        self.l1 = 1.0  # Length of link 1 (m)
        self.l2 = 0.7  # Length of link 2 (m)
        self.l3 = 0.4  # Length of link 3 (m)
        
        # Friction parameters
        self.f_v = 1.0  # Viscous friction
        self.f_c = 2.0  # Coulomb friction
        
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

    def uncertainty(self, t):
        """
        Compute uncertainties at time t.
        
        Args:
            t: Current time
        """
        self.disturb_q = np.array([expr(t) for expr in self.disturb_q_func]).reshape(3, 1)
        self.unmodeled_dynamics_q = np.array([expr(t) for expr in self.unmodeled_dynamics_q_func]).reshape(3, 1)

    def matrices(self, theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot):
        """
        Compute dynamics matrices in task space.
        
        Args:
            theta1, theta2, theta3: Joint angles (rad)
            theta1_dot, theta2_dot, theta3_dot: Joint velocities (rad/s)
            
        Returns:
            Tuple of (M_x, C_x, G_x, F_x, B_x) - dynamics matrices in task space
        """
        # Trigonometric terms
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        s2 = np.sin(theta2)
        c2 = np.cos(theta2)
        s3 = np.sin(theta3)
        c3 = np.cos(theta3)
        s12 = np.sin(theta1 + theta2)
        c12 = np.cos(theta1 + theta2)
        s23 = np.sin(theta2 + theta3)
        c23 = np.cos(theta2 + theta3)
        s123 = np.sin(theta1 + theta2 + theta3)
        c123 = np.cos(theta1 + theta2 + theta3)
        
        # Mass/inertia matrix (joint space) - 3x3
        self.M_q = np.array([
            [(self.m1 + self.m2 + self.m3) * self.l1**2 + (self.m2 + self.m3) * self.l2**2 + self.m3 * self.l3**2 +
             2 * self.m3 * self.l1 * self.l3 * c23 +
             2 * (self.m2 + self.m3) * self.l1 * self.l2 * c2 +
             2 * self.m3 * self.l2 * self.l3 * c3,
             
             (self.m2 + self.m3) * self.l2**2 + self.m3 * self.l3**2 +
             self.m3 * self.l1 * self.l3 * c23 +
             (self.m2 + self.m3) * self.l1 * self.l2 * c2 +
             2 * self.m3 * self.l2 * self.l3 * c3,
             
             self.m3 * self.l3**2 +
             self.m3 * self.l1 * self.l3 * c23 +
             self.m3 * self.l2 * self.l3 * c3],
            
            [(self.m2 + self.m3) * self.l2**2 + self.m3 * self.l3**2 +
             self.m3 * self.l1 * self.l3 * c23 +
             (self.m2 + self.m3) * self.l1 * self.l2 * c2 +
             2 * self.m3 * self.l2 * self.l3 * c3,
             
             self.m2 * self.l2**2 + self.m3 * self.l2**2 + self.m3 * self.l3**2 +
             2 * self.m3 * self.l2 * self.l3 * c3,
             
             self.m3 * self.l3**2 +
             self.m3 * self.l2 * self.l3 * c3],
            
            [self.m3 * self.l3**2 +
             self.m3 * self.l1 * self.l3 * c23 +
             self.m3 * self.l2 * self.l3 * c3,
             
             self.m3 * self.l3**2 +
             self.m3 * self.l2 * self.l3 * c3,
             
             self.m3 * self.l3**2]
        ])
        
        # Coriolis and centrifugal matrix (joint space) - 3x3
        # Using the proper skew-symmetric formulation
        self.C_q = np.array([
            [
                0.5 * theta2_dot * (-self.l1 * self.l2 * (2 * self.m2 + 2 * self.m3) * np.sin(theta2) -
                                    2 * self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3)) +
                0.5 * theta3_dot * (-2 * self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3) -
                                    2 * self.l2 * self.l3 * self.m3 * np.sin(theta3)),
                
                -0.5 * self.l1 * self.l3 * self.m3 * theta3_dot * np.sin(theta2 + theta3) +
                0.5 * theta1_dot * (-self.l1 * self.l2 * (2 * self.m2 + 2 * self.m3) * np.sin(theta2) -
                                    2 * self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3)) +
                theta2_dot * (-self.l1 * self.l2 * (self.m2 + self.m3) * np.sin(theta2) -
                              self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3)) +
                0.5 * theta3_dot * (-self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3) -
                                    2 * self.l2 * self.l3 * self.m3 * np.sin(theta3)),
                
                -0.5 * self.l1 * self.l3 * self.m3 * theta2_dot * np.sin(theta2 + theta3) +
                0.5 * theta1_dot * (-2 * self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3) -
                                    2 * self.l2 * self.l3 * self.m3 * np.sin(theta3)) +
                0.5 * theta2_dot * (-self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3) -
                                    2 * self.l2 * self.l3 * self.m3 * np.sin(theta3)) +
                theta3_dot * (-self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3) -
                              self.l2 * self.l3 * self.m3 * np.sin(theta3))
            ],
            [
                0.5 * self.l1 * self.l3 * self.m3 * theta3_dot * np.sin(theta2 + theta3) +
                0.5 * theta1_dot * (self.l1 * self.l2 * (2 * self.m2 + 2 * self.m3) * np.sin(theta2) +
                                    2 * self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3)) +
                0.5 * theta2_dot * (self.l1 * self.l2 * (self.m2 + self.m3) * np.sin(theta2) +
                                    self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3)) +
                0.5 * theta2_dot * (-self.l1 * self.l2 * self.m2 * np.sin(theta2) -
                                    self.l1 * self.l2 * self.m3 * np.sin(theta2) -
                                    self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3)) +
                0.5 * theta3_dot * (-self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3) -
                                    2 * self.l2 * self.l3 * self.m3 * np.sin(theta3)),
                
                -self.l2 * self.l3 * self.m3 * theta3_dot * np.sin(theta3),
                
                -self.l2 * self.l3 * self.m3 * theta1_dot * np.sin(theta3) -
                self.l2 * self.l3 * self.m3 * theta2_dot * np.sin(theta3) -
                self.l2 * self.l3 * self.m3 * theta3_dot * np.sin(theta3)
            ],
            [
                -0.5 * self.l1 * self.l3 * self.m3 * theta2_dot * np.sin(theta2 + theta3) +
                0.5 * theta1_dot * (2 * self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3) +
                                    2 * self.l2 * self.l3 * self.m3 * np.sin(theta3)) +
                0.5 * theta2_dot * (self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3) +
                                    2 * self.l2 * self.l3 * self.m3 * np.sin(theta3)) +
                0.5 * theta3_dot * (-self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3) -
                                    self.l2 * self.l3 * self.m3 * np.sin(theta3)) +
                0.5 * theta3_dot * (self.l1 * self.l3 * self.m3 * np.sin(theta2 + theta3) +
                                    self.l2 * self.l3 * self.m3 * np.sin(theta3)),
                
                self.l2 * self.l3 * self.m3 * theta1_dot * np.sin(theta3) +
                self.l2 * self.l3 * self.m3 * theta2_dot * np.sin(theta3),
                
                0
            ]
        ])
        
        # Gravity vector (joint space)
        self.G_q = np.array([
            [(self.m1 + self.m2 + self.m3) * self.g * self.l1 * c1 +
             (self.m2 + self.m3) * self.g * self.l2 * c12 +
             self.m3 * self.g * self.l3 * c123],
            
            [(self.m2 + self.m3) * self.g * self.l2 * c12 +
             self.m3 * self.g * self.l3 * c123],
            
            [self.m3 * self.g * self.l3 * c123]
        ])
        
        # Friction forces
        q_dot = np.array([[theta1_dot], [theta2_dot], [theta3_dot]])
        self.F_q = self.f_v * q_dot + self.f_c * np.tanh(q_dot)
        
        # Input matrix
        self.B_q = np.eye(3)
        
        # Jacobian matrix (maps joint velocities to end-effector velocities)
        # For 3-link: [x, y, orientation]
        jacobian = np.array([
            [-self.l1 * s1 - self.l2 * s12 - self.l3 * s123,
             -self.l2 * s12 - self.l3 * s123,
             -self.l3 * s123],
            
            [self.l1 * c1 + self.l2 * c12 + self.l3 * c123,
             self.l2 * c12 + self.l3 * c123,
             self.l3 * c123],
            
            [0, 0, 1]  # Orientation component
        ])
        
        # Jacobian time derivative
        jacobian_dot = np.array([
            [-self.l1 * theta1_dot * c1
             - self.l2 * (theta1_dot + theta2_dot) * c12
             - self.l3 * (theta1_dot + theta2_dot + theta3_dot) * c123,
             
             -self.l2 * (theta1_dot + theta2_dot) * c12
             - self.l3 * (theta1_dot + theta2_dot + theta3_dot) * c123,
             
             -self.l3 * (theta1_dot + theta2_dot + theta3_dot) * c123],
            
            [-self.l1 * theta1_dot * s1
             - self.l2 * (theta1_dot + theta2_dot) * s12
             - self.l3 * (theta1_dot + theta2_dot + theta3_dot) * s123,
             
             -self.l2 * (theta1_dot + theta2_dot) * s12
             - self.l3 * (theta1_dot + theta2_dot + theta3_dot) * s123,
             
             -self.l3 * (theta1_dot + theta2_dot + theta3_dot) * s123],
            
            [0, 0, 0]
        ])
        
        # Transform to task space
        # Use pseudo-inverse to handle singular configurations
        try:
            J_inv = np.linalg.inv(jacobian)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse with small regularization
            J_inv = np.linalg.pinv(jacobian, rcond=1e-6)
        
        M_x = J_inv.T @ self.M_q @ J_inv
        C_x = J_inv.T @ (self.C_q - self.M_q @ J_inv @ jacobian_dot) @ J_inv
        G_x = J_inv.T @ self.G_q
        F_x = J_inv.T @ self.F_q
        B_x = J_inv.T @ self.B_q
        
        return M_x, C_x, G_x, F_x, B_x

    def run(self, q, q_dot, taw_motor, dt):
        """
        Simulate one time step of the dynamics.
        
        Args:
            q: Joint positions (3x1 array)
            q_dot: Joint velocities (3x1 array)
            taw_motor: Motor torques (3x1 array)
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

