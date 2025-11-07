"""
3-Link Manipulator Controller Module

This module implements an adaptive sliding mode controller for a 3-link
robotic manipulator with PID-based sliding surface and uncertainty estimation.
"""

import numpy as np


class Controller3Link:
    """
    Adaptive sliding mode controller for 3-link manipulator.
    
    This controller uses:
    - PID-based sliding surface
    - Adaptive uncertainty estimation (d_hat, kesi_hat)
    - Task-space control with computed torque method
    - Torque saturation limits
    - Handles 3D task space (x, y, orientation)
    
    Attributes:
        dynamics: Dynamics model instance
        l1, l2, l3: Link lengths (m)
        k_p, k_d, k_i: PID gains
        landa_1, landa_2: Adaptive gain parameters
        E_int: Integrated error
        d_hat: Estimated disturbance bound
        kesi_hat: Estimated uncertainty bound
        upper_limit, lower_limit: Torque saturation limits
    """
    
    def __init__(self, dynamics, k_p, k_d, k_i, landa_1, landa_2):
        """
        Initialize the controller.
        
        Args:
            dynamics: Dynamics model instance
            k_p: Proportional gain
            k_d: Derivative gain
            k_i: Integral gain
            landa_1: Adaptive gain for disturbance estimation
            landa_2: Adaptive gain for uncertainty estimation
        """
        self.dynamics = dynamics
        self.l1 = dynamics.l1
        self.l2 = dynamics.l2
        self.l3 = dynamics.l3
        
        # Set controller gains
        self.update(k_p, k_d, k_i, landa_1, landa_2)
        
        # Initialize state variables
        self.E_int = np.zeros((3, 1))  # Integrated error (3D task space)
        self.d_hat = 0  # Estimated disturbance bound
        self.kesi_hat = 0  # Estimated uncertainty bound
        
        # Torque limits
        self.upper_limit = 200
        self.lower_limit = -150

    def update(self, k_p, k_d, k_i, landa_1, landa_2):
        """
        Update controller gains.
        
        Args:
            k_p: Proportional gain
            k_d: Derivative gain
            k_i: Integral gain
            landa_1: Adaptive gain for disturbance estimation
            landa_2: Adaptive gain for uncertainty estimation
        """
        self.k_p = k_p
        self.k_d = k_d
        self.k_i = k_i
        self.landa_1 = landa_1
        self.landa_2 = landa_2

    def run(self, q, q_dot, X_d, X_d_dot, X_d_ddot, dt):
        """
        Compute control torques and simulate one time step.
        
        Args:
            q: Joint positions (3x1 array)
            q_dot: Joint velocities (3x1 array)
            X_d: Desired end-effector position (2x1 array, will be augmented to 3x1)
            X_d_dot: Desired end-effector velocity (2x1 array, will be augmented to 3x1)
            X_d_ddot: Desired end-effector acceleration (2x1 array, will be augmented to 3x1)
            dt: Time step (s)
            
        Returns:
            Tuple of (q, q_dot, q_ddot, taw_motor, work) from dynamics.run()
        """
        # Extract joint angles and velocities
        theta1, theta2, theta3 = q.flatten()
        theta1_dot, theta2_dot, theta3_dot = q_dot.flatten()
        
        # Compute forward kinematics (end-effector position + orientation)
        X = np.array([
            [self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2) + self.l3 * np.cos(theta1 + theta2 + theta3)],
            [self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2) + self.l3 * np.sin(theta1 + theta2 + theta3)],
            [theta3]  # Orientation component
        ])
        
        # Compute end-effector velocity
        X_dot = np.array([
            [-self.l1 * theta1_dot * np.sin(theta1)
             - self.l2 * (theta1_dot + theta2_dot) * np.sin(theta1 + theta2)
             - self.l3 * (theta1_dot + theta2_dot + theta3_dot) * np.sin(theta1 + theta2 + theta3)],
            
            [self.l1 * theta1_dot * np.cos(theta1)
             + self.l2 * (theta1_dot + theta2_dot) * np.cos(theta1 + theta2)
             + self.l3 * (theta1_dot + theta2_dot + theta3_dot) * np.cos(theta1 + theta2 + theta3)],
            
            [theta3_dot]
        ])
        
        # Get dynamics matrices in task space
        M_x, C_x, G_x, F_x, B_x = self.dynamics.matrices(theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot)
        
        # Augment desired trajectory with orientation component
        # Using fixed orientation (pi/3) for the third link
        X_d = np.append(X_d, np.pi / 3)
        X_d_dot = np.append(X_d_dot, 0)
        X_d_ddot = np.append(X_d_ddot, 0)
        X_d = X_d.reshape(-1, 1)
        X_d_dot = X_d_dot.reshape(-1, 1)
        X_d_ddot = X_d_ddot.reshape(-1, 1)
        
        # Compute tracking errors
        error = X - X_d
        error_dot = X_dot - X_d_dot
        
        # Compute virtual control (sliding surface reference)
        v = X_d_dot - (self.k_p / self.k_d) * error - (self.k_i / self.k_d) * self.E_int
        v_dot = X_d_ddot - (self.k_p / self.k_d) * error_dot - (self.k_i / self.k_d) * error
        
        # Sliding surface
        s = X_dot - v
        
        # Compute control force in task space
        # Note: Using C_x @ X_dot instead of C_x @ v for better performance
        f_motor = (M_x @ v_dot + 
                   C_x @ X_dot + 
                   G_x + 
                   F_x - 
                   self.k_d * M_x @ s - 
                   self.d_hat * M_x @ np.tanh(s) - 
                   self.kesi_hat * M_x @ np.tanh(s))
        
        # Transform to joint torques
        taw_motor = np.linalg.inv(B_x) @ f_motor
        
        # Apply torque saturation
        for i in range(len(taw_motor)):
            if taw_motor[i] > self.upper_limit:
                taw_motor[i] = self.upper_limit - 5 * i
            elif taw_motor[i] < self.lower_limit:
                taw_motor[i] = self.lower_limit + 5 * i
        
        # Update adaptive estimates
        d_hat_dot = self.landa_1 * (np.abs(s[0]) + np.abs(s[1]))
        kesi_hat_dot = self.landa_2 * (np.abs(s[0]) + np.abs(s[1]))
        self.d_hat += d_hat_dot * dt
        self.kesi_hat += kesi_hat_dot * dt
        
        # Update integrated error
        self.E_int += error * dt
        
        # Run dynamics simulation
        return self.dynamics.run(q, q_dot, taw_motor, dt)

