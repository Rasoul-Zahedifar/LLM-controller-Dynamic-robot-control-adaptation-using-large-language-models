"""
System Parameters Configuration Module

This module provides a configuration class for system parameters including
initial conditions, gains, uncertainties, and simulation settings.
"""

import numpy as np
import sympy as sp


class SystemParams:
    """
    Configuration class for system parameters.
    
    This class holds all configuration parameters for the robotic manipulator
    control system including initial conditions, controller gains, reference
    signals, and simulation parameters.
    """
    
    def __init__(self, system_type='2link'):
        """
        Initialize system parameters.
        
        Args:
            system_type: '2link' or '3link' to specify manipulator type
        """
        self.system_type = system_type
        
        # Time symbol for symbolic expressions
        self.t = sp.Symbol('t')
        
        # Set default parameters based on system type
        if system_type == '2link':
            self._init_2link_defaults()
        elif system_type == '3link':
            self._init_3link_defaults()
        else:
            raise ValueError(f"Unknown system type: {system_type}")
    
    def _init_2link_defaults(self):
        """Initialize default parameters for 2-link system."""
        # Initial conditions
        self.q_init = np.array([[0.0], [0.0]])  # Initial joint angles
        self.q_dot_init = np.array([[0.0], [0.0]])  # Initial joint velocities
        
        # Controller gains [k_p, k_d, k_i, landa_1, landa_2]
        self.gains = [50.0, 30.0, 20.0, 0.5, 0.5]
        
        # Reference signals (symbolic expressions)
        # Example: circular trajectory
        self.current_ref_sig = [
            0.5 * sp.cos(self.t),  # x reference
            0.5 * sp.sin(self.t)   # y reference
        ]
        
        self.new_ref_sig = [
            0.6 * sp.cos(1.2 * self.t),  # Modified x reference
            0.6 * sp.sin(1.2 * self.t)   # Modified y reference
        ]
        
        # Uncertainties (disturbances and unmodeled dynamics)
        self.current_uncertainty = (
            [0 * self.t, 0 * self.t],  # Disturbances
            [0 * self.t, 0 * self.t]   # Unmodeled dynamics
        )
        
        self.new_uncertainty = (
            [0.1 * sp.sin(2 * self.t), 0.1 * sp.cos(2 * self.t)],  # Disturbances
            [0.05 * self.t, 0.05 * self.t]  # Unmodeled dynamics
        )
        
        # Simulation parameters
        self.dt = 0.01  # Time step (s)
        self.sim_time = 10.0  # Simulation duration (s)
        self.sim_num_trial = 5  # Number of simulation trials
        self.attempt_num_trial = 3  # Number of tuning attempts
        
    def _init_3link_defaults(self):
        """Initialize default parameters for 3-link system."""
        # Initial conditions
        self.q_init = np.array([[0.0], [0.0], [0.0]])  # Initial joint angles
        self.q_dot_init = np.array([[0.0], [0.0], [0.0]])  # Initial joint velocities
        
        # Controller gains [k_p, k_d, k_i, landa_1, landa_2]
        self.gains = [60.0, 35.0, 25.0, 0.6, 0.6]
        
        # Reference signals (symbolic expressions)
        # Example: circular trajectory
        self.current_ref_sig = [
            0.5 * sp.cos(self.t),  # x reference
            0.5 * sp.sin(self.t)   # y reference
        ]
        
        self.new_ref_sig = [
            0.6 * sp.cos(1.2 * self.t),  # Modified x reference
            0.6 * sp.sin(1.2 * self.t)   # Modified y reference
        ]
        
        # Uncertainties (disturbances and unmodeled dynamics)
        self.current_uncertainty = (
            [0 * self.t, 0 * self.t, 0 * self.t],  # Disturbances
            [0 * self.t, 0 * self.t, 0 * self.t]   # Unmodeled dynamics
        )
        
        self.new_uncertainty = (
            [0.1 * sp.sin(2 * self.t), 0.1 * sp.cos(2 * self.t), 0.05 * sp.sin(self.t)],  # Disturbances
            [0.05 * self.t, 0.05 * self.t, 0.03 * self.t]  # Unmodeled dynamics
        )
        
        # Simulation parameters
        self.dt = 0.01  # Time step (s)
        self.sim_time = 10.0  # Simulation duration (s)
        self.sim_num_trial = 5  # Number of simulation trials
        self.attempt_num_trial = 3  # Number of tuning attempts
    
    def get_timesteps(self):
        """
        Generate timesteps array for simulation.
        
        Returns:
            numpy array of timesteps
        """
        return np.arange(0, self.sim_time, self.dt)
    
    def update_gains(self, k_p, k_d, k_i, landa_1, landa_2):
        """
        Update controller gains.
        
        Args:
            k_p: Proportional gain
            k_d: Derivative gain
            k_i: Integral gain
            landa_1: Adaptive gain 1
            landa_2: Adaptive gain 2
        """
        self.gains = [k_p, k_d, k_i, landa_1, landa_2]
    
    def set_reference_signal(self, ref_sig_expressions, is_new=False):
        """
        Set reference signal expressions.
        
        Args:
            ref_sig_expressions: List of SymPy expressions for reference signals
            is_new: If True, sets new_ref_sig; otherwise sets current_ref_sig
        """
        if is_new:
            self.new_ref_sig = ref_sig_expressions
        else:
            self.current_ref_sig = ref_sig_expressions
    
    def set_uncertainty(self, disturbances, unmodeled_dynamics, is_new=False):
        """
        Set uncertainty expressions.
        
        Args:
            disturbances: List of SymPy expressions for disturbances
            unmodeled_dynamics: List of SymPy expressions for unmodeled dynamics
            is_new: If True, sets new_uncertainty; otherwise sets current_uncertainty
        """
        uncertainty = (disturbances, unmodeled_dynamics)
        if is_new:
            self.new_uncertainty = uncertainty
        else:
            self.current_uncertainty = uncertainty

