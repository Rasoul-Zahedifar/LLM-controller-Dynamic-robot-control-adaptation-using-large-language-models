"""
Reference Signal Generator Module

This module provides classes for generating reference signals and their derivatives
for robotic manipulator control systems.
"""

import numpy as np
import sympy as sp


class RefSigGen:
    """
    Reference Signal Generator for current/initial reference trajectories.
    
    Generates reference signals and their first and second derivatives using
    symbolic differentiation with SymPy.
    
    Attributes:
        ref_signals: List of lambdified reference signal functions
        ref_signals_dot: List of first derivative functions
        ref_signals_ddot: List of second derivative functions
        desired_path: List storing the trajectory history
    """
    
    def __init__(self, ref_signals):
        """
        Initialize the reference signal generator.
        
        Args:
            ref_signals: List of SymPy expressions representing reference signals
                        as functions of time variable 't'
        """
        # Get the time variable from sympy
        t = sp.Symbol('t')
        
        # Compute derivatives symbolically
        ref_dot = [sp.diff(expr, t) for expr in ref_signals]
        ref_ddot = [sp.diff(expr, t) for expr in ref_dot]
        
        # Convert symbolic expressions to numerical functions
        self.ref_signals = [sp.lambdify(t, expr) for expr in ref_signals]
        self.ref_signals_dot = [sp.lambdify(t, expr) for expr in ref_dot]
        self.ref_signals_ddot = [sp.lambdify(t, expr) for expr in ref_ddot]
        
        # Store trajectory history
        self.desired_path = []

    def run(self, t):
        """
        Compute reference signals and derivatives at time t.
        
        Args:
            t: Current time value
            
        Returns:
            Tuple of (ref_sig, ref_sig_dot, ref_sig_ddot) as numpy arrays
            Each is shaped as (2, 1) for 2-link or (3, 1) for 3-link systems
        """
        # Evaluate all reference signals at time t
        ref_signals_val = [expr(t) for expr in self.ref_signals]
        ref_signals_dot_val = [expr(t) for expr in self.ref_signals_dot]
        ref_signals_ddot_val = [expr(t) for expr in self.ref_signals_ddot]
        
        # Convert to numpy arrays
        ref_sig = np.array([ref_signals_val])
        self.desired_path.append(ref_sig.flatten())
        
        ref_sig_dot = np.array([ref_signals_dot_val])
        ref_sig_ddot = np.array([ref_signals_ddot_val])
        
        # Reshape to column vectors
        n_dims = len(ref_signals_val)
        return (ref_sig.reshape(n_dims, 1), 
                ref_sig_dot.reshape(n_dims, 1), 
                ref_sig_ddot.reshape(n_dims, 1))


class NewRefSigGen:
    """
    New Reference Signal Generator for updated/adapted reference trajectories.
    
    Similar to RefSigGen but used for new reference signals after adaptation
    or environmental changes.
    
    Attributes:
        new_ref_signals: List of lambdified new reference signal functions
        new_ref_signals_dot: List of first derivative functions
        new_ref_signals_ddot: List of second derivative functions
        new_desired_path: List storing the new trajectory history
    """
    
    def __init__(self, new_ref_signals):
        """
        Initialize the new reference signal generator.
        
        Args:
            new_ref_signals: List of SymPy expressions representing new reference
                           signals as functions of time variable 't'
        """
        # Get the time variable from sympy
        t = sp.Symbol('t')
        
        # Compute derivatives symbolically
        new_ref_dot = [sp.diff(expr, t) for expr in new_ref_signals]
        new_ref_ddot = [sp.diff(expr, t) for expr in new_ref_dot]
        
        # Convert symbolic expressions to numerical functions
        self.new_ref_signals = [sp.lambdify(t, expr) for expr in new_ref_signals]
        self.new_ref_signals_dot = [sp.lambdify(t, expr) for expr in new_ref_dot]
        self.new_ref_signals_ddot = [sp.lambdify(t, expr) for expr in new_ref_ddot]
        
        # Store trajectory history
        self.new_desired_path = []

    def run(self, t):
        """
        Compute new reference signals and derivatives at time t.
        
        Args:
            t: Current time value
            
        Returns:
            Tuple of (new_ref_sig, new_ref_sig_dot, new_ref_sig_ddot) as numpy arrays
            Each is shaped as (2, 1) for 2-link or (3, 1) for 3-link systems
        """
        # Evaluate all new reference signals at time t
        new_ref_signals_val = [expr(t) for expr in self.new_ref_signals]
        new_ref_signals_dot_val = [expr(t) for expr in self.new_ref_signals_dot]
        new_ref_signals_ddot_val = [expr(t) for expr in self.new_ref_signals_ddot]
        
        # Convert to numpy arrays
        new_ref_sig = np.array([new_ref_signals_val])
        self.new_desired_path.append(new_ref_sig.flatten())
        
        new_ref_sig_dot = np.array([new_ref_signals_dot_val])
        new_ref_sig_ddot = np.array([new_ref_signals_ddot_val])
        
        # Reshape to column vectors
        n_dims = len(new_ref_signals_val)
        return (new_ref_sig.reshape(n_dims, 1), 
                new_ref_sig_dot.reshape(n_dims, 1), 
                new_ref_sig_ddot.reshape(n_dims, 1))

