"""
3-Link Manipulator Runner Module

This module provides the main simulation runner for the 3-link manipulator
system with LLM-based controller tuning capabilities.
"""

import os
import shutil
import numpy as np
from llm_controller.dynamics.dynamics_3link import Dynamics3Link
from llm_controller.controllers.controller_3link import Controller3Link
from llm_controller.core.reference_signal import RefSigGen, NewRefSigGen


class Runner3Link:
    """
    Main simulation runner for 3-link manipulator with LLM tuning.
    
    This class orchestrates the simulation, performance evaluation, and
    LLM-based controller parameter tuning for a 3-link robotic manipulator.
    
    Attributes:
        dynamics: Dynamics model instance
        controller: Controller instance
        ref_sig_gen: Reference signal generator
        new_ref_sig_gen: New reference signal generator
        llm: LLM chain instance for tuning
        Various tracking arrays for simulation data
    """
    
    def __init__(self, llm, prompt, q_init, q_dot_init, current_ref_sig, new_ref_sig,
                 current_uncertainty, new_uncertainty, gains, update_gains,
                 sim_num_trial, attempt_num_trial, output_dir):
        """
        Initialize the runner.
        
        Args:
            llm: LLM chain instance
            prompt: Prompt template
            q_init: Initial joint positions (3x1)
            q_dot_init: Initial joint velocities (3x1)
            current_ref_sig: Current reference signal expressions
            new_ref_sig: New reference signal expressions
            current_uncertainty: Current uncertainty expressions
            new_uncertainty: New uncertainty expressions
            gains: Initial controller gains [k_p, k_d, k_i, landa_1, landa_2]
            update_gains: Whether to update gains with LLM
            sim_num_trial: Number of simulation trials
            attempt_num_trial: Number of tuning attempts
            output_dir: Directory for output files
        """
        # LLM configuration
        self.llm_model = llm
        self.prompt = prompt
        self.update_gains = update_gains
        
        # Controller gains
        self.current_k_p = gains[0]
        self.current_k_d = gains[1]
        self.current_k_i = gains[2]
        self.current_landa_1 = gains[3]
        self.current_landa_2 = gains[4]
        
        # Initialize dynamics and controller
        self.dynamics = Dynamics3Link(current_uncertainty)
        self.new_uncertainty = new_uncertainty
        self.controller = Controller3Link(
            self.dynamics,
            self.current_k_p,
            self.current_k_d,
            self.current_k_i,
            self.current_landa_1,
            self.current_landa_2
        )
        
        # System parameters
        self.l1 = self.controller.l1
        self.l2 = self.controller.l2
        self.l3 = self.controller.l3
        
        # Reference signals
        self.current_ref_sig = current_ref_sig
        self.ref_sig_gen = RefSigGen(current_ref_sig)
        self.new_ref_sig = new_ref_sig
        self.new_ref_sig_gen = NewRefSigGen(new_ref_sig)
        
        # Initial conditions
        self.q_init = q_init
        self.q_dot_init = q_dot_init
        
        # Simulation parameters
        self.sim_num_trial = sim_num_trial
        self.attempt_num_trial = attempt_num_trial
        self.output_dir = output_dir
        
        # Clear output directory if it exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data storage
        self.theta1_traj = []
        self.theta2_traj = []
        self.theta3_traj = []
        self.end_effectors1 = []
        self.end_effectors2 = []
        self.end_effectors3 = []
        self.end_effectors3_d = []
        self.reachable_check = []
        self.controller_input = []
        self.work_done = []
        self.error = []
        self.error_dot = []
        self.error_ddot = []
        
        # Performance metrics
        self.performance_metric_init()
    
    def performance_metric_init(self):
        """Initialize performance metric tracking variables."""
        self.rise_time1 = None
        self.rise_time2 = None
        self.rise_time3 = None
        self.reached_rise1 = False
        self.reached_rise2 = False
        self.reached_rise3 = False
        
        self.peak_value1 = -np.inf
        self.peak_value2 = -np.inf
        self.peak_value3 = -np.inf
        self.peak_time1 = None
        self.peak_time2 = None
        self.peak_time3 = None
        
        self.overshoot1 = None
        self.overshoot2 = None
        self.overshoot3 = None
        
        self.settle_time1 = None
        self.settle_start_time1 = None
        self.settle_time2 = None
        self.settle_start_time2 = None
        self.settle_time3 = None
        self.settle_start_time3 = None
        
        self.settle_threshold = 0.05
        self.settle_time_window = 0.1
        self.threshold = 0.9
    
    def performance_metric(self, X, ref_X, t):
        """
        Update performance metrics during simulation.
        
        Args:
            X: Current end-effector position (3x1: x, y, orientation)
            ref_X: Reference end-effector position (3x1)
            t: Current time
        """
        # Rise time tracking for x and y coordinates
        if not self.reached_rise1 and np.abs(X[0, 0]) >= self.threshold * np.abs(ref_X[0, 0]):
            self.reached_rise1 = True
            self.rise_time1 = t
        
        if not self.reached_rise2 and np.abs(X[1, 0]) >= self.threshold * np.abs(ref_X[1, 0]):
            self.reached_rise2 = True
            self.rise_time2 = t
        
        # Peak/overshoot tracking
        if self.reached_rise1 and (np.abs(X[0, 0] - ref_X[0, 0]) > self.peak_value1):
            self.peak_value1 = np.abs(X[0, 0] - ref_X[0, 0])
            self.peak_time1 = t
            if np.abs(ref_X[0, 0]) > 1e-6:
                self.overshoot1 = (self.peak_value1 / np.abs(ref_X[0, 0])) * 100
        
        if self.reached_rise2 and (np.abs(X[1, 0] - ref_X[1, 0]) > self.peak_value2):
            self.peak_value2 = np.abs(X[1, 0] - ref_X[1, 0])
            self.peak_time2 = t
            if np.abs(ref_X[1, 0]) > 1e-6:
                self.overshoot2 = (self.peak_value2 / np.abs(ref_X[1, 0])) * 100
        
        # Settling time tracking
        if self.reached_rise1:
            if np.abs(X[0, 0] - ref_X[0, 0]) <= self.settle_threshold * np.abs(ref_X[0, 0]):
                if self.settle_start_time1 is None:
                    self.settle_start_time1 = t
                elif t - self.settle_start_time1 >= self.settle_time_window:
                    self.settle_time1 = t
            else:
                self.settle_start_time1 = None
        
        if self.reached_rise2:
            if np.abs(X[1, 0] - ref_X[1, 0]) <= self.settle_threshold * np.abs(ref_X[1, 0]):
                if self.settle_start_time2 is None:
                    self.settle_start_time2 = t
                elif t - self.settle_start_time2 >= self.settle_time_window:
                    self.settle_time2 = t
            else:
                self.settle_start_time2 = None
    
    def run_sim(self, timesteps, dt):
        """
        Run a single simulation trial.
        
        Args:
            timesteps: Array of timesteps
            dt: Time step size
            
        Returns:
            Dictionary containing simulation results and metrics
        """
        # Reset state
        q = self.q_init.copy()
        q_dot = self.q_dot_init.copy()
        
        # Storage for this trial
        theta1_traj = []
        theta2_traj = []
        theta3_traj = []
        end_effectors1 = []
        end_effectors2 = []
        end_effectors3 = []
        end_effectors3_d = []
        controller_input = []
        work_done = []
        error = []
        
        # Reset performance metrics
        self.performance_metric_init()
        
        # Simulation loop
        for t in timesteps:
            # Get reference trajectory (2D: x, y)
            ref_X, ref_X_dot, ref_X_ddot = self.ref_sig_gen.run(t)
            end_effectors3_d.append(ref_X.flatten())
            
            # Update uncertainties
            self.dynamics.uncertainty(t)
            
            # Run controller (augments ref to 3D internally)
            q, q_dot, q_ddot, taw_motor, work = self.controller.run(
                q.astype(np.float64),
                q_dot.astype(np.float64),
                ref_X,
                ref_X_dot,
                ref_X_ddot,
                dt
            )
            
            # Store data
            theta1_traj.append(q[0, 0] % (2 * np.pi))
            theta2_traj.append(q[1, 0])
            theta3_traj.append(q[2, 0])
            controller_input.append(taw_motor.flatten())
            work_done.append(work)
            
            # Compute end-effector positions for each link
            end_effector1 = np.array([
                [self.l1 * np.cos(q[0, 0])],
                [self.l1 * np.sin(q[0, 0])]
            ])
            end_effector2 = np.array([
                [self.l1 * np.cos(q[0, 0]) + self.l2 * np.cos(q[0, 0] + q[1, 0])],
                [self.l1 * np.sin(q[0, 0]) + self.l2 * np.sin(q[0, 0] + q[1, 0])]
            ])
            end_effector3 = np.array([
                [self.l1 * np.cos(q[0, 0]) + self.l2 * np.cos(q[0, 0] + q[1, 0]) + 
                 self.l3 * np.cos(q[0, 0] + q[1, 0] + q[2, 0])],
                [self.l1 * np.sin(q[0, 0]) + self.l2 * np.sin(q[0, 0] + q[1, 0]) + 
                 self.l3 * np.sin(q[0, 0] + q[1, 0] + q[2, 0])]
            ])
            
            end_effectors1.append(end_effector1.flatten())
            end_effectors2.append(end_effector2.flatten())
            end_effectors3.append(end_effector3.flatten())
            
            # Compute error (only for x, y position)
            err = end_effector3 - ref_X
            error.append(np.linalg.norm(err))
            
            # Update performance metrics (augment ref_X to 3D for metric calculation)
            ref_X_3d = np.append(ref_X, [[np.pi / 3]], axis=0)
            X_3d = np.append(end_effector3, [[q[2, 0]]], axis=0)
            self.performance_metric(X_3d, ref_X_3d, t)
        
        # Compute summary metrics
        avg_error = np.mean(error)
        max_error = np.max(error)
        total_work = np.sum(work_done)
        
        return {
            'theta1_traj': np.array(theta1_traj),
            'theta2_traj': np.array(theta2_traj),
            'theta3_traj': np.array(theta3_traj),
            'end_effectors1': np.array(end_effectors1),
            'end_effectors2': np.array(end_effectors2),
            'end_effectors3': np.array(end_effectors3),
            'end_effectors3_d': np.array(end_effectors3_d),
            'controller_input': np.array(controller_input),
            'work_done': work_done,
            'error': error,
            'avg_error': avg_error,
            'max_error': max_error,
            'total_work': total_work,
            'rise_time1': self.rise_time1,
            'rise_time2': self.rise_time2,
            'overshoot1': self.overshoot1,
            'overshoot2': self.overshoot2,
            'settle_time1': self.settle_time1,
            'settle_time2': self.settle_time2
        }
    
    def run(self, timesteps, dt):
        """
        Run the complete simulation with multiple trials.
        
        Args:
            timesteps: Array of timesteps
            dt: Time step size
            
        Returns:
            List of results from each trial
        """
        results = []
        
        for trial in range(self.sim_num_trial):
            print(f"Running trial {trial + 1}/{self.sim_num_trial}...")
            result = self.run_sim(timesteps, dt)
            results.append(result)
            
            # Store cumulative data
            self.theta1_traj.append(result['theta1_traj'])
            self.theta2_traj.append(result['theta2_traj'])
            self.theta3_traj.append(result['theta3_traj'])
            self.end_effectors3.append(result['end_effectors3'])
            self.end_effectors3_d.append(result['end_effectors3_d'])
            self.error.append(result['error'])
        
        return results
    
    def get_summary_metrics(self, results):
        """
        Compute summary metrics across all trials.
        
        Args:
            results: List of trial results
            
        Returns:
            Dictionary of summary metrics
        """
        avg_errors = [r['avg_error'] for r in results]
        max_errors = [r['max_error'] for r in results]
        total_works = [r['total_work'] for r in results]
        
        return {
            'mean_avg_error': np.mean(avg_errors),
            'mean_max_error': np.mean(max_errors),
            'mean_total_work': np.mean(total_works),
            'std_avg_error': np.std(avg_errors),
            'std_max_error': np.std(max_errors),
            'std_total_work': np.std(total_works)
        }

