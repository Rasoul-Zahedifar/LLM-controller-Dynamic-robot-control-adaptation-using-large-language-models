"""
Animation and Visualization Module

This module provides utilities for creating animations and visualizations
of the robotic manipulator simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_animation(results, l1, l2, output_path='animation_2link.mp4', fps=30):
    """
    Create an animation for 2-link manipulator simulation.
    
    Args:
        results: Simulation results dictionary
        l1: Length of link 1
        l2: Length of link 2
        output_path: Path to save animation file
        fps: Frames per second
        
    Returns:
        matplotlib FuncAnimation object
    """
    theta1_traj = results['theta1_traj']
    theta2_traj = results['theta2_traj']
    end_effectors2 = results['end_effectors2']
    end_effectors2_d = results['end_effectors2_d']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Setup workspace plot
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title('2-Link Manipulator Workspace')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Setup error plot
    ax2.set_xlim(0, len(theta1_traj))
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.set_title('Tracking Error')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Error (m)')
    
    # Initialize plot elements
    line_link1, = ax1.plot([], [], 'b-', linewidth=3, label='Link 1')
    line_link2, = ax1.plot([], [], 'r-', linewidth=3, label='Link 2')
    point_joint1, = ax1.plot([], [], 'ko', markersize=8)
    point_joint2, = ax1.plot([], [], 'ko', markersize=8)
    point_ee, = ax1.plot([], [], 'go', markersize=10, label='End Effector')
    traj_desired, = ax1.plot([], [], 'b--', alpha=0.5, label='Desired')
    traj_actual, = ax1.plot([], [], 'r-', alpha=0.5, label='Actual')
    
    error_line, = ax2.plot([], [], 'r-')
    
    ax1.legend()
    
    def init():
        line_link1.set_data([], [])
        line_link2.set_data([], [])
        point_joint1.set_data([], [])
        point_joint2.set_data([], [])
        point_ee.set_data([], [])
        traj_desired.set_data([], [])
        traj_actual.set_data([], [])
        error_line.set_data([], [])
        return (line_link1, line_link2, point_joint1, point_joint2, 
                point_ee, traj_desired, traj_actual, error_line)
    
    def animate(frame):
        # Compute joint positions
        x1 = l1 * np.cos(theta1_traj[frame])
        y1 = l1 * np.sin(theta1_traj[frame])
        x2 = x1 + l2 * np.cos(theta1_traj[frame] + theta2_traj[frame])
        y2 = y1 + l2 * np.sin(theta1_traj[frame] + theta2_traj[frame])
        
        # Update links
        line_link1.set_data([0, x1], [0, y1])
        line_link2.set_data([x1, x2], [y1, y2])
        
        # Update joints
        point_joint1.set_data([x1], [y1])
        point_joint2.set_data([x2], [y2])
        point_ee.set_data([x2], [y2])
        
        # Update trajectories
        traj_desired.set_data(end_effectors2_d[:frame+1, 0], end_effectors2_d[:frame+1, 1])
        traj_actual.set_data(end_effectors2[:frame+1, 0], end_effectors2[:frame+1, 1])
        
        # Update error plot
        errors = np.linalg.norm(end_effectors2[:frame+1] - end_effectors2_d[:frame+1], axis=1)
        error_line.set_data(range(frame+1), errors)
        
        return (line_link1, line_link2, point_joint1, point_joint2, 
                point_ee, traj_desired, traj_actual, error_line)
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(theta1_traj),
                        interval=1000/fps, blit=True, repeat=True)
    
    # Save animation if path provided
    if output_path:
        try:
            anim.save(output_path, writer='ffmpeg', fps=fps)
            print(f"Animation saved to {output_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")
            print("Displaying animation instead...")
            plt.show()
    
    return anim


def create_animation_3link(results, l1, l2, l3, output_path='animation_3link.mp4', fps=30):
    """
    Create an animation for 3-link manipulator simulation.
    
    Args:
        results: Simulation results dictionary
        l1: Length of link 1
        l2: Length of link 2
        l3: Length of link 3
        output_path: Path to save animation file
        fps: Frames per second
        
    Returns:
        matplotlib FuncAnimation object
    """
    theta1_traj = results['theta1_traj']
    theta2_traj = results['theta2_traj']
    theta3_traj = results['theta3_traj']
    end_effectors3 = results['end_effectors3']
    end_effectors3_d = results['end_effectors3_d']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Setup workspace plot
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title('3-Link Manipulator Workspace')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Setup error plot
    ax2.set_xlim(0, len(theta1_traj))
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.set_title('Tracking Error')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Error (m)')
    
    # Initialize plot elements
    line_link1, = ax1.plot([], [], 'b-', linewidth=3, label='Link 1')
    line_link2, = ax1.plot([], [], 'r-', linewidth=3, label='Link 2')
    line_link3, = ax1.plot([], [], 'g-', linewidth=3, label='Link 3')
    point_joint1, = ax1.plot([], [], 'ko', markersize=8)
    point_joint2, = ax1.plot([], [], 'ko', markersize=8)
    point_joint3, = ax1.plot([], [], 'ko', markersize=8)
    point_ee, = ax1.plot([], [], 'mo', markersize=10, label='End Effector')
    traj_desired, = ax1.plot([], [], 'b--', alpha=0.5, label='Desired')
    traj_actual, = ax1.plot([], [], 'r-', alpha=0.5, label='Actual')
    
    error_line, = ax2.plot([], [], 'r-')
    
    ax1.legend()
    
    def init():
        line_link1.set_data([], [])
        line_link2.set_data([], [])
        line_link3.set_data([], [])
        point_joint1.set_data([], [])
        point_joint2.set_data([], [])
        point_joint3.set_data([], [])
        point_ee.set_data([], [])
        traj_desired.set_data([], [])
        traj_actual.set_data([], [])
        error_line.set_data([], [])
        return (line_link1, line_link2, line_link3, point_joint1, point_joint2, 
                point_joint3, point_ee, traj_desired, traj_actual, error_line)
    
    def animate(frame):
        # Compute joint positions
        x1 = l1 * np.cos(theta1_traj[frame])
        y1 = l1 * np.sin(theta1_traj[frame])
        x2 = x1 + l2 * np.cos(theta1_traj[frame] + theta2_traj[frame])
        y2 = y1 + l2 * np.sin(theta1_traj[frame] + theta2_traj[frame])
        x3 = x2 + l3 * np.cos(theta1_traj[frame] + theta2_traj[frame] + theta3_traj[frame])
        y3 = y2 + l3 * np.sin(theta1_traj[frame] + theta2_traj[frame] + theta3_traj[frame])
        
        # Update links
        line_link1.set_data([0, x1], [0, y1])
        line_link2.set_data([x1, x2], [y1, y2])
        line_link3.set_data([x2, x3], [y2, y3])
        
        # Update joints
        point_joint1.set_data([x1], [y1])
        point_joint2.set_data([x2], [y2])
        point_joint3.set_data([x3], [y3])
        point_ee.set_data([x3], [y3])
        
        # Update trajectories
        traj_desired.set_data(end_effectors3_d[:frame+1, 0], end_effectors3_d[:frame+1, 1])
        traj_actual.set_data(end_effectors3[:frame+1, 0], end_effectors3[:frame+1, 1])
        
        # Update error plot
        errors = np.linalg.norm(end_effectors3[:frame+1] - end_effectors3_d[:frame+1], axis=1)
        error_line.set_data(range(frame+1), errors)
        
        return (line_link1, line_link2, line_link3, point_joint1, point_joint2, 
                point_joint3, point_ee, traj_desired, traj_actual, error_line)
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(theta1_traj),
                        interval=1000/fps, blit=True, repeat=True)
    
    # Save animation if path provided
    if output_path:
        try:
            anim.save(output_path, writer='ffmpeg', fps=fps)
            print(f"Animation saved to {output_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")
            print("Displaying animation instead...")
            plt.show()
    
    return anim


def plot_results(results, output_dir='./output'):
    """
    Create static plots of simulation results.
    
    Args:
        results: Simulation results dictionary
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot trajectories
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Joint angles
    axes[0, 0].plot(results['theta1_traj'], label='θ1')
    axes[0, 0].plot(results['theta2_traj'], label='θ2')
    if 'theta3_traj' in results:
        axes[0, 0].plot(results['theta3_traj'], label='θ3')
    axes[0, 0].set_title('Joint Angles')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Angle (rad)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # End-effector trajectory
    ee_key = 'end_effectors3' if 'end_effectors3' in results else 'end_effectors2'
    ee_d_key = 'end_effectors3_d' if 'end_effectors3_d' in results else 'end_effectors2_d'
    
    axes[0, 1].plot(results[ee_key][:, 0], results[ee_key][:, 1], 'r-', label='Actual')
    axes[0, 1].plot(results[ee_d_key][:, 0], results[ee_d_key][:, 1], 'b--', label='Desired')
    axes[0, 1].set_title('End-Effector Trajectory')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].axis('equal')
    
    # Tracking error
    axes[1, 0].plot(results['error'])
    axes[1, 0].set_title('Tracking Error')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Error (m)')
    axes[1, 0].grid(True)
    
    # Control inputs
    axes[1, 1].plot(results['controller_input'])
    axes[1, 1].set_title('Control Torques')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Torque (Nm)')
    axes[1, 1].legend([f'τ{i+1}' for i in range(results['controller_input'].shape[1])])
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/results.png', dpi=150)
    print(f"Results plot saved to {output_dir}/results.png")
    plt.close()

