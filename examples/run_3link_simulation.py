#!/usr/bin/env python3
"""
Example script for running 3-link manipulator simulation.

This script demonstrates how to use the LLM Controller package to simulate
a 3-link robotic manipulator with adaptive control.
"""

import numpy as np
import sympy as sp
from llm_controller.config.llm_config import get_llm_instance, create_prompt, create_chat_history
from llm_controller.config.system_params import SystemParams
from llm_controller.core.llm_chain import LLMChain
from llm_controller.runners.runner_3link import Runner3Link
from llm_controller.visualization.animator import plot_results, create_animation_3link


def main():
    """Run 3-link manipulator simulation."""
    print("="*80)
    print("3-Link Manipulator Simulation with LLM-Based Controller Tuning")
    print("="*80)
    
    # Initialize system parameters
    print("\n[1/5] Initializing system parameters...")
    params = SystemParams(system_type='3link')
    
    # Set up LLM (optional - comment out if you don't have an API key)
    print("[2/5] Setting up LLM (optional)...")
    try:
        llm = get_llm_instance()  # Reads from OPENAI_API_KEY environment variable
        prompt = create_prompt()
        chat_history = create_chat_history()
        llm_chain = LLMChain(llm, prompt, chat_history)
        print("  ✓ LLM initialized successfully")
    except Exception as e:
        print(f"  ⚠ Could not initialize LLM: {e}")
        print("  → Continuing without LLM tuning...")
        llm_chain = None
    
    # Create runner
    print("[3/5] Creating simulation runner...")
    runner = Runner3Link(
        llm=llm_chain,
        prompt=None,
        q_init=params.q_init,
        q_dot_init=params.q_dot_init,
        current_ref_sig=params.current_ref_sig,
        new_ref_sig=params.new_ref_sig,
        current_uncertainty=params.current_uncertainty,
        new_uncertainty=params.new_uncertainty,
        gains=params.gains,
        update_gains=False,  # Set to True to enable LLM tuning
        sim_num_trial=3,
        attempt_num_trial=1,
        output_dir='./output_3link'
    )
    print(f"  ✓ Runner created with gains: k_p={params.gains[0]}, k_d={params.gains[1]}, "
          f"k_i={params.gains[2]}, λ1={params.gains[3]}, λ2={params.gains[4]}")
    
    # Run simulation
    print("[4/5] Running simulation...")
    timesteps = params.get_timesteps()
    results = runner.run(timesteps, params.dt)
    
    # Get summary metrics
    summary = runner.get_summary_metrics(results)
    print("\n  Simulation Results:")
    print(f"    Mean Average Error: {summary['mean_avg_error']:.6f} m")
    print(f"    Mean Max Error:     {summary['mean_max_error']:.6f} m")
    print(f"    Mean Total Work:    {summary['mean_total_work']:.6f} J")
    
    # Visualize results
    print("[5/5] Creating visualizations...")
    plot_results(results[0], output_dir='./output_3link')
    print("  ✓ Static plots saved to ./output_3link/results.png")
    
    # Optionally create animation (requires ffmpeg)
    try:
        print("  Creating animation (this may take a moment)...")
        create_animation_3link(results[0], runner.l1, runner.l2, runner.l3,
                              output_path='./output_3link/animation.mp4', fps=30)
    except Exception as e:
        print(f"  ⚠ Could not create animation: {e}")
        print("  → Install ffmpeg to enable animation export")
    
    print("\n" + "="*80)
    print("Simulation complete!")
    print("="*80)


if __name__ == '__main__':
    main()

