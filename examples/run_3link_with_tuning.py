#!/usr/bin/env python3
"""
Example script for running 3-link manipulator simulation WITH LLM TUNING.

This script demonstrates the complete LLM-based controller tuning process,
where the LLM iteratively adjusts gains based on performance metrics for
a 3-link robotic manipulator.
"""

import numpy as np
import sympy as sp
from llm_controller.config.llm_config import get_llm_instance, create_prompt, create_chat_history
from llm_controller.config.system_params import SystemParams
from llm_controller.core.llm_chain import LLMChain
from llm_controller.runners.runner_3link import Runner3Link
from llm_controller.visualization.animator import plot_results, create_animation_3link


# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_TUNING_ATTEMPTS = 5  # Maximum number of tuning iterations
SIM_TRIALS_PER_ATTEMPT = 2  # Number of simulation trials per tuning iteration
OUTPUT_DIR = './output_3link_tuning'  # Output directory for results

# Optional: Override initial gains (set to None to use defaults from SystemParams)
# Format: [k_p, k_d, k_i, landa_1, landa_2]
INITIAL_GAINS = None  # Use None for default, or e.g., [15.0, 8.0, 3.0, 0.15, 0.15] for custom
# ============================================================================


def main():
    """Run 3-link manipulator simulation with LLM tuning."""
    print("="*80)
    print("3-Link Manipulator Simulation with LLM-Based Controller Tuning")
    print("="*80)
    
    # Initialize system parameters
    print("\n[1/6] Initializing system parameters...")
    params = SystemParams(system_type='3link')
    
    # Use custom initial gains if specified
    if INITIAL_GAINS is not None:
        params.gains = INITIAL_GAINS
        print(f"  Using custom initial gains: {INITIAL_GAINS}")
    
    # Set up LLM
    print("[2/6] Setting up LLM...")
    try:
        llm = get_llm_instance()
        prompt = create_prompt()
        chat_history = create_chat_history()
        llm_chain = LLMChain(llm, prompt, chat_history)
        print("  ✓ LLM initialized successfully")
    except Exception as e:
        print(f"  ✗ Could not initialize LLM: {e}")
        print("  → LLM tuning requires an API key. Exiting...")
        return
    
    print(f"[3/6] Starting LLM tuning loop (max {MAX_TUNING_ATTEMPTS} attempts)...")
    print(f"      Running {SIM_TRIALS_PER_ATTEMPT} trials per attempt\n")
    
    # Initial gains
    current_gains = params.gains.copy()
    best_gains = current_gains.copy()
    best_error = float('inf')
    
    tuning_history = []
    
    # LLM Tuning Loop
    for attempt in range(MAX_TUNING_ATTEMPTS):
        print(f"\n{'='*80}")
        print(f"TUNING ATTEMPT {attempt + 1}/{MAX_TUNING_ATTEMPTS}")
        print(f"{'='*80}")
        print(f"Current gains: k_p={current_gains[0]:.2f}, k_d={current_gains[1]:.2f}, "
              f"k_i={current_gains[2]:.2f}, λ1={current_gains[3]:.2f}, λ2={current_gains[4]:.2f}")
        
        # Create runner with current gains
        runner = Runner3Link(
            llm=llm_chain,
            prompt=None,
            q_init=params.q_init,
            q_dot_init=params.q_dot_init,
            current_ref_sig=params.current_ref_sig,
            new_ref_sig=params.new_ref_sig,
            current_uncertainty=params.current_uncertainty,
            new_uncertainty=params.new_uncertainty,
            gains=current_gains,
            update_gains=False,
            sim_num_trial=SIM_TRIALS_PER_ATTEMPT,
            attempt_num_trial=1,
            output_dir=f'{OUTPUT_DIR}/attempt_{attempt+1}'
        )
        
        # Run simulation
        print(f"\nRunning {SIM_TRIALS_PER_ATTEMPT} simulation trials...")
        timesteps = params.get_timesteps()
        results = runner.run(timesteps, params.dt)
        
        # Save visualizations for this attempt
        attempt_dir = f'{OUTPUT_DIR}/attempt_{attempt+1}'
        plot_results(results[0], output_dir=attempt_dir)
        print(f"  Results saved to {attempt_dir}/results.png")
        try:
            create_animation_3link(results[0], runner.l1, runner.l2, runner.l3,
                           output_path=f'{attempt_dir}/animation.mp4', fps=30)
            print(f"  Animation saved to {attempt_dir}/animation.mp4")
        except Exception as e:
            print(f"  ⚠ Could not create animation: {e}")
        
        # Get performance metrics
        summary = runner.get_summary_metrics(results)
        mean_avg_error = summary['mean_avg_error']
        mean_max_error = summary['mean_max_error']
        mean_total_work = summary['mean_total_work']
        
        print(f"\nPerformance Metrics:")
        print(f"  Mean Average Error: {mean_avg_error:.6f} m")
        print(f"  Mean Max Error:     {mean_max_error:.6f} m")
        print(f"  Mean Total Work:    {mean_total_work:.6f} J")
        
        # Track best performance
        if mean_avg_error < best_error:
            best_error = mean_avg_error
            best_gains = current_gains.copy()
            print(f"  ✓ New best performance!")
        
        # Store history
        tuning_history.append({
            'attempt': attempt + 1,
            'gains': current_gains.copy(),
            'mean_avg_error': mean_avg_error,
            'mean_max_error': mean_max_error,
            'mean_total_work': mean_total_work
        })
        
        # Ask LLM if performance is satisfactory
        satisfier_prompt = f"""
The 3-link manipulator controller has been tested with the following performance:
- Mean Average Tracking Error: {mean_avg_error:.6f} m
- Mean Maximum Error: {mean_max_error:.6f} m  
- Mean Total Work: {mean_total_work:.6f} J

Current controller gains:
- k_p = {current_gains[0]:.2f}
- k_d = {current_gains[1]:.2f}
- k_i = {current_gains[2]:.2f}
- landa_1 = {current_gains[3]:.2f}
- landa_2 = {current_gains[4]:.2f}

Is this performance satisfactory for a 3-link robotic manipulator tracking task? 
Consider that lower tracking error is better, and the system should be stable.
"""
        
        print("\n[LLM] Evaluating performance...")
        is_satisfied, llm_evaluation = llm_chain.run_satisfier(satisfier_prompt)
        print(f"[LLM] Response: {llm_evaluation[:200]}...")
        print(f"[LLM] Satisfied: {is_satisfied}")
        
        if is_satisfied:
            print(f"\n✓ LLM is satisfied with current performance!")
            print(f"  Stopping tuning at attempt {attempt + 1}")
            break
        
        # If not satisfied and not last attempt, ask LLM for new gains
        if attempt < MAX_TUNING_ATTEMPTS - 1:
            helper_prompt = f"""
The current controller performance is not satisfactory. Here are the metrics:
- Mean Average Tracking Error: {mean_avg_error:.6f} m
- Mean Maximum Error: {mean_max_error:.6f} m
- Mean Total Work: {mean_total_work:.6f} J

Current controller gains:
- k_p = {current_gains[0]:.2f}
- k_d = {current_gains[1]:.2f}
- k_i = {current_gains[2]:.2f}
- landa_1 = {current_gains[3]:.2f}
- landa_2 = {current_gains[4]:.2f}

Please suggest new controller gains to improve performance for this 3-link manipulator. 
The tracking error is high, so we need better control.
Remember: higher gains give faster response but may cause instability.
For 3-link systems, we typically need slightly higher gains than 2-link systems due to increased complexity.
"""
            
            print("\n[LLM] Requesting new controller gains...")
            k_p, k_d, k_i, landa_1, landa_2, llm_suggestion = llm_chain.run_helper(helper_prompt)
            print(f"[LLM] Response: {llm_suggestion[:200]}...")
            print(f"[LLM] Suggested gains: k_p={k_p:.2f}, k_d={k_d:.2f}, k_i={k_i:.2f}, "
                  f"λ1={landa_1:.2f}, λ2={landa_2:.2f}")
            
            # Update gains for next iteration
            current_gains = [k_p, k_d, k_i, landa_1, landa_2]
    
    # Final summary
    print(f"\n\n{'='*80}")
    print("TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"\nTuning History:")
    for entry in tuning_history:
        print(f"  Attempt {entry['attempt']}: "
              f"Error={entry['mean_avg_error']:.6f} m, "
              f"Gains=[{entry['gains'][0]:.1f}, {entry['gains'][1]:.1f}, "
              f"{entry['gains'][2]:.1f}, {entry['gains'][3]:.2f}, {entry['gains'][4]:.2f}]")
    
    print(f"\nBest Performance:")
    print(f"  Average Error: {best_error:.6f} m")
    print(f"  Best Gains: k_p={best_gains[0]:.2f}, k_d={best_gains[1]:.2f}, "
          f"k_i={best_gains[2]:.2f}, λ1={best_gains[3]:.2f}, λ2={best_gains[4]:.2f}")
    
    # Run final simulation with best gains and create visualizations
    print(f"\n[4/6] Running final simulation with best gains...")
    final_runner = Runner3Link(
        llm=llm_chain,
        prompt=None,
        q_init=params.q_init,
        q_dot_init=params.q_dot_init,
        current_ref_sig=params.current_ref_sig,
        new_ref_sig=params.new_ref_sig,
        current_uncertainty=params.current_uncertainty,
        new_uncertainty=params.new_uncertainty,
        gains=best_gains,
        update_gains=False,
        sim_num_trial=1,
        attempt_num_trial=1,
        output_dir=f'{OUTPUT_DIR}/final'
    )
    
    final_results = final_runner.run(timesteps, params.dt)
    
    # Create visualizations
    print("[5/6] Creating visualizations...")
    plot_results(final_results[0], output_dir=f'{OUTPUT_DIR}/final')
    print(f"  ✓ Static plots saved to {OUTPUT_DIR}/final/results.png")
    
    print("[6/6] Creating animation...")
    try:
        create_animation_3link(final_results[0], final_runner.l1, final_runner.l2, final_runner.l3,
                        output_path=f'{OUTPUT_DIR}/final/animation.mp4', fps=30)
        print(f"  ✓ Animation saved to {OUTPUT_DIR}/final/animation.mp4")
    except Exception as e:
        print(f"  ⚠ Could not create animation: {e}")
    
    print(f"\n{'='*80}")
    print("SIMULATION AND TUNING COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

