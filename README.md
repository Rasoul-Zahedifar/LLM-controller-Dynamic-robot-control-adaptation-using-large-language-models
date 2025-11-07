# LLM Controller

A modular Python package for LLM-based robotic manipulator control with self-tuning capabilities.

## Overview

This package provides a complete framework for simulating and controlling 2-link and 3-link robotic manipulators using adaptive sliding mode control with LLM-based parameter tuning. The system can automatically adjust controller gains based on performance metrics using large language models.

## Features

- **Modular Architecture**: Clean separation of dynamics, controllers, runners, and visualization
- **2-Link and 3-Link Support**: Complete implementations for both manipulator types
- **Adaptive Control**: Sliding mode control with uncertainty estimation
- **LLM Integration**: Automatic controller tuning using OpenAI's GPT models
- **Visualization**: Animation and plotting capabilities for simulation results
- **Flexible Configuration**: Easy-to-use configuration system for parameters

## Project Structure

```
LLM-Controller/
├── llm_controller/          # Main package
│   ├── core/               # Core modules (reference signals, LLM chain)
│   ├── dynamics/           # Dynamics models (2-link, 3-link)
│   ├── controllers/        # Controller implementations
│   ├── runners/            # Simulation runners
│   ├── visualization/      # Animation and plotting
│   ├── config/             # Configuration utilities
│   └── utils/              # Utility functions
├── examples/               # Example scripts
│   ├── run_2link_simulation.py
│   └── run_3link_simulation.py
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd LLM-Controller
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the package

```bash
pip install -e .
```

## Quick Start

### Running a 2-Link Simulation

```bash
python examples/run_2link_simulation.py
```

### Running a 3-Link Simulation

```bash
python examples/run_3link_simulation.py
```

### Basic Usage in Python

```python
import numpy as np
from llm_controller.config.system_params import SystemParams
from llm_controller.runners.runner_2link import Runner

# Initialize parameters
params = SystemParams(system_type='2link')

# Create runner
runner = Runner(
    llm=None,  # Optional: provide LLM chain for tuning
    prompt=None,
    q_init=params.q_init,
    q_dot_init=params.q_dot_init,
    current_ref_sig=params.current_ref_sig,
    new_ref_sig=params.new_ref_sig,
    current_uncertainty=params.current_uncertainty,
    new_uncertainty=params.new_uncertainty,
    gains=params.gains,
    update_gains=False,
    sim_num_trial=3,
    attempt_num_trial=1,
    output_dir='./output'
)

# Run simulation
timesteps = params.get_timesteps()
results = runner.run(timesteps, params.dt)

# Get metrics
summary = runner.get_summary_metrics(results)
print(f"Mean Average Error: {summary['mean_avg_error']:.6f} m")
```

## Configuration

### System Parameters

The `SystemParams` class provides easy configuration:

```python
from llm_controller.config.system_params import SystemParams

# Create parameters for 2-link system
params = SystemParams(system_type='2link')

# Modify controller gains
params.update_gains(k_p=50.0, k_d=30.0, k_i=20.0, landa_1=0.5, landa_2=0.5)

# Set custom reference signal
import sympy as sp
t = sp.Symbol('t')
params.set_reference_signal([0.5 * sp.cos(t), 0.5 * sp.sin(t)])
```

### LLM Configuration

To use LLM-based tuning, set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or in Python:

```python
from llm_controller.config.llm_config import get_llm_instance, create_prompt, create_chat_history
from llm_controller.core.llm_chain import LLMChain

llm = get_llm_instance(api_key='your-api-key-here')
prompt = create_prompt()
chat_history = create_chat_history()
llm_chain = LLMChain(llm, prompt, chat_history)
```

## Components

### Dynamics Models

- `Dynamics` (2-link): Mass-inertia matrices, Coriolis forces, gravity, friction
- `Dynamics3Link` (3-link): Extended dynamics for 3-link manipulator

### Controllers

- `Controller` (2-link): Adaptive sliding mode control with PID sliding surface
- `Controller3Link` (3-link): Extended controller for 3-link system

### Reference Signal Generators

- `RefSigGen`: Generates reference trajectories and derivatives
- `NewRefSigGen`: Generates adapted reference trajectories

### Runners

- `Runner` (2-link): Complete simulation orchestration
- `Runner3Link` (3-link): Extended runner for 3-link system

### Visualization

- `create_animation()`: Generate animated visualizations
- `plot_results()`: Create static result plots

## Advanced Usage

### Custom Reference Trajectories

```python
import sympy as sp

t = sp.Symbol('t')

# Circular trajectory
ref_sig = [0.5 * sp.cos(t), 0.5 * sp.sin(t)]

# Figure-eight trajectory
ref_sig = [sp.sin(t), sp.sin(2*t)/2]

# Custom trajectory
ref_sig = [0.3 * sp.cos(2*t) + 0.2, 0.3 * sp.sin(2*t)]
```

### Adding Uncertainties

```python
import sympy as sp

t = sp.Symbol('t')

# Disturbances and unmodeled dynamics
disturbances = [0.1 * sp.sin(2*t), 0.1 * sp.cos(2*t)]
unmodeled = [0.05 * t, 0.05 * t]

uncertainty = (disturbances, unmodeled)
```

### Performance Metrics

The runner tracks various performance metrics:

- Average tracking error
- Maximum tracking error
- Total work done
- Rise time
- Overshoot
- Settling time

## Requirements

- Python 3.8+
- NumPy
- SymPy
- Matplotlib
- LangChain
- OpenAI API (optional, for LLM tuning)
- FFmpeg (optional, for animation export)

## Troubleshooting

### Animation Export Issues

If you encounter issues exporting animations, ensure FFmpeg is installed:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### LLM Connection Issues

If LLM tuning fails:
1. Check that your OpenAI API key is set correctly
2. Verify internet connectivity
3. Check API quota and billing status

### Import Errors

If you encounter import errors:
1. Ensure the package is installed: `pip install -e .`
2. Check that you're in the correct virtual environment
3. Verify all dependencies are installed: `pip install -r requirements.txt`

## Original Notebooks

This modular codebase was extracted from the original Jupyter notebooks:
- `self_tunning.ipynb` - 2-link manipulator implementation
- `self_tunning_3link.ipynb` - 3-link manipulator implementation

The notebooks remain unchanged and can still be used independently.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{llm_controller,
  title = {LLM Controller: Adaptive Robotic Manipulator Control with LLM-Based Tuning},
  author = {LLM Controller Team},
  year = {2024},
  url = {https://github.com/yourusername/LLM-Controller}
}
```

## Contact

For questions or support, please open an issue on GitHub.
