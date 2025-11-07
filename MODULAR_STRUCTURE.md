# Modular Structure Documentation

This document describes the modular Python codebase created from the original Jupyter notebooks (`self_tunning.ipynb` and `self_tunning_3link.ipynb`).

## Overview

The original notebooks have been refactored into a clean, modular Python package with proper separation of concerns, reusable components, and professional software engineering practices.

## Package Structure

```
LLM-Controller/
├── llm_controller/                 # Main package
│   ├── __init__.py                # Package initialization
│   │
│   ├── core/                      # Core functionality
│   │   ├── __init__.py
│   │   ├── reference_signal.py    # RefSigGen, NewRefSigGen classes
│   │   └── llm_chain.py          # LLMChain class for LLM integration
│   │
│   ├── dynamics/                  # Dynamics models
│   │   ├── __init__.py
│   │   ├── dynamics_2link.py     # Dynamics class for 2-link manipulator
│   │   └── dynamics_3link.py     # Dynamics3Link class for 3-link manipulator
│   │
│   ├── controllers/               # Controller implementations
│   │   ├── __init__.py
│   │   ├── controller_2link.py   # Controller class for 2-link
│   │   └── controller_3link.py   # Controller3Link class for 3-link
│   │
│   ├── runners/                   # Simulation runners
│   │   ├── __init__.py
│   │   ├── runner_2link.py       # Runner class for 2-link simulations
│   │   └── runner_3link.py       # Runner3Link class for 3-link simulations
│   │
│   ├── visualization/             # Visualization and animation
│   │   ├── __init__.py
│   │   └── animator.py           # Animation and plotting functions
│   │
│   ├── config/                    # Configuration utilities
│   │   ├── __init__.py
│   │   ├── llm_config.py         # LLM setup and configuration
│   │   └── system_params.py      # System parameters configuration
│   │
│   └── utils/                     # Utility functions
│       └── __init__.py
│
├── examples/                      # Example scripts
│   ├── run_2link_simulation.py   # 2-link simulation example
│   └── run_3link_simulation.py   # 3-link simulation example
│
├── setup.py                       # Package installation script
├── requirements.txt               # Python dependencies
├── README.md                      # Main documentation
└── MODULAR_STRUCTURE.md          # This file
```

## Module Descriptions

### 1. Core Modules (`llm_controller/core/`)

#### `reference_signal.py`
- **RefSigGen**: Generates reference signals and their derivatives using symbolic differentiation
- **NewRefSigGen**: Generates adapted reference signals for changed trajectories
- **Key Features**:
  - Symbolic computation with SymPy
  - Automatic derivative calculation
  - Trajectory history tracking

#### `llm_chain.py`
- **LLMChain**: Manages LLM interactions for controller tuning
- **Key Features**:
  - Pattern-based response parsing
  - Conversation history management
  - Automatic retry logic for malformed responses
  - Gain extraction and validation

### 2. Dynamics Modules (`llm_controller/dynamics/`)

#### `dynamics_2link.py`
- **Dynamics**: Complete dynamics model for 2-link manipulator
- **Key Features**:
  - Mass/inertia matrices
  - Coriolis and centrifugal forces
  - Gravity compensation
  - Friction modeling (viscous + Coulomb)
  - Uncertainty handling
  - Task-space transformation via Jacobian

#### `dynamics_3link.py`
- **Dynamics3Link**: Extended dynamics for 3-link manipulator
- **Key Features**:
  - 3x3 dynamics matrices
  - Extended Jacobian for 3D task space
  - Orientation control
  - All features from 2-link plus third link

### 3. Controller Modules (`llm_controller/controllers/`)

#### `controller_2link.py`
- **Controller**: Adaptive sliding mode controller for 2-link
- **Key Features**:
  - PID-based sliding surface
  - Adaptive uncertainty estimation (d_hat, kesi_hat)
  - Computed torque method
  - Torque saturation
  - Task-space control

#### `controller_3link.py`
- **Controller3Link**: Extended controller for 3-link
- **Key Features**:
  - 3D task space control (x, y, orientation)
  - Extended adaptive estimation
  - All features from 2-link controller

### 4. Runner Modules (`llm_controller/runners/`)

#### `runner_2link.py`
- **Runner**: Orchestrates 2-link simulations
- **Key Features**:
  - Multi-trial simulation management
  - Performance metric tracking (rise time, overshoot, settling time)
  - Data collection and storage
  - Summary statistics computation

#### `runner_3link.py`
- **Runner3Link**: Extended runner for 3-link
- **Key Features**:
  - 3-link specific metrics
  - Extended trajectory tracking
  - All features from 2-link runner

### 5. Visualization Module (`llm_controller/visualization/`)

#### `animator.py`
- **Functions**:
  - `create_animation()`: Generate 2-link animations
  - `create_animation_3link()`: Generate 3-link animations
  - `plot_results()`: Create static result plots
- **Key Features**:
  - Workspace visualization
  - Trajectory comparison (desired vs actual)
  - Error plotting
  - Control input visualization
  - Animation export (MP4)

### 6. Configuration Modules (`llm_controller/config/`)

#### `llm_config.py`
- **Functions**:
  - `get_llm_instance()`: Create LLM instance
  - `create_prompt()`: Generate prompt templates
  - `create_chat_history()`: Initialize chat history
- **Key Features**:
  - Environment variable support
  - Customizable prompts
  - Multiple LLM model support

#### `system_params.py`
- **SystemParams**: Centralized parameter configuration
- **Key Features**:
  - Default parameters for 2-link and 3-link
  - Easy parameter modification
  - Reference signal configuration
  - Uncertainty configuration
  - Timestep generation

## Key Improvements Over Notebooks

### 1. Modularity
- **Before**: All code in single notebook cells
- **After**: Separated into logical modules with clear responsibilities

### 2. Reusability
- **Before**: Code duplication between 2-link and 3-link notebooks
- **After**: Shared base functionality, easy to extend

### 3. Maintainability
- **Before**: Difficult to modify and test
- **After**: Each module can be tested and modified independently

### 4. Documentation
- **Before**: Minimal inline comments
- **After**: Comprehensive docstrings, type hints, and documentation

### 5. Configuration
- **Before**: Hardcoded parameters scattered throughout
- **After**: Centralized configuration system

### 6. Professional Structure
- **Before**: Notebook-style code
- **After**: Production-ready Python package with setup.py

## Usage Examples

### Basic 2-Link Simulation

```python
from llm_controller.config.system_params import SystemParams
from llm_controller.runners.runner_2link import Runner

# Initialize
params = SystemParams(system_type='2link')

# Create runner
runner = Runner(
    llm=None,
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

# Run
timesteps = params.get_timesteps()
results = runner.run(timesteps, params.dt)
```

### With LLM Tuning

```python
from llm_controller.config.llm_config import get_llm_instance, create_prompt, create_chat_history
from llm_controller.core.llm_chain import LLMChain

# Setup LLM
llm = get_llm_instance()
prompt = create_prompt()
chat_history = create_chat_history()
llm_chain = LLMChain(llm, prompt, chat_history)

# Use in runner
runner = Runner(
    llm=llm_chain,
    prompt=prompt,
    # ... other parameters ...
    update_gains=True  # Enable LLM tuning
)
```

### Custom Configuration

```python
import sympy as sp
from llm_controller.config.system_params import SystemParams

params = SystemParams(system_type='2link')

# Custom reference signal
t = sp.Symbol('t')
params.set_reference_signal([
    0.6 * sp.cos(1.5 * t),
    0.6 * sp.sin(1.5 * t)
])

# Custom gains
params.update_gains(k_p=60.0, k_d=35.0, k_i=25.0, landa_1=0.6, landa_2=0.6)

# Custom uncertainties
disturbances = [0.2 * sp.sin(3*t), 0.2 * sp.cos(3*t)]
unmodeled = [0.1 * t, 0.1 * t]
params.set_uncertainty(disturbances, unmodeled)
```

## Installation and Usage

### Installation

```bash
# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Running Examples

```bash
# 2-link simulation
python examples/run_2link_simulation.py

# 3-link simulation
python examples/run_3link_simulation.py

# Or using console scripts (after installation)
llm-controller-2link
llm-controller-3link
```

## Testing

The modular structure makes it easy to add unit tests:

```python
# Example test structure
import pytest
from llm_controller.dynamics.dynamics_2link import Dynamics

def test_dynamics_initialization():
    uncertainty = ([0, 0], [0, 0])
    dynamics = Dynamics(uncertainty)
    assert dynamics.m1 == 4.0
    assert dynamics.m2 == 2.5
    # ... more tests
```

## Extension Guide

### Adding a New Manipulator Type

1. Create `dynamics_Nlink.py` in `llm_controller/dynamics/`
2. Create `controller_Nlink.py` in `llm_controller/controllers/`
3. Create `runner_Nlink.py` in `llm_controller/runners/`
4. Add animation function in `llm_controller/visualization/animator.py`
5. Create example script in `examples/`

### Adding New Control Strategies

1. Create new controller class inheriting from base controller
2. Override `run()` method with new control law
3. Add to `llm_controller/controllers/` module

### Adding New Performance Metrics

1. Add metric calculation in `Runner.performance_metric()`
2. Update `get_summary_metrics()` to include new metrics
3. Update visualization to display new metrics

## Comparison: Notebooks vs Modular Code

| Aspect | Notebooks | Modular Code |
|--------|-----------|--------------|
| Structure | Single file | Multiple organized modules |
| Reusability | Low | High |
| Testing | Difficult | Easy |
| Version Control | Poor | Excellent |
| Collaboration | Challenging | Straightforward |
| Documentation | Inline only | Comprehensive docstrings |
| Deployment | Manual | Package installation |
| Maintenance | Difficult | Easy |
| Extensibility | Limited | Designed for extension |

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- NumPy: Numerical computations
- SymPy: Symbolic mathematics
- Matplotlib: Visualization
- LangChain: LLM integration
- OpenAI: GPT models

## Notes

- Original notebooks remain unchanged and functional
- All functionality from notebooks is preserved
- Code is backward compatible with notebook usage patterns
- Additional features added (better error handling, logging, etc.)

## Future Enhancements

Potential improvements:
1. Add unit tests and integration tests
2. Add type checking with mypy
3. Add CI/CD pipeline
4. Add more controller types (MPC, LQR, etc.)
5. Add real-time visualization
6. Add ROS integration
7. Add hardware interface support

## Conclusion

This modular structure provides a professional, maintainable, and extensible framework for robotic manipulator control research. The code is production-ready while maintaining all the functionality of the original notebooks.

