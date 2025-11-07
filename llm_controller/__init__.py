"""
LLM Controller Package
A modular system for LLM-based robotic manipulator control with self-tuning capabilities.
"""

__version__ = "1.0.0"
__author__ = "LLM Controller Team"

from llm_controller.core.reference_signal import RefSigGen, NewRefSigGen
from llm_controller.core.llm_chain import LLMChain
from llm_controller.dynamics.dynamics_2link import Dynamics
from llm_controller.dynamics.dynamics_3link import Dynamics3Link
from llm_controller.controllers.controller_2link import Controller
from llm_controller.controllers.controller_3link import Controller3Link
from llm_controller.runners.runner_2link import Runner
from llm_controller.runners.runner_3link import Runner3Link

__all__ = [
    'RefSigGen',
    'NewRefSigGen',
    'LLMChain',
    'Dynamics',
    'Dynamics3Link',
    'Controller',
    'Controller3Link',
    'Runner',
    'Runner3Link',
]

