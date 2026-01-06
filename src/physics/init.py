"""
Physics module for PDE solvers
Contains implementations of wave and heat equation solvers using Fourier series
"""

from .wave_equation import VibratingString
from .heat_equation import HeatDiffusion
from .base_equation import BasePDESolver

__all__ = [
    'VibratingString',
    'HeatDiffusion',
    'BasePDESolver'
]

# Module metadata
__module_name__ = "Physics Solvers"
__description__ = "Fourier series based PDE solvers for wave and heat equations"
