"""
PDE Fourier Simulator
A comprehensive Python application for simulating and visualizing solutions 
to the Wave Equation and Heat Diffusion Equation using Fourier series.
"""

__version__ = "1.0.0"
__author__ = "Mohamad Mahdy Sobhany Poor"
__email__ = "mahdyyyyy03@gmail.com"
__license__ = "MIT"

# Make main classes easily accessible
from .physics.wave_equation import VibratingString
from .physics.heat_equation import HeatDiffusion
from .gui.main_window import SimulationGUI

__all__ = [
    'VibratingString',
    'HeatDiffusion', 
    'SimulationGUI',
    '__version__',
    '__author__'
]
