"""
GUI module for PDE Fourier Simulator
Contains all graphical user interface components
"""

from .main_window import SimulationGUI
from .styles import COLORS, configure_styles

# These will be imported when the respective modules are created
try:
    from .wave_controls import WaveControls
    from .heat_controls import HeatControls
    from .plotting import PlottingUtils
    
    __all__ = [
        'SimulationGUI',
        'WaveControls',
        'HeatControls',
        'PlottingUtils',
        'COLORS',
        'configure_styles'
    ]
except ImportError:
    # If separate control modules don't exist yet
    __all__ = [
        'SimulationGUI',
        'COLORS',
        'configure_styles'
    ]

# Module metadata
__module_name__ = "GUI Components"
__description__ = "Graphical user interface for PDE simulations"
