"""
Utilities module for PDE Fourier Simulator
Helper functions for validation, export, and other utilities
"""

from .validators import (
    validate_wave_inputs,
    validate_heat_inputs,
    validate_positive,
    validate_range,
    validate_integer,
    ValidationError
)

from .export import (
    export_to_csv,
    export_to_json,
    save_figure,
    export_data_matrix,
    export_coefficients,
    export_animation_frames,
    create_report
)

__all__ = [
    # Validators
    'validate_wave_inputs',
    'validate_heat_inputs',
    'validate_positive',
    'validate_range',
    'validate_integer',
    'ValidationError',
    
    # Export functions
    'export_to_csv',
    'export_to_json',
    'save_figure',
    'export_data_matrix',
    'export_coefficients',
    'export_animation_frames',
    'create_report'
]

# Module metadata
__module_name__ = "Utilities"
__description__ = "Helper functions for validation and data export"
__version__ = "1.0.0"

# Utility constants
SUPPORTED_EXPORT_FORMATS = {
    'csv': ['.csv'],
    'json': ['.json'],
    'image': ['.png', '.jpg', '.jpeg', '.pdf', '.svg', '.eps'],
    'data': ['.npz', '.mat', '.h5', '.hdf5']
}

DEFAULT_EXPORT_SETTINGS = {
    'csv_delimiter': ',',
    'csv_precision': 6,
    'figure_dpi': 150,
    'figure_quality': 95,
    'json_indent': 2,
    'compression': True
}

# File size limits (in MB)
MAX_FILE_SIZE = {
    'csv': 100,
    'json': 50,
    'image': 20,
    'animation': 200
}
