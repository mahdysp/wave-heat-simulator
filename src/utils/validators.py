"""Input validation utilities for PDE simulations"""

import numpy as np
from typing import Union, Tuple, List, Optional


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_positive(value: float, name: str, allow_zero: bool = False) -> bool:
    """
    Validate that a value is positive
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        allow_zero: Whether to allow zero as valid
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")
    return True


def validate_range(value: float, min_val: float, max_val: float, 
                  name: str, inclusive: bool = True) -> bool:
    """
    Validate that a value is within a range
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the parameter
        inclusive: Whether boundaries are inclusive
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if inclusive:
        if not (min_val <= value <= max_val):
            raise ValidationError(
                f"{name} must be between {min_val} and {max_val} (inclusive), got {value}"
            )
    else:
        if not (min_val < value < max_val):
            raise ValidationError(
                f"{name} must be between {min_val} and {max_val} (exclusive), got {value}"
            )
    return True


def validate_integer(value: Union[int, float], name: str, 
                    min_val: Optional[int] = None,
                    max_val: Optional[int] = None) -> bool:
    """
    Validate that a value is an integer within bounds
    
    Args:
        value: Value to validate
        name: Name of the parameter
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check if it's actually an integer
        if not isinstance(value, int):
            if isinstance(value, float) and not value.is_integer():
                raise ValidationError(f"{name} must be an integer, got {value}")
            value = int(value)
        
        # Check bounds if provided
        if min_val is not None and value < min_val:
            raise ValidationError(f"{name} must be at least {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"{name} must be at most {max_val}, got {value}")
        
        return True
        
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{name} must be a valid integer: {str(e)}")


def validate_wave_inputs(L: float, c: float, num_terms: int, 
                        sim_time: float) -> Tuple[float, float, int, float]:
    """
    Validate wave simulation inputs
    
    Args:
        L: String length
        c: Wave speed
        num_terms: Number of Fourier terms
        sim_time: Simulation time
        
    Returns:
        Validated tuple of (L, c, num_terms, sim_time)
        
    Raises:
        ValidationError: If any validation fails
    """
    try:
        # Validate length
        validate_positive(L, "Length (L)")
        validate_range(L, 0.01, 1000, "Length (L)")
        
        # Validate wave speed
        validate_positive(c, "Wave speed (c)")
        validate_range(c, 0.01, 1000, "Wave speed (c)")
        
        # Validate number of terms
        validate_integer(num_terms, "Number of Fourier terms", min_val=1, max_val=500)
        
        # Validate simulation time
        validate_positive(sim_time, "Simulation time")
        validate_range(sim_time, 0.001, 10000, "Simulation time")
        
        # Additional physics validation
        period = 2 * L / c
        if sim_time < period / 10:
            raise ValidationError(
                f"Simulation time ({sim_time:.3f}s) is too short. "
                f"Recommended minimum: {period/10:.3f}s (1/10 of period)"
            )
        
        # Warn about excessive simulation time
        if sim_time > 100 * period:
            print(f"Warning: Simulation time is {sim_time/period:.1f} periods, "
                  f"which may be excessive")
        
        return L, c, int(num_terms), sim_time
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Unexpected error validating wave inputs: {str(e)}")


def validate_heat_inputs(L: float, alpha: float, num_terms: int,
                        sim_time: float) -> Tuple[float, float, int, float]:
    """
    Validate heat simulation inputs
    
    Args:
        L: Rod length
        alpha: Thermal diffusivity
        num_terms: Number of Fourier terms
        sim_time: Simulation time
        
    Returns:
        Validated tuple of (L, alpha, num_terms, sim_time)
        
    Raises:
        ValidationError: If any validation fails
    """
    try:
        # Validate length
        validate_positive(L, "Length (L)")
        validate_range(L, 0.001, 1000, "Length (L)")
        
        # Validate thermal diffusivity
        validate_positive(alpha, "Thermal diffusivity (α)")
        validate_range(alpha, 1e-10, 1, "Thermal diffusivity (α)")
        
        # Validate number of terms
        validate_integer(num_terms, "Number of Fourier terms", min_val=1, max_val=500)
        
        # Validate simulation time
        validate_positive(sim_time, "Simulation time")
        validate_range(sim_time, 0.001, 100000, "Simulation time")
        
        # Additional physics validation
        diffusion_time = L**2 / alpha  # Characteristic diffusion time
        
        if sim_time < diffusion_time / 1000:
            raise ValidationError(
                f"Simulation time ({sim_time:.3e}s) is too short. "
                f"Recommended minimum: {diffusion_time/1000:.3e}s"
            )
        
        # Warn about steady state
        steady_state_time = 5 * diffusion_time / (np.pi**2)
        if sim_time > steady_state_time:
            print(f"Note: System will reach steady state around {steady_state_time:.3f}s")
        
        return L, alpha, int(num_terms), sim_time
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Unexpected error validating heat inputs: {str(e)}")


def validate_boundary_conditions(bc_type: str, T_left: float = None, 
                                T_right: float = None) -> bool:
    """
    Validate boundary condition parameters
    
    Args:
        bc_type: Type of boundary condition ('dirichlet' or 'neumann')
        T_left: Left boundary temperature (for Dirichlet)
        T_right: Right boundary temperature (for Dirichlet)
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    valid_types = ['dirichlet', 'neumann']
    if bc_type.lower() not in valid_types:
        raise ValidationError(
            f"Invalid boundary condition type: {bc_type}. "
            f"Must be one of {valid_types}"
        )
    
    if bc_type.lower() == 'dirichlet':
        if T_left is not None:
            validate_range(T_left, -1000, 1000, "Left boundary temperature")
        if T_right is not None:
            validate_range(T_right, -1000, 1000, "Right boundary temperature")
    
    return True


def validate_initial_conditions(shape_type: str, amplitude: float = None,
                               velocity_type: str = None, 
                               velocity_amplitude: float = None) -> bool:
    """
    Validate initial condition parameters
    
    Args:
        shape_type: Type of initial shape
        amplitude: Amplitude of initial shape
        velocity_type: Type of initial velocity (for wave equation)
        velocity_amplitude: Amplitude of initial velocity
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    valid_shapes = ['triangular', 'sinusoidal', 'gaussian', 'step', 'plucked', 'uniform']
    if shape_type.lower() not in valid_shapes:
        raise ValidationError(
            f"Invalid initial shape type: {shape_type}. "
            f"Must be one of {valid_shapes}"
        )
    
    if amplitude is not None:
        validate_positive(amplitude, "Amplitude", allow_zero=True)
        validate_range(amplitude, 0, 1000, "Amplitude")
    
    if velocity_type is not None:
        valid_velocities = ['zero', 'sinusoidal', 'gaussian', 'plucked']
        if velocity_type.lower() not in valid_velocities:
            raise ValidationError(
                f"Invalid velocity type: {velocity_type}. "
                f"Must be one of {valid_velocities}"
            )
    
    if velocity_amplitude is not None:
        validate_positive(velocity_amplitude, "Velocity amplitude", allow_zero=True)
        validate_range(velocity_amplitude, 0, 1000, "Velocity amplitude")
    
    return True #by-mahdysp


def validate_export_path(filepath: str, file_type: str) -> bool:
    """
    Validate export file path
    
    Args:
        filepath: Path to export file
        file_type: Type of file ('csv', 'json', 'image', etc.)
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    import os
    
    if not filepath:
        raise ValidationError("Export path cannot be empty")
    
    # Check directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        raise ValidationError(f"Directory does not exist: {directory}")
    
    # Check file extension
    ext = os.path.splitext(filepath)[1].lower()
    valid_extensions = {
        'csv': ['.csv'],
        'json': ['.json'],
        'image': ['.png', '.jpg', '.jpeg', '.pdf', '.svg'],
        'data': ['.npz', '.mat', '.h5']
    }
    
    if file_type in valid_extensions:
        if ext not in valid_extensions[file_type]:
            raise ValidationError(
                f"Invalid extension {ext} for {file_type}. "
                f"Expected one of {valid_extensions[file_type]}"
            )
    
    # Check write permission
    if directory:
        if not os.access(directory, os.W_OK):
            raise ValidationError(f"No write permission for directory: {directory}")
    
    return True


def validate_array_dimensions(array: np.ndarray, expected_dims: int = None,
                             expected_shape: Tuple[int, ...] = None,
                             name: str = "Array") -> bool:
    """
    Validate numpy array dimensions and shape
    
    Args:
        array: Numpy array to validate
        expected_dims: Expected number of dimensions
        expected_shape: Expected shape tuple
        name: Name for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(array, np.ndarray):
        raise ValidationError(f"{name} must be a numpy array")
    
    if expected_dims is not None:
        if array.ndim != expected_dims:
            raise ValidationError(
                f"{name} must have {expected_dims} dimensions, got {array.ndim}"
            )
    
    if expected_shape is not None:
        if array.shape != expected_shape:
            raise ValidationError(
                f"{name} must have shape {expected_shape}, got {array.shape}"
            )
    
    # Check for NaN or Inf
    if np.any(np.isnan(array)):
        raise ValidationError(f"{name} contains NaN values")
    
    if np.any(np.isinf(array)):
        raise ValidationError(f"{name} contains infinite values")
    
    return True
