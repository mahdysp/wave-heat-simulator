"""Base class for PDE equations"""
import numpy as np

# Compatibility for different NumPy versions
if hasattr(np, 'trapezoid'):
    trapz_func = np.trapezoid
else:
    trapz_func = np.trapz

class BasePDESolver:
    """Base class for partial differential equation solvers"""
    
    def __init__(self, L=1.0, num_terms=50):
        self.L = L
        self.num_terms = num_terms
 #by-mahdysp        
    def compute_coefficients(self, f, **kwargs):
        """Compute Fourier coefficients - to be implemented by subclasses"""
        raise NotImplementedError
        
    def solution(self, x, t):
        """Get solution at position x and time t - to be implemented by subclasses"""
        raise NotImplementedError
