"""Heat diffusion equation solver using Fourier series"""
import numpy as np
from .base_equation import BasePDESolver, trapz_func

class HeatDiffusion(BasePDESolver):
    """Heat diffusion simulation using Fourier series"""
    
    MATERIALS = {
        'Copper': 1.11e-4,
        'Aluminum': 9.7e-5,
        'Iron': 2.3e-5,
        'Steel': 1.2e-5,
        'Silver': 1.66e-4,
        'Gold': 1.27e-4,
        'Glass': 3.4e-7,
        'Wood': 8.2e-8,
        'Custom': 0.01
    }
    
    def __init__(self, L=1.0, alpha=0.01, num_terms=50, boundary_type='dirichlet'):
        super().__init__(L, num_terms)
        self.alpha = alpha
        self.boundary_type = boundary_type
        self.Bn = None
        self.T_left = 0  #by-mahdysp
        self.T_right = 0
        self._n_array = np.arange(1, num_terms + 1)
        self._lambda_n = (self._n_array * np.pi / L) ** 2
    
    def set_boundary_temperatures(self, T_left, T_right):
        """Set boundary temperatures for Dirichlet conditions"""
        self.T_left = T_left
        self.T_right = T_right
    
    def _steady_state_linear(self, x):
        """Linear steady state for Dirichlet BC"""
        return self.T_left + (self.T_right - self.T_left) * x / self.L
    
    def compute_coefficients(self, f, num_points=1000):
        """Compute Fourier coefficients for heat equation"""
        x = np.linspace(0, self.L, num_points)
        dx = x[1] - x[0]
        
        if self.boundary_type == 'dirichlet' and (self.T_left != 0 or self.T_right != 0):
            f_vals = np.asarray(f(x), dtype=float) - self._steady_state_linear(x)
        else:
            f_vals = np.asarray(f(x), dtype=float)
        
        if self.boundary_type == 'dirichlet':
            self.Bn = np.zeros(self.num_terms)
            for n in range(1, self.num_terms + 1):
                sin_mode = np.sin(n * np.pi * x / self.L)
                self.Bn[n - 1] = (2 / self.L) * trapz_func(f_vals * sin_mode, dx=dx)
        else:  # Neumann
            self.Bn = np.zeros(self.num_terms + 1)
            self.Bn[0] = (1 / self.L) * trapz_func(f_vals, dx=dx)
            for n in range(1, self.num_terms + 1):
                cos_mode = np.cos(n * np.pi * x / self.L)
                self.Bn[n] = (2 / self.L) * trapz_func(f_vals * cos_mode, dx=dx)
        
        return self.Bn
    
    def solution(self, x, t):
        """Get temperature at position x and time t"""
        x = np.atleast_1d(x)
        
        if self.boundary_type == 'dirichlet':
            u = np.zeros_like(x, dtype=float)
            for n in range(1, self.num_terms + 1):
                lambda_n = self._lambda_n[n - 1]
                decay = np.exp(-self.alpha * lambda_n * t)
                mode = np.sin(n * np.pi * x / self.L)
                u += self.Bn[n - 1] * decay * mode
            u += self._steady_state_linear(x)
        else:  # Neumann
            u = self.Bn[0] * np.ones_like(x)
            for n in range(1, self.num_terms + 1):
                lambda_n = self._lambda_n[n - 1]
                decay = np.exp(-self.alpha * lambda_n * t)
                mode = np.cos(n * np.pi * x / self.L)
                u += self.Bn[n] * decay * mode
        return u
    
    def steady_state(self, x):
        """Get steady state temperature"""
        x = np.atleast_1d(x)
        if self.boundary_type == 'dirichlet':
            return self._steady_state_linear(x)
        else:
            return self.Bn[0] * np.ones_like(x)
    
    def time_to_percentage(self, x_point, percentage, max_time=None):
        """Calculate time to reach a percentage of initial temperature difference"""
        x_arr = np.array([x_point])
        
        try:
            initial_temp = self.solution(x_arr, 0)[0]
            steady = self.steady_state(x_arr)[0]
        except Exception:
            return None
        
        initial_diff = abs(initial_temp - steady)
        
        if initial_diff < 1e-10:
            return 0.0
        
        if percentage <= 0:
            return None
        if percentage >= 100:
            return 0.0
        
        lambda_1 = (np.pi / self.L) ** 2
        tau_1 = 1.0 / (self.alpha * lambda_1)
        
        if max_time is None:
            max_time = 10 * tau_1
        
        t_low, t_high = 0.0, max_time
        target_diff = initial_diff * (percentage / 100.0)
        
        for _ in range(50):
            t_mid = (t_low + t_high) / 2
            
            try:
                current_temp = self.solution(x_arr, t_mid)[0]
            except Exception:
                return None
            
            current_diff = abs(current_temp - steady)
            
            if abs(current_diff - target_diff) < 1e-8 * initial_diff:
                return t_mid
            
            if current_diff > target_diff:
                t_low = t_mid
            else:
                t_high = t_mid
        
        return (t_low + t_high) / 2
    
    def get_decay_constant(self, n=1):
        """Get decay time constant for nth mode"""
        lambda_n = (n * np.pi / self.L) ** 2
        if abs(self.alpha * lambda_n) > 1e-10:
            return 1 / (self.alpha * lambda_n)
        return float('inf')
