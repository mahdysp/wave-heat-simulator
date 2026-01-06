"""Wave equation solver using Fourier series"""
import numpy as np
from .base_equation import BasePDESolver, trapz_func

class VibratingString(BasePDESolver):
    """Vibrating string simulation using Fourier series"""
    
    def __init__(self, L=1.0, c=1.0, num_terms=50):
        super().__init__(L, num_terms)
        self.c = c
        self.An = None
        self.Bn = None
        self._n_array = np.arange(1, num_terms + 1)
        self._omega_n = self._n_array * np.pi * c / L
    
    def compute_coefficients(self, f, g=None, num_points=1000):
        """
        Compute Fourier coefficients for wave equation
        f: initial displacement function
        g: initial velocity function
        """
        if g is None:
            g = lambda x: np.zeros_like(x)
        
        x = np.linspace(0, self.L, num_points)
        dx = x[1] - x[0]
        f_vals = np.asarray(f(x), dtype=float)
        g_vals = np.asarray(g(x), dtype=float)
        
        self.An = np.zeros(self.num_terms)
        self.Bn = np.zeros(self.num_terms)
        
        for n in range(1, self.num_terms + 1):
            sin_mode = np.sin(n * np.pi * x / self.L)
            self.An[n - 1] = (2 / self.L) * trapz_func(f_vals * sin_mode, dx=dx)
            integral = trapz_func(g_vals * sin_mode, dx=dx)
            if abs(self._omega_n[n - 1]) > 1e-10:
                self.Bn[n - 1] = (2 / self.L) * integral / self._omega_n[n - 1]
            else:
                self.Bn[n - 1] = 0
        
        return self.An, self.Bn
    
    def solution(self, x, t):
        """Get displacement at position x and time t"""
        x = np.atleast_1d(x)
        u = np.zeros_like(x, dtype=float)
        
        for n in range(1, self.num_terms + 1):
            omega_n = self._omega_n[n - 1]
            mode = np.sin(n * np.pi * x / self.L)
            time_part = self.An[n - 1] * np.cos(omega_n * t) + self.Bn[n - 1] * np.sin(omega_n * t)
            u += time_part * mode
        return u
    
    def get_mode(self, n, x):
        """Get nth normal mode"""
        return np.sin(n * np.pi * x / self.L)
    
    def get_nodal_points(self, n):
        """Get nodal points for nth mode"""
        return [k * self.L / n for k in range(n + 1)]
    
    def get_natural_frequency(self, n):
        """Get natural frequency for nth mode"""
        return n * self.c / (2 * self.L)
    
    def compute_energy(self, x, t):
        """Compute kinetic, potential and total energy"""
        dx = x[1] - x[0]
        u = self.solution(x, t)
        dt = 0.0001
        u_plus = self.solution(x, t + dt)
        u_minus = self.solution(x, t - dt)
        dudt = (u_plus - u_minus) / (2 * dt)
        dudx = np.gradient(u, dx)
        kinetic = 0.5 * trapz_func(dudt ** 2, dx=dx)
        potential = 0.5 * self.c ** 2 * trapz_func(dudx ** 2, dx=dx)
        return kinetic, potential, kinetic + potential
    
    def create_odd_periodic_extension(self, f):
        """Create odd periodic extension of function f"""
        L = self.L
        
        def f_extended(x):
            x = np.atleast_1d(x).astype(float)
            x_mod = np.mod(x, 2 * L)
            result = np.zeros_like(x_mod)
            mask1 = x_mod <= L
            if np.any(mask1):
                result[mask1] = f(x_mod[mask1])
            mask2 = x_mod > L
            if np.any(mask2):
                result[mask2] = -f(2 * L - x_mod[mask2])
            return result
        
        return f_extended
