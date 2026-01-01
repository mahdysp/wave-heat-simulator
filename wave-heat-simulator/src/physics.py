"""
Physics Engine for Wave and Heat Equations
Using Fourier Series Analytical Solutions
"""

import numpy as np

# Compatibility for different NumPy versions
if hasattr(np, 'trapezoid'):
    trapz_func = np.trapezoid
else:
    trapz_func = np.trapz


class VibratingString:
    """
    Vibrating string simulation using Fourier series.
    
    Solves the wave equation: ∂²u/∂t² = c² ∂²u/∂x²
    
    With boundary conditions: u(0,t) = u(L,t) = 0
    And initial conditions: u(x,0) = f(x), ∂u/∂t(x,0) = g(x)
    """

    def __init__(self, L=1.0, c=1.0, num_terms=50):
        """
        Initialize the vibrating string.
        
        Parameters:
            L: Length of the string
            c: Wave speed
            num_terms: Number of Fourier terms to use
        """
        self.L = L
        self.c = c
        self.num_terms = num_terms
        self.An = None
        self.Bn = None
        self._n_array = np.arange(1, num_terms + 1)
        self._omega_n = self._n_array * np.pi * c / L

    def compute_coefficients(self, f, g=None, num_points=1000):
        """
        Compute Fourier coefficients from initial conditions.
        
        Parameters:
            f: Initial displacement function f(x) = u(x, 0)
            g: Initial velocity function g(x) = ∂u/∂t(x, 0)
            num_points: Number of points for numerical integration
        
        Returns:
            An, Bn: Fourier coefficients
        """
        if g is None:
            g = lambda x: np.zeros_like(x)

        x = np.linspace(0, self.L, num_points)
        dx = x[1] - x[0]
        f_vals = f(x)
        g_vals = g(x)

        self.An = np.zeros(self.num_terms)
        self.Bn = np.zeros(self.num_terms)

        for n in range(1, self.num_terms + 1):
            sin_mode = np.sin(n * np.pi * x / self.L)
            self.An[n - 1] = (2 / self.L) * trapz_func(f_vals * sin_mode, dx=dx)
            integral = trapz_func(g_vals * sin_mode, dx=dx)
            self.Bn[n - 1] = (2 / self.L) * integral / self._omega_n[n - 1]

        return self.An, self.Bn

    def solution(self, x, t):
        """
        Compute the solution u(x, t) at given position and time.
        
        Parameters:
            x: Position(s) - scalar or array
            t: Time - scalar
        
        Returns:
            u: Displacement at (x, t)
        """
        x = np.atleast_1d(x)
        u = np.zeros_like(x, dtype=float)

        for n in range(1, self.num_terms + 1):
            omega_n = self._omega_n[n - 1]
            mode = np.sin(n * np.pi * x / self.L)
            time_part = self.An[n - 1] * np.cos(omega_n * t) + self.Bn[n - 1] * np.sin(omega_n * t)
            u += time_part * mode
        return u

    def get_mode(self, n, x):
        """Get the n-th mode shape φₙ(x) = sin(nπx/L)"""
        return np.sin(n * np.pi * x / self.L)

    def get_nodal_points(self, n):
        """Get nodal points (zeros) for the n-th mode"""
        return [k * self.L / n for k in range(n + 1)]

    def get_natural_frequency(self, n):
        """Get the n-th natural frequency in Hz"""
        return n * self.c / (2 * self.L)

    def compute_energy(self, x, t):
        """
        Compute kinetic, potential, and total energy.
        
        Parameters:
            x: Position array
            t: Time
        
        Returns:
            kinetic, potential, total: Energy values
        """
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
        """Create odd periodic extension for D'Alembert solution"""
        L = self.L
#by-mahdysp
        def f_extended(x):
            x_mod = np.mod(x, 2 * L)
            result = np.zeros_like(x_mod)
            mask1 = x_mod <= L
            result[mask1] = f(x_mod[mask1])
            mask2 = x_mod > L
            result[mask2] = -f(2 * L - x_mod[mask2])
            return result

        return f_extended


class HeatDiffusion:
    """
    Heat diffusion simulation using Fourier series.
    
    Solves the heat equation: ∂u/∂t = α ∂²u/∂x²
    
    With boundary conditions: 
        - Dirichlet: u(0,t) = T_left, u(L,t) = T_right
        - Neumann: ∂u/∂x(0,t) = ∂u/∂x(L,t) = 0
    """

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
        """
        Initialize heat diffusion simulation.
        
        Parameters:
            L: Length of the rod
            alpha: Thermal diffusivity (m²/s)
            num_terms: Number of Fourier terms
            boundary_type: 'dirichlet' or 'neumann'
        """
        self.L = L
        self.alpha = alpha
        self.num_terms = num_terms
        self.boundary_type = boundary_type
        self.Bn = None
        self.T_left = 0
        self.T_right = 0
        self._n_array = np.arange(1, num_terms + 1)
        self._lambda_n = (self._n_array * np.pi / L) ** 2

    def set_boundary_temperatures(self, T_left, T_right):
        """Set boundary temperatures for Dirichlet conditions"""
        self.T_left = T_left
        self.T_right = T_right

    def _steady_state_linear(self, x):
        """Linear steady-state for non-homogeneous Dirichlet BC"""
        return self.T_left + (self.T_right - self.T_left) * x / self.L

    def compute_coefficients(self, f, num_points=1000):
        """
        Compute Fourier coefficients from initial temperature distribution.
        
        Parameters:
            f: Initial temperature function f(x) = T(x, 0)
            num_points: Number of points for numerical integration
        
        Returns:
            Bn: Fourier coefficients
        """
        x = np.linspace(0, self.L, num_points)
        dx = x[1] - x[0]
	#by-mahdysp
        if self.boundary_type == 'dirichlet' and (self.T_left != 0 or self.T_right != 0):
            f_vals = f(x) - self._steady_state_linear(x)
        else:
            f_vals = f(x)

        if self.boundary_type == 'dirichlet':
            self.Bn = np.zeros(self.num_terms)
            for n in range(1, self.num_terms + 1):
                sin_mode = np.sin(n * np.pi * x / self.L)
                self.Bn[n - 1] = (2 / self.L) * trapz_func(f_vals * sin_mode, dx=dx)
        else:
            self.Bn = np.zeros(self.num_terms + 1)
            self.Bn[0] = (1 / self.L) * trapz_func(f_vals, dx=dx)
            for n in range(1, self.num_terms + 1):
                cos_mode = np.cos(n * np.pi * x / self.L)
                self.Bn[n] = (2 / self.L) * trapz_func(f_vals * cos_mode, dx=dx)

        return self.Bn

    def solution(self, x, t):
        """
        Compute the temperature T(x, t) at given position and time.
        
        Parameters:
            x: Position(s) - scalar or array
            t: Time - scalar
        
        Returns:
            T: Temperature at (x, t)
        """
        x = np.atleast_1d(x)

        if self.boundary_type == 'dirichlet':
            u = np.zeros_like(x, dtype=float)
            for n in range(1, self.num_terms + 1):
                lambda_n = self._lambda_n[n - 1]
                decay = np.exp(-self.alpha * lambda_n * t)
                mode = np.sin(n * np.pi * x / self.L)
                u += self.Bn[n - 1] * decay * mode
            u += self._steady_state_linear(x)
        else:
            u = self.Bn[0] * np.ones_like(x)
            for n in range(1, self.num_terms + 1):
                lambda_n = self._lambda_n[n - 1]
                decay = np.exp(-self.alpha * lambda_n * t)
                mode = np.cos(n * np.pi * x / self.L)
                u += self.Bn[n] * decay * mode
        return u

    def steady_state(self, x):
        """Compute steady-state temperature distribution"""
        x = np.atleast_1d(x)
        if self.boundary_type == 'dirichlet':
            return self._steady_state_linear(x)
        else:
            return self.Bn[0] * np.ones_like(x)

    def time_to_percentage(self, x_point, percentage, max_time=1000):
        """
        Calculate time to reach a certain percentage of steady state.
        
        Parameters:
            x_point: Position to monitor
            percentage: Target percentage remaining (e.g., 10 means 10% of initial diff)
            max_time: Maximum time to search
        
        Returns:
            Time in seconds, or None if not reached
        """
        dt = 0.001
        t = 0
        x_arr = np.array([x_point])
        initial_temp = self.solution(x_arr, 0)[0]
        steady = self.steady_state(x_arr)[0]
        initial_diff = abs(initial_temp - steady)

        if initial_diff < 1e-10:
            return 0

        target_diff = initial_diff * (percentage / 100.0)

        while t < max_time:
            current_temp = self.solution(x_arr, t)[0]
            current_diff = abs(current_temp - steady)
            if current_diff <= target_diff:
                return t
            t += dt
        return None
		#by-mahdysp
    def get_decay_constant(self, n=1):
        """Get the time constant τₙ for the n-th mode"""
        lambda_n = (n * np.pi / self.L) ** 2
        return 1 / (self.alpha * lambda_n)
