"""Plotting and visualization utilities"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tkinter import messagebox
from .wave_controls import get_wave_initial_shape
from .heat_controls import get_heat_initial_temp


class PlottingManager:
    """Manages all plotting operations"""
    
    def __init__(self, app):
        self.app = app
    
    # ==================== WAVE PLOTTING ====================
    
    def plot_wave(self):
        """Plot wave equation solution at different times"""
        if self.app.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            L = self.app.wave_L_var.get()
            T = self.app.wave_time_var.get()
            x = np.linspace(0, L, 500)
            times = [0, T / 6, T / 3, T / 2, 2 * T / 3, T]
            
            for i, t in enumerate(times):
                ax = self.app.fig.add_subplot(2, 3, i + 1)
                u = self.app.wave_sim.solution(x, t)
                ax.plot(x, u, 'b-', linewidth=2)
                ax.axhline(0, color='k', linewidth=0.5)
                ax.fill_between(x, 0, u, alpha=0.3)
                ax.set_xlabel('x')
                ax.set_ylabel('u(x,t)')
                ax.set_title(f't = {t:.3f}')
                ax.set_ylim([-self.app.wave_amp_var.get() * 1.5, 
                            self.app.wave_amp_var.get() * 1.5])
                ax.grid(True, alpha=0.3)
            
            self.app.fig.suptitle('Wave Equation - Time Evolution', 
                                fontsize=14, fontweight='bold')
            self.app.fig.tight_layout()
            self.app.canvas.draw()
            self.app.status_var.set("Wave plot completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Plotting failed: {str(e)}")
    
    def animate_wave(self):
        """Create wave animation"""
        if self.app.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            L = self.app.wave_L_var.get()
            T = self.app.wave_time_var.get()
            amp = self.app.wave_amp_var.get()
            x = np.linspace(0, L, 500)
            
            ax = self.app.fig.add_subplot(111)
            line, = ax.plot([], [], 'b-', linewidth=2)
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlim(0, L)
            ax.set_ylim(-amp * 1.5, amp * 1.5)
            ax.set_xlabel('Position x')
            ax.set_ylabel('Displacement u(x,t)')
            ax.set_title('Vibrating String Animation')
            ax.grid(True, alpha=0.3)
            ax.plot([0, L], [0, 0], 'ko', markersize=8)
            
            wave_sim = self.app.wave_sim
            
            def init():
                line.set_data([], [])
                time_text.set_text('')
                return line, time_text
            
            def animate(frame):
                t = frame * T / 200
                u = wave_sim.solution(x, t)
                line.set_data(x, u)
                time_text.set_text(f't = {t:.3f} s')
                return line, time_text
            
            self.app.animation = FuncAnimation(
                self.app.fig, animate, init_func=init, frames=200,
                interval=50, blit=True, repeat=True
            )
            self.app.canvas.draw()
            self.app.status_var.set("Animation running... Press Space or Escape to stop")
            
        except Exception as e:
            messagebox.showerror("Error", f"Animation failed: {str(e)}")
    def show_wave_modes(self):
        """Show natural modes of vibration"""
        if self.app.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            L = self.app.wave_L_var.get()
            x = np.linspace(0, L, 500)
            colors = plt.cm.viridis(np.linspace(0, 1, 5))
            
            ax = self.app.fig.add_subplot(111)
            for n in range(1, 6):
                mode = self.app.wave_sim.get_mode(n, x)
                freq = self.app.wave_sim.get_natural_frequency(n)
                ax.plot(x, mode + 2.5 * (5 - n), color=colors[n - 1], linewidth=2,
                       label=f'Mode {n}: f = {freq:.3f} Hz')
                ax.axhline(y=2.5 * (5 - n), color='gray', linewidth=0.5, linestyle='--')
            
            ax.set_xlabel('Position x')
            ax.set_ylabel('Mode Shape (offset)')
            ax.set_title('Natural Modes of Vibrating String: φₙ(x) = sin(nπx/L)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            self.app.fig.tight_layout()
            self.app.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show modes: {str(e)}")
    
    def show_nodal_points(self):
        """Show nodal points for different modes"""
        if self.app.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            L = self.app.wave_L_var.get()
            ax = self.app.fig.add_subplot(111)
            colors = plt.cm.Set1(np.linspace(0, 1, 7))
            
            for n in range(1, 8):
                nodes = self.app.wave_sim.get_nodal_points(n)
                ax.scatter(nodes, [n] * len(nodes), s=150, c=[colors[n - 1]],
                          edgecolors='black', linewidth=1.5, label=f'Mode {n}: {len(nodes)} nodes')
                ax.hlines(n, 0, L, colors='gray', linestyles='--', alpha=0.5)
                
                x_mode = np.linspace(0, L, 200)
                mode_shape = 0.3 * np.sin(n * np.pi * x_mode / L)
                ax.plot(x_mode, n + mode_shape, color=colors[n - 1], alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Position x')
            ax.set_ylabel('Mode Number')
            ax.set_title('Nodal Points (Zero Displacement Positions)')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05 * L, 1.05 * L)
            self.app.fig.tight_layout() #by-mahdysp
            self.app.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show nodal points: {str(e)}")
    
    def show_energy_plot(self):
        """Show energy conservation in wave equation"""
        if self.app.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            self.app.status_var.set("Computing energy...")
            self.app.root.update()
            
            L = self.app.wave_L_var.get()
            T = self.app.wave_time_var.get()
            x = np.linspace(0, L, 500)
            t_range = np.linspace(0, T, 100)
            
            kinetic, potential, total = [], [], []
            for t in t_range:
                K, P, E = self.app.wave_sim.compute_energy(x, t)
                kinetic.append(K)
                potential.append(P)
                total.append(E)
            
            ax = self.app.fig.add_subplot(111)
            ax.plot(t_range, kinetic, 'b-', linewidth=2, label='Kinetic Energy')
            ax.plot(t_range, potential, 'r-', linewidth=2, label='Potential Energy')
            ax.plot(t_range, total, 'g--', linewidth=2.5, label='Total Energy')
            
            E_mean = np.mean(total)
            E_std = np.std(total)
            if E_mean > 1e-10:
                variation = 100 * E_std / E_mean
            else:
                variation = 0
            ax.axhline(y=E_mean, color='purple', linestyle=':', alpha=0.7)
            ax.text(0.98, 0.02, f'Energy variation: {variation:.4f}%',
                   transform=ax.transAxes, ha='right', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow'))
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Energy')
            ax.set_title('Energy Conservation in Vibrating String')
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.app.fig.tight_layout()
            self.app.canvas.draw()
            self.app.status_var.set("Energy plot completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show energy plot: {str(e)}")
    
    def show_wave_spectrum(self):
        """Show Fourier coefficients spectrum"""
        if self.app.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            ax1 = self.app.fig.add_subplot(121)
            ax2 = self.app.fig.add_subplot(122)
            
            n_vals = np.arange(1, len(self.app.wave_sim.An) + 1)
            
            ax1.stem(n_vals, self.app.wave_sim.An, basefmt=' ', linefmt='b-', markerfmt='bo')
            ax1.axhline(y=0, color='k', linewidth=0.5)
            ax1.set_xlabel('Mode Number (n)')
            ax1.set_ylabel('Aₙ')
            ax1.set_title('Coefficients Aₙ (from initial shape)')
            ax1.grid(True, alpha=0.3)
            
            ax2.stem(n_vals, self.app.wave_sim.Bn, basefmt=' ', linefmt='r-', markerfmt='ro')
            ax2.axhline(y=0, color='k', linewidth=0.5)
            ax2.set_xlabel('Mode Number (n)')
            ax2.set_ylabel('Bₙ')
            ax2.set_title('Coefficients Bₙ (from initial velocity)')
            ax2.grid(True, alpha=0.3)
            
            self.app.fig.suptitle('Fourier Series Coefficients', fontsize=14, fontweight='bold')
            self.app.fig.tight_layout()
            self.app.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show spectrum: {str(e)}")
    
    def show_dalembert(self):
        """Show D'Alembert solution visualization"""
        if self.app.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            L = self.app.wave_L_var.get()
            c = self.app.wave_c_var.get()
            T = self.app.wave_time_var.get()
            amp = self.app.wave_amp_var.get()
            
            f = get_wave_initial_shape(self.app)
            f_ext = self.app.wave_sim.create_odd_periodic_extension(f)
            x_ext = np.linspace(-2 * L, 3 * L, 1000)
            times = [0, T / 8, T / 4, T / 2]
            
            for i, t in enumerate(times):
                ax = self.app.fig.add_subplot(2, 2, i + 1)
                forward = 0.5 * f_ext(x_ext - c * t)
                backward = 0.5 * f_ext(x_ext + c * t)
                total = forward + backward
                
                ax.plot(x_ext, forward, 'b--', alpha=0.6, linewidth=1.5, label='F(x-ct)/2')
                ax.plot(x_ext, backward, 'r--', alpha=0.6, linewidth=1.5, label='F(x+ct)/2')
                ax.plot(x_ext, total, 'g-', linewidth=2, label='Total')
                ax.axvspan(0, L, alpha=0.2, color='yellow', label='Physical domain')
                ax.axvline(0, color='k', linewidth=2)
                ax.axvline(L, color='k', linewidth=2)
                ax.set_xlim(-0.5 * L, 1.5 * L)
                ax.set_ylim(-amp * 1.5, amp * 1.5)
                ax.set_xlabel('x')
                ax.set_ylabel('u')
                ax.set_title(f't = {t:.3f} s')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend(fontsize=7, loc='upper right')
            
            self.app.fig.suptitle("D'Alembert Solution: u(x,t) = ½[F(x-ct) + F(x+ct)]",
                                fontsize=12, fontweight='bold')
            self.app.fig.tight_layout()
            self.app.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show D'Alembert view: {str(e)}")
    
    # ==================== HEAT PLOTTING ====================
    
    def plot_heat(self):
        """Plot heat equation solution at different times"""
        if self.app.heat_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            L = self.app.heat_L_var.get()
            T = self.app.heat_time_var.get()
            x = np.linspace(0, L, 500)
            times = [0, T / 10, T / 4, T / 2, 3 * T / 4, T]
            
            for i, t in enumerate(times):
                ax = self.app.fig.add_subplot(2, 3, i + 1)
                temp = self.app.heat_sim.solution(x, t)
                ax.plot(x, temp, 'r-', linewidth=2)
                ax.fill_between(x, 0, temp, alpha=0.3, color='red')
                ax.plot(x, self.app.heat_sim.steady_state(x), 'g--', linewidth=1.5, alpha=0.7)
                ax.set_xlabel('x')
                ax.set_ylabel('T(x,t)')
                ax.set_title(f't = {t:.3f}')
                ax.grid(True, alpha=0.3)
            
            self.app.fig.suptitle('Heat Equation - Temperature Evolution', 
                                fontsize=14, fontweight='bold')
            self.app.fig.tight_layout()
            self.app.canvas.draw()
            self.app.status_var.set("Heat plot completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Plotting failed: {str(e)}")
    
    def animate_heat(self):
        """Create heat diffusion animation"""
        if self.app.heat_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            L = self.app.heat_L_var.get()
            T = self.app.heat_time_var.get()
            T_max = self.app.heat_max_temp_var.get()
            x = np.linspace(0, L, 500)
            
            ax = self.app.fig.add_subplot(111)
            line, = ax.plot([], [], 'r-', linewidth=2)
            ax.plot(x, self.app.heat_sim.steady_state(x), 'g--', 
                   linewidth=1.5, alpha=0.7, label='Steady State')
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlim(0, L)
            ax.set_ylim(-T_max * 0.1, T_max * 1.2)
            ax.set_xlabel('Position x')
            ax.set_ylabel('Temperature T(x,t)')
            ax.set_title('Heat Diffusion Animation')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            heat_sim = self.app.heat_sim
            
            def init():
                line.set_data([], [])
                time_text.set_text('')
                return line, time_text
            
            def animate(frame):
                t = frame * T / 150
                temp = heat_sim.solution(x, t)
                line.set_data(x, temp)
                time_text.set_text(f't = {t:.3f} s')
                return line, time_text
            
            self.app.animation = FuncAnimation(
                self.app.fig, animate, init_func=init, frames=150,
                interval=50, blit=True, repeat=True
            )
            self.app.canvas.draw()
            self.app.status_var.set("Animation running... Press Space or Escape to stop")
            
        except Exception as e:
            messagebox.showerror("Error", f"Animation failed: {str(e)}")
    
    def compare_materials(self):
        """Compare different materials heat diffusion"""
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            self.app.status_var.set("Comparing materials...")
            self.app.root.update()
            
            L = self.app.heat_L_var.get()
            num_terms = self.app.heat_terms_var.get()
            x = np.linspace(0, L, 200)
            f = get_heat_initial_temp(self.app)
            
            materials = ['Copper', 'Aluminum', 'Iron', 'Steel', 'Glass', 'Wood']
            colors = plt.cm.viridis(np.linspace(0, 1, len(materials)))
            
            ax1 = self.app.fig.add_subplot(121)
            ax2 = self.app.fig.add_subplot(122)
            
            t_fixed = 1.0
            decay_times = []
            
            for i, mat in enumerate(materials):
                from ..physics.heat_equation import HeatDiffusion
                alpha = HeatDiffusion.MATERIALS[mat]
                heat = HeatDiffusion(L=L, alpha=alpha, num_terms=num_terms, 
                                   boundary_type=self.app.heat_bc_var.get())
                heat.compute_coefficients(f)
                temp = heat.solution(x, t_fixed) #by-mahdysp
                ax1.plot(x, temp, color=colors[i], linewidth=2, label=f'{mat}')
                
                tau = heat.get_decay_constant(1)
                decay_times.append((mat, tau))
            
            ax1.set_xlabel('Position x')
            ax1.set_ylabel('Temperature')
            ax1.set_title(f'Temperature at t = {t_fixed} s')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            names = [d[0] for d in decay_times]
            taus = [d[1] for d in decay_times]
            ax2.barh(names, taus, color=colors)
            ax2.set_xlabel('Time Constant τ₁ (seconds)')
            ax2.set_title('Thermal Time Constants')
            ax2.grid(True, alpha=0.3, axis='x')
            
            self.app.fig.suptitle('Material Comparison', fontsize=14, fontweight='bold')
            self.app.fig.tight_layout()
            self.app.canvas.draw()
            self.app.status_var.set("Material comparison completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {str(e)}")
    
    def show_steady_state(self):
        """Show approach to steady state"""
        if self.app.heat_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            L = self.app.heat_L_var.get()
            T = self.app.heat_time_var.get()
            x = np.linspace(0, L, 200)
            
            ax = self.app.fig.add_subplot(111)
            times = np.linspace(0, T, 10)
            colors = plt.cm.hot(np.linspace(0.8, 0.2, len(times)))
            
            for i, t in enumerate(times):
                temp = self.app.heat_sim.solution(x, t)
                ax.plot(x, temp, color=colors[i], linewidth=2, label=f't = {t:.2f}')
            
            ax.plot(x, self.app.heat_sim.steady_state(x), 'g--', 
                   linewidth=3, label='Steady State')
            ax.set_xlabel('Position x')
            ax.set_ylabel('Temperature')
            ax.set_title('Approach to Steady State')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            self.app.fig.tight_layout()
            self.app.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show steady state: {str(e)}")
    
    def show_center_temp(self):
        """Show temperature at center vs time"""
        if self.app.heat_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            self.app.status_var.set("Computing center temperature...")
            self.app.root.update()
            
            L = self.app.heat_L_var.get()
            T = self.app.heat_time_var.get()
            t_range = np.linspace(0, T, 500)
            x_center = np.array([L / 2])
            
            T_center = [self.app.heat_sim.solution(x_center, t)[0] for t in t_range]
            steady = self.app.heat_sim.steady_state(x_center)[0]
            tau = self.app.heat_sim.get_decay_constant(1)
            
            ax = self.app.fig.add_subplot(111)
            ax.plot(t_range, T_center, 'b-', linewidth=2, label='T(L/2, t)')
            ax.axhline(steady, color='r', linestyle='--', linewidth=2, 
                      label=f'Steady = {steady:.2f}')
            
            for n in [1, 2, 3, 5]:
                if n * tau < T:
                    ax.axvline(x=n * tau, color='green', linestyle=':', alpha=0.7)
                    ax.text(n * tau, ax.get_ylim()[1] * 0.95, f'{n}τ', 
                           ha='center', fontsize=9, color='green')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Temperature at Center')
            ax.set_title(f'Center Temperature vs Time (τ₁ = {tau:.4f} s)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.app.fig.tight_layout()
            self.app.canvas.draw()
            self.app.status_var.set("Center temperature plot completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show center temp: {str(e)}")
    
    def show_numerical_analysis(self):
        """Show numerical analysis table"""
        if self.app.heat_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        
        try:
            self.app.stop_animation()
            self.app.fig.clear()
            
            self.app.status_var.set("Computing numerical analysis...")
            self.app.root.update()
            
            L = self.app.heat_L_var.get()
            x_center = L / 2
            
            ax = self.app.fig.add_subplot(111)
            ax.axis('off')
            
            initial_temp = self.app.heat_sim.solution(np.array([x_center]), 0)[0]
            steady_temp = self.app.heat_sim.steady_state(np.array([x_center]))[0]
            tau = self.app.heat_sim.get_decay_constant(1)
            
            percentages = [90, 75, 50, 25, 10, 5, 1]
            results = []
            
            for pct in percentages:
                try:
                    t = self.app.heat_sim.time_to_percentage(x_center, pct)
                    if t is not None and t < 1e6:
                        results.append([f"{pct}%", f"{t:.4f} s"])
                    else:
                        results.append([f"{pct}%", "N/A"])
                except Exception:
                    results.append([f"{pct}%", "Error"])
            
            table_data = [["% Remaining", "Time Required"]] + results
            table = ax.table(
                cellText=table_data,
                loc='center',
                cellLoc='center',
                colWidths=[0.3, 0.3]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.8)
            
            for i in range(2):
                table[(0, i)].set_facecolor('#4a9eff')
                table[(0, i)].set_text_props(color='white', fontweight='bold')
            
            ax.set_title(
                f'Numerical Analysis: Time to Reach Temperature\n'
                f'Center (x = L/2) | Initial: {initial_temp:.2f} | '
                f'Steady: {steady_temp:.2f} | τ₁ = {tau:.4f}s',
                fontsize=12, fontweight='bold', pad=20
            )
            
            self.app.fig.tight_layout()
            self.app.canvas.draw()
            self.app.status_var.set("Numerical analysis completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Numerical analysis failed: {str(e)}")
            self.app.status_var.set("Error during numerical analysis")
