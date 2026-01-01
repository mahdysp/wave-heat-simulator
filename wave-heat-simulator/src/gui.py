"""
GUI Application for Wave and Heat Equation Simulator
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json

from physics import VibratingString, HeatDiffusion


class SimulationGUI:
    """Main GUI Application"""

    def __init__(self, root):
        self.root = root
        self.root.title("Wave & Heat Equation Simulator")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.colors = {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'accent': '#4a9eff',
            'success': '#4caf50',
            'warning': '#ff9800',
            'error': '#f44336',
            'panel': '#3c3c3c'
        }

        self.root.configure(bg=self.colors['bg'])
        self._configure_styles()

        self.wave_sim = None
        self.heat_sim = None
        self.animation = None
        self.is_animating = False

        self._create_menu()
        self._create_main_layout()

    def _configure_styles(self):
        """Configure ttk styles"""
        self.style.configure('TFrame', background=self.colors['bg'])
        self.style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'])
        self.style.configure('TButton', padding=6)
        self.style.configure('TNotebook', background=self.colors['bg'])
        self.style.configure('TNotebook.Tab', padding=[12, 6])
        self.style.configure('TLabelframe', background=self.colors['bg'])
        self.style.configure('TLabelframe.Label', background=self.colors['bg'], foreground=self.colors['fg'])

    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export CSV", command=self.export_csv)
        file_menu.add_command(label="Export JSON", command=self.export_json)
        file_menu.add_command(label="Save Figure", command=self.save_figure)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_docs)

    def _create_main_layout(self):
        """Create main application layout"""
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for controls
        self.left_panel = ttk.Frame(self.main_container, width=400)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_panel.pack_propagate(False)

        # Right panel for plots
        self.right_panel = ttk.Frame(self.main_container)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Wave tab
        self.wave_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.wave_frame, text="  Wave Equation  ")
        self._create_wave_controls()

        # Heat tab
        self.heat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.heat_frame, text="  Heat Equation  ")
        self._create_heat_controls()

        # Plot area
        self._create_plot_area()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_scrollable_frame(self, parent):
        """Create a scrollable frame"""
        canvas = tk.Canvas(parent, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return scrollable_frame

    def _create_wave_controls(self):
        """Create wave equation controls"""
        scrollable_frame = self._create_scrollable_frame(self.wave_frame)

        # Parameters frame
        params_frame = ttk.LabelFrame(scrollable_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(params_frame, text="Length (L):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.wave_L_var = tk.DoubleVar(value=1.0)
        ttk.Entry(params_frame, textvariable=self.wave_L_var, width=15).grid(row=0, column=1, pady=2)

        ttk.Label(params_frame, text="Wave Speed (c):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.wave_c_var = tk.DoubleVar(value=1.0)
        ttk.Entry(params_frame, textvariable=self.wave_c_var, width=15).grid(row=1, column=1, pady=2)

        ttk.Label(params_frame, text="Fourier Terms:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.wave_terms_var = tk.IntVar(value=20)
        ttk.Spinbox(params_frame, from_=1, to=200, textvariable=self.wave_terms_var, width=13).grid(row=2, column=1, pady=2)

        ttk.Label(params_frame, text="Sim Time:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.wave_time_var = tk.DoubleVar(value=4.0)
        ttk.Entry(params_frame, textvariable=self.wave_time_var, width=15).grid(row=3, column=1, pady=2)

        # Initial Shape frame
        shape_frame = ttk.LabelFrame(scrollable_frame, text="Initial Shape f(x)", padding=10)
        shape_frame.pack(fill=tk.X, padx=5, pady=5)

        self.wave_shape_var = tk.StringVar(value="triangular")
        shapes = [("Triangular", "triangular"), ("Sinusoidal", "sinusoidal"),
                  ("Plucked", "plucked"), ("Gaussian", "gaussian")]

        for i, (text, val) in enumerate(shapes):
            ttk.Radiobutton(shape_frame, text=text, variable=self.wave_shape_var, value=val).grid(
                row=i // 2, column=i % 2, sticky=tk.W, padx=5, pady=2)

        ttk.Label(shape_frame, text="Amplitude:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.wave_amp_var = tk.DoubleVar(value=0.5)
        ttk.Scale(shape_frame, from_=0.1, to=1.0, variable=self.wave_amp_var, orient=tk.HORIZONTAL).grid(
            row=2, column=1, sticky=tk.EW, pady=2)

        # Initial Velocity frame
        velocity_frame = ttk.LabelFrame(scrollable_frame, text="Initial Velocity g(x)", padding=10)
        velocity_frame.pack(fill=tk.X, padx=5, pady=5)

        self.wave_velocity_var = tk.StringVar(value="zero")
        velocities = [("Zero", "zero"), ("Sinusoidal", "sinusoidal"),
                      ("Gaussian", "gaussian"), ("Plucked", "plucked")]

        for i, (text, val) in enumerate(velocities):
            ttk.Radiobutton(velocity_frame, text=text, variable=self.wave_velocity_var, value=val).grid(
                row=i // 2, column=i % 2, sticky=tk.W, padx=5, pady=2)

        ttk.Label(velocity_frame, text="Velocity Amp:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.wave_vel_amp_var = tk.DoubleVar(value=0.0)
        ttk.Entry(velocity_frame, textvariable=self.wave_vel_amp_var, width=15).grid(row=2, column=1, pady=2)

        hint_label = ttk.Label(velocity_frame, text="(Set Amp > 0 to enable)", font=('Helvetica', 8))
        hint_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Buttons frame
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        ttk.Button(btn_frame, text="Compute", command=self.compute_wave).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Plot", command=self.plot_wave).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Animate", command=self.animate_wave).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Stop", command=self.stop_animation).pack(side=tk.LEFT, padx=2)

        # Analysis frame
        analysis_frame = ttk.LabelFrame(scrollable_frame, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(analysis_frame, text="Natural Modes", command=self.show_wave_modes).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Nodal Points", command=self.show_nodal_points).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Energy Plot", command=self.show_energy_plot).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Fourier Spectrum", command=self.show_wave_spectrum).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="D'Alembert View", command=self.show_dalembert).pack(fill=tk.X, pady=2)

        # Info display
        self.wave_info_text = tk.Text(scrollable_frame, height=10, width=40, 
                                       bg=self.colors['panel'], fg=self.colors['fg'])
        self.wave_info_text.pack(fill=tk.X, padx=5, pady=5)

    def _create_heat_controls(self):
        """Create heat equation controls"""
        scrollable_frame = self._create_scrollable_frame(self.heat_frame)

        # Parameters frame
        params_frame = ttk.LabelFrame(scrollable_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(params_frame, text="Length (L):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.heat_L_var = tk.DoubleVar(value=1.0)
        ttk.Entry(params_frame, textvariable=self.heat_L_var, width=15).grid(row=0, column=1, pady=2)

        ttk.Label(params_frame, text="Material:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.heat_material_var = tk.StringVar(value="Custom")
        material_combo = ttk.Combobox(params_frame, textvariable=self.heat_material_var,
                                      values=list(HeatDiffusion.MATERIALS.keys()), width=12)
        material_combo.grid(row=1, column=1, pady=2)
        material_combo.bind('<<ComboboxSelected>>', self.on_material_change)

        ttk.Label(params_frame, text="Diffusivity (α):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.heat_alpha_var = tk.DoubleVar(value=0.01)
        ttk.Entry(params_frame, textvariable=self.heat_alpha_var, width=15).grid(row=2, column=1, pady=2)

        ttk.Label(params_frame, text="Fourier Terms:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.heat_terms_var = tk.IntVar(value=50)
        ttk.Spinbox(params_frame, from_=1, to=200, textvariable=self.heat_terms_var, width=13).grid(row=3, column=1, pady=2)

        ttk.Label(params_frame, text="Sim Time:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.heat_time_var = tk.DoubleVar(value=10.0)
        ttk.Entry(params_frame, textvariable=self.heat_time_var, width=15).grid(row=4, column=1, pady=2)

        # Boundary Conditions frame
        bc_frame = ttk.LabelFrame(scrollable_frame, text="Boundary Conditions", padding=10)
        bc_frame.pack(fill=tk.X, padx=5, pady=5)

        self.heat_bc_var = tk.StringVar(value="dirichlet")
        ttk.Radiobutton(bc_frame, text="Dirichlet (T=const)", variable=self.heat_bc_var, 
                        value="dirichlet").pack(anchor=tk.W)
        ttk.Radiobutton(bc_frame, text="Neumann (Insulated)", variable=self.heat_bc_var, 
                        value="neumann").pack(anchor=tk.W)

        bc_temp_frame = ttk.Frame(bc_frame)
        bc_temp_frame.pack(fill=tk.X, pady=5)
        ttk.Label(bc_temp_frame, text="T(0):").pack(side=tk.LEFT)
        self.heat_T_left_var = tk.DoubleVar(value=0)
        ttk.Entry(bc_temp_frame, textvariable=self.heat_T_left_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(bc_temp_frame, text="T(L):").pack(side=tk.LEFT)
        self.heat_T_right_var = tk.DoubleVar(value=0)
        ttk.Entry(bc_temp_frame, textvariable=self.heat_T_right_var, width=8).pack(side=tk.LEFT, padx=5)

        # Initial Temperature frame
        init_frame = ttk.LabelFrame(scrollable_frame, text="Initial Temperature", padding=10)
        init_frame.pack(fill=tk.X, padx=5, pady=5)

        self.heat_init_var = tk.StringVar(value="sinusoidal")
        init_types = [("Sinusoidal", "sinusoidal"), ("Triangular", "triangular"),
                      ("Step", "step"), ("Gaussian", "gaussian"), ("Uniform", "uniform")]

        for i, (text, val) in enumerate(init_types):
            ttk.Radiobutton(init_frame, text=text, variable=self.heat_init_var, value=val).grid(
                row=i // 2, column=i % 2, sticky=tk.W, padx=5, pady=2)

        ttk.Label(init_frame, text="Max Temp:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.heat_max_temp_var = tk.DoubleVar(value=100)
        ttk.Entry(init_frame, textvariable=self.heat_max_temp_var, width=15).grid(row=3, column=1, pady=2)

        # Buttons frame
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        ttk.Button(btn_frame, text="Compute", command=self.compute_heat).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Plot", command=self.plot_heat).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Animate", command=self.animate_heat).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Stop", command=self.stop_animation).pack(side=tk.LEFT, padx=2)

        # Analysis frame
        analysis_frame = ttk.LabelFrame(scrollable_frame, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(analysis_frame, text="Compare Materials", command=self.compare_materials).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Steady State", command=self.show_steady_state).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Center Temp vs Time", command=self.show_center_temp).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Numerical Analysis", command=self.show_numerical_analysis).pack(fill=tk.X, pady=2)

        # Info display
        self.heat_info_text = tk.Text(scrollable_frame, height=8, width=40, 
                                       bg=self.colors['panel'], fg=self.colors['fg'])
        self.heat_info_text.pack(fill=tk.X, padx=5, pady=5)

    def _create_plot_area(self):
        """Create matplotlib plot area"""
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#f5f5f5')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(self.right_panel)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    # ==================== WAVE METHODS ====================

    def get_wave_initial_shape(self):
        """Get initial shape function based on selection"""
        L = self.wave_L_var.get()
        amp = self.wave_amp_var.get()
        shape = self.wave_shape_var.get()

        if shape == "triangular":
            return lambda x: np.where(x <= L / 2, 2 * amp * x / L, 2 * amp * (L - x) / L)
        elif shape == "sinusoidal":
            return lambda x: amp * np.sin(np.pi * x / L)
        elif shape == "plucked":
            x0 = L / 4
            return lambda x: np.where(x <= x0, amp * x / x0, amp * (L - x) / (L - x0))
        elif shape == "gaussian":
            return lambda x: amp * np.exp(-50 * (x - L / 2) ** 2)
        return lambda x: amp * np.sin(np.pi * x / L)

    def get_wave_initial_velocity(self):
        """Get initial velocity function based on selection"""
        L = self.wave_L_var.get()
        amp = self.wave_vel_amp_var.get()
        vel_type = self.wave_velocity_var.get()

        if vel_type == "zero" or amp == 0:
            return lambda x: np.zeros_like(x)
        elif vel_type == "sinusoidal":
            return lambda x: amp * np.sin(np.pi * x / L)
        elif vel_type == "gaussian":
            return lambda x: amp * np.exp(-50 * (x - L / 2) ** 2)
        elif vel_type == "plucked":
            return lambda x: np.where(x <= L / 2, 2 * amp * x / L, 2 * amp * (L - x) / L)
        return lambda x: np.zeros_like(x)

    def compute_wave(self):
        """Compute wave equation solution"""
        try:
            L = self.wave_L_var.get()
            c = self.wave_c_var.get()
            num_terms = self.wave_terms_var.get()

            self.wave_sim = VibratingString(L=L, c=c, num_terms=num_terms)

            f = self.get_wave_initial_shape()
            g = self.get_wave_initial_velocity()

            self.wave_sim.compute_coefficients(f, g)

            # Update info display
            self.wave_info_text.delete(1.0, tk.END)
            self.wave_info_text.insert(tk.END, "═" * 35 + "\n")
            self.wave_info_text.insert(tk.END, "  WAVE SIMULATION COMPUTED\n")
            self.wave_info_text.insert(tk.END, "═" * 35 + "\n\n")
            self.wave_info_text.insert(tk.END, f"Parameters:\n")
            self.wave_info_text.insert(tk.END, f"  L = {L}, c = {c}\n")
            self.wave_info_text.insert(tk.END, f"  Fourier Terms = {num_terms}\n")
            self.wave_info_text.insert(tk.END, f"  Period T = {2 * L / c:.4f} s\n\n")
            self.wave_info_text.insert(tk.END, f"Initial Conditions:\n")
            self.wave_info_text.insert(tk.END, f"  Shape: {self.wave_shape_var.get()}\n")
            self.wave_info_text.insert(tk.END, f"  Velocity: {self.wave_velocity_var.get()}\n")
            self.wave_info_text.insert(tk.END, f"  Vel. Amp: {self.wave_vel_amp_var.get()}\n\n")
            self.wave_info_text.insert(tk.END, "Coefficients (first 5):\n")
            for i in range(min(5, num_terms)):
                self.wave_info_text.insert(tk.END, f"  A{i + 1} = {self.wave_sim.An[i]:+.6f}\n")
                self.wave_info_text.insert(tk.END, f"  B{i + 1} = {self.wave_sim.Bn[i]:+.6f}\n")

            self.status_var.set("Wave solution computed successfully")
            messagebox.showinfo("Success", "Wave solution computed!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_wave(self):
        """Plot wave solution at different times"""
        if self.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        L = self.wave_L_var.get()
        T = self.wave_time_var.get()
        x = np.linspace(0, L, 500)
        times = [0, T / 6, T / 3, T / 2, 2 * T / 3, T]

        for i, t in enumerate(times):
            ax = self.fig.add_subplot(2, 3, i + 1)
            u = self.wave_sim.solution(x, t)
            ax.plot(x, u, 'b-', linewidth=2)
            ax.axhline(0, color='k', linewidth=0.5)
            ax.fill_between(x, 0, u, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('u(x,t)')
            ax.set_title(f't = {t:.3f}')
            ax.set_ylim([-self.wave_amp_var.get() * 1.5, self.wave_amp_var.get() * 1.5])
            ax.grid(True, alpha=0.3)

        self.fig.suptitle('Wave Equation - Time Evolution', fontsize=14, fontweight='bold')
        self.fig.tight_layout()
        self.canvas.draw()

    def animate_wave(self):
        """Animate wave solution"""
        if self.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        L = self.wave_L_var.get()
        T = self.wave_time_var.get()
        amp = self.wave_amp_var.get()
        x = np.linspace(0, L, 500)

        ax = self.fig.add_subplot(111)
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

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(frame):
            t = frame * T / 200
            u = self.wave_sim.solution(x, t)
            line.set_data(x, u)
            time_text.set_text(f't = {t:.3f} s')
            return line, time_text

        self.animation = FuncAnimation(self.fig, animate, init_func=init, frames=200,
                                       interval=50, blit=True, repeat=True)
        self.canvas.draw()
        self.status_var.set("Animation running...")

    def show_wave_modes(self):
        """Show natural modes of vibration"""
        if self.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        L = self.wave_L_var.get()
        x = np.linspace(0, L, 500)
        colors = plt.cm.viridis(np.linspace(0, 1, 5))

        ax = self.fig.add_subplot(111)
        for n in range(1, 6):
            mode = self.wave_sim.get_mode(n, x)
            freq = self.wave_sim.get_natural_frequency(n)
            ax.plot(x, mode + 2.5 * (5 - n), color=colors[n - 1], linewidth=2,
                    label=f'Mode {n}: f = {freq:.3f} Hz')
            ax.axhline(y=2.5 * (5 - n), color='gray', linewidth=0.5, linestyle='--')

        ax.set_xlabel('Position x')
        ax.set_ylabel('Mode Shape (offset)')
        ax.set_title('Natural Modes of Vibrating String: φₙ(x) = sin(nπx/L)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def show_nodal_points(self):
        """Show nodal points for each mode"""
        if self.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        L = self.wave_L_var.get()
        ax = self.fig.add_subplot(111)
        colors = plt.cm.Set1(np.linspace(0, 1, 7))

        for n in range(1, 8):
            nodes = self.wave_sim.get_nodal_points(n)
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
        self.fig.tight_layout()
        self.canvas.draw()

    def show_energy_plot(self):
        """Show energy conservation plot"""
        if self.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        L = self.wave_L_var.get()
        T = self.wave_time_var.get()
        x = np.linspace(0, L, 500)
        t_range = np.linspace(0, T, 100)

        kinetic, potential, total = [], [], []
        for t in t_range:
            K, P, E = self.wave_sim.compute_energy(x, t)
            kinetic.append(K)
            potential.append(P)
            total.append(E)

        ax = self.fig.add_subplot(111)
        ax.plot(t_range, kinetic, 'b-', linewidth=2, label='Kinetic Energy')
        ax.plot(t_range, potential, 'r-', linewidth=2, label='Potential Energy')
        ax.plot(t_range, total, 'g--', linewidth=2.5, label='Total Energy')

        E_mean = np.mean(total)
        E_std = np.std(total)
        ax.axhline(y=E_mean, color='purple', linestyle=':', alpha=0.7)
        ax.text(0.98, 0.02, f'Energy variation: {100 * E_std / E_mean:.4f}%',
                transform=ax.transAxes, ha='right', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Conservation in Vibrating String')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def show_wave_spectrum(self):
        """Show Fourier spectrum"""
        if self.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        n_vals = np.arange(1, len(self.wave_sim.An) + 1)

        ax1.stem(n_vals, self.wave_sim.An, basefmt=' ', linefmt='b-', markerfmt='bo')
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.set_xlabel('Mode Number (n)')
        ax1.set_ylabel('Aₙ')
        ax1.set_title('Coefficients Aₙ (from initial shape)')
        ax1.grid(True, alpha=0.3)

        ax2.stem(n_vals, self.wave_sim.Bn, basefmt=' ', linefmt='r-', markerfmt='ro')
        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.set_xlabel('Mode Number (n)')
        ax2.set_ylabel('Bₙ')
        ax2.set_title('Coefficients Bₙ (from initial velocity)')
        ax2.grid(True, alpha=0.3)

        self.fig.suptitle('Fourier Series Coefficients', fontsize=14, fontweight='bold')
        self.fig.tight_layout()
        self.canvas.draw()

    def show_dalembert(self):
        """Show D'Alembert solution visualization"""
        if self.wave_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        L = self.wave_L_var.get()
        c = self.wave_c_var.get()
        T = self.wave_time_var.get()
        amp = self.wave_amp_var.get()

        f = self.get_wave_initial_shape()
        f_ext = self.wave_sim.create_odd_periodic_extension(f)
        x_ext = np.linspace(-2 * L, 3 * L, 1000)
        times = [0, T / 8, T / 4, T / 2]

        for i, t in enumerate(times):
            ax = self.fig.add_subplot(2, 2, i + 1)
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

        self.fig.suptitle("D'Alembert Solution: u(x,t) = ½[F(x-ct) + F(x+ct)]",
                          fontsize=12, fontweight='bold')
        self.fig.tight_layout()
        self.canvas.draw()

    # ==================== HEAT METHODS ====================

    def on_material_change(self, event):
        """Handle material selection change"""
        material = self.heat_material_var.get()
        alpha = HeatDiffusion.MATERIALS.get(material, 0.01)
        self.heat_alpha_var.set(alpha)

    def get_heat_initial_temp(self):
        """Get initial temperature function based on selection"""
        L = self.heat_L_var.get()
        T_max = self.heat_max_temp_var.get()
        init_type = self.heat_init_var.get()

        if init_type == "sinusoidal":
            return lambda x: T_max * np.sin(np.pi * x / L)
        elif init_type == "triangular":
            return lambda x: T_max * (1 - 2 * np.abs(x - L / 2) / L)
        elif init_type == "step":
            return lambda x: np.where(x < L / 2, T_max, 0.0)
        elif init_type == "gaussian":
            return lambda x: T_max * np.exp(-50 * (x - L / 2) ** 2)
        elif init_type == "uniform":
            return lambda x: T_max * np.ones_like(x)
        return lambda x: T_max * np.sin(np.pi * x / L)

    def compute_heat(self):
        """Compute heat equation solution"""
        try:
            L = self.heat_L_var.get()
            alpha = self.heat_alpha_var.get()
            num_terms = self.heat_terms_var.get()
            bc_type = self.heat_bc_var.get()

            self.heat_sim = HeatDiffusion(L=L, alpha=alpha, num_terms=num_terms, boundary_type=bc_type)
            if bc_type == 'dirichlet':
                self.heat_sim.set_boundary_temperatures(self.heat_T_left_var.get(), self.heat_T_right_var.get())

            f = self.get_heat_initial_temp()
            self.heat_sim.compute_coefficients(f)

            tau = self.heat_sim.get_decay_constant(1)

            self.heat_info_text.delete(1.0, tk.END)
            self.heat_info_text.insert(tk.END, "═" * 35 + "\n")
            self.heat_info_text.insert(tk.END, "  HEAT SIMULATION COMPUTED\n")
            self.heat_info_text.insert(tk.END, "═" * 35 + "\n\n")
            self.heat_info_text.insert(tk.END, f"L = {L}, α = {alpha:.2e}\n")
            self.heat_info_text.insert(tk.END, f"BC: {bc_type.capitalize()}\n")
            self.heat_info_text.insert(tk.END, f"Fourier Terms = {num_terms}\n")
            self.heat_info_text.insert(tk.END, f"Time constant τ₁ = {tau:.4f} s\n")
            self.heat_info_text.insert(tk.END, f"Est. steady state: {5 * tau:.2f} s\n")

            self.status_var.set("Heat solution computed successfully")
            messagebox.showinfo("Success", "Heat solution computed!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_heat(self):
        """Plot heat solution at different times"""
        if self.heat_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        L = self.heat_L_var.get()
        T = self.heat_time_var.get()
        x = np.linspace(0, L, 500)
        times = [0, T / 10, T / 4, T / 2, 3 * T / 4, T]

        for i, t in enumerate(times):
            ax = self.fig.add_subplot(2, 3, i + 1)
            temp = self.heat_sim.solution(x, t)
            ax.plot(x, temp, 'r-', linewidth=2)
            ax.fill_between(x, 0, temp, alpha=0.3, color='red')
            ax.plot(x, self.heat_sim.steady_state(x), 'g--', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('x')
            ax.set_ylabel('T(x,t)')
            ax.set_title(f't = {t:.3f}')
            ax.grid(True, alpha=0.3)

        self.fig.suptitle('Heat Equation - Temperature Evolution', fontsize=14, fontweight='bold')
        self.fig.tight_layout()
        self.canvas.draw()

    def animate_heat(self):
        """Animate heat solution"""
        if self.heat_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        L = self.heat_L_var.get()
        T = self.heat_time_var.get()
        T_max = self.heat_max_temp_var.get()
        x = np.linspace(0, L, 500)

        ax = self.fig.add_subplot(111)
        line, = ax.plot([], [], 'r-', linewidth=2)
        ax.plot(x, self.heat_sim.steady_state(x), 'g--', linewidth=1.5, alpha=0.7, label='Steady State')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlim(0, L)
        ax.set_ylim(-T_max * 0.1, T_max * 1.2)
        ax.set_xlabel('Position x')
        ax.set_ylabel('Temperature T(x,t)')
        ax.set_title('Heat Diffusion Animation')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(frame):
            t = frame * T / 150
            temp = self.heat_sim.solution(x, t)
            line.set_data(x, temp)
            time_text.set_text(f't = {t:.3f} s')
            return line, time_text

        self.animation = FuncAnimation(self.fig, animate, init_func=init, frames=150,
                                       interval=50, blit=True, repeat=True)
        self.canvas.draw()
        self.status_var.set("Animation running...")

    def compare_materials(self):
        """Compare different materials"""
        self.stop_animation()
        self.fig.clear()

        L = self.heat_L_var.get()
        num_terms = self.heat_terms_var.get()
        x = np.linspace(0, L, 200)
        f = self.get_heat_initial_temp()

        materials = ['Copper', 'Aluminum', 'Iron', 'Steel', 'Glass', 'Wood']
        colors = plt.cm.viridis(np.linspace(0, 1, len(materials)))

        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        t_fixed = 1.0
        decay_times = []

        for i, mat in enumerate(materials):
            alpha = HeatDiffusion.MATERIALS[mat]
            heat = HeatDiffusion(L=L, alpha=alpha, num_terms=num_terms, boundary_type=self.heat_bc_var.get())
            heat.compute_coefficients(f)
            temp = heat.solution(x, t_fixed)
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
#by-mahdysp
        self.fig.suptitle('Material Comparison', fontsize=14, fontweight='bold')
        self.fig.tight_layout()
        self.canvas.draw()

    def show_steady_state(self):
        """Show approach to steady state"""
        if self.heat_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        L = self.heat_L_var.get()
        T = self.heat_time_var.get()
        x = np.linspace(0, L, 200)

        ax = self.fig.add_subplot(111)
        times = np.linspace(0, T, 10)
        colors = plt.cm.hot(np.linspace(0.8, 0.2, len(times)))

        for i, t in enumerate(times):
            temp = self.heat_sim.solution(x, t)
            ax.plot(x, temp, color=colors[i], linewidth=2, label=f't = {t:.2f}')

        ax.plot(x, self.heat_sim.steady_state(x), 'g--', linewidth=3, label='Steady State')
        ax.set_xlabel('Position x')
        ax.set_ylabel('Temperature')
        ax.set_title('Approach to Steady State')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def show_center_temp(self):
        """Show center temperature vs time"""
        if self.heat_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        L = self.heat_L_var.get()
        T = self.heat_time_var.get()
        t_range = np.linspace(0, T, 500)
        x_center = np.array([L / 2])

        T_center = [self.heat_sim.solution(x_center, t)[0] for t in t_range]
        steady = self.heat_sim.steady_state(x_center)[0]
        tau = self.heat_sim.get_decay_constant(1)

        ax = self.fig.add_subplot(111)
        ax.plot(t_range, T_center, 'b-', linewidth=2, label='T(L/2, t)')
        ax.axhline(steady, color='r', linestyle='--', linewidth=2, label=f'Steady = {steady:.2f}')

        for n in [1, 2, 3, 5]:
            if n * tau < T:
                ax.axvline(x=n * tau, color='green', linestyle=':', alpha=0.7)
                ax.text(n * tau, ax.get_ylim()[1] * 0.95, f'{n}τ', ha='center', fontsize=9, color='green')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature at Center')
        ax.set_title(f'Center Temperature vs Time (τ₁ = {tau:.4f} s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def show_numerical_analysis(self):
        """Show numerical analysis table"""
        if self.heat_sim is None:
            messagebox.showwarning("Warning", "Compute first!")
            return
        self.stop_animation()
        self.fig.clear()

        L = self.heat_L_var.get()
        x_center = L / 2

        ax = self.fig.add_subplot(111)
        ax.axis('off')

        percentages = [90, 75, 50, 25, 10, 5, 1]
        results = []
        for pct in percentages:
            t = self.heat_sim.time_to_percentage(x_center, pct)
            results.append([f"{pct}%", f"{t:.4f} s" if t else "N/A"])

        table_data = [["% Remaining", "Time Required"]] + results
        table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        for i in range(2):
            table[(0, i)].set_facecolor('#4a9eff')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        initial_temp = self.heat_sim.solution(np.array([x_center]), 0)[0]
        steady_temp = self.heat_sim.steady_state(np.array([x_center]))[0]
        tau = self.heat_sim.get_decay_constant(1)

        ax.set_title(f'Numerical Analysis: Time to Reach Temperature\n'
                     f'Center (x = L/2) | Initial: {initial_temp:.1f} | Steady: {steady_temp:.1f} | τ₁ = {tau:.4f}s',
                     fontsize=12, fontweight='bold', pad=20)
        self.fig.tight_layout()
        self.canvas.draw()

    # ==================== UTILITY METHODS ====================

    def stop_animation(self):
        """Stop any running animation"""
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
        self.status_var.set("Animation stopped")

    def export_csv(self):
        """Export simulation data to CSV"""
        if self.wave_sim is None and self.heat_sim is None:
            messagebox.showwarning("Warning", "No data to export!")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if filename:
            sim = self.wave_sim if self.wave_sim else self.heat_sim
            L = self.wave_L_var.get() if self.wave_sim else self.heat_L_var.get()
            T = self.wave_time_var.get() if self.wave_sim else self.heat_time_var.get()
            x = np.linspace(0, L, 100)
            t_array = np.linspace(0, T, 50)

            with open(filename, 'w') as f:
                f.write("t," + ",".join([f"{xi:.4f}" for xi in x]) + "\n")
                for t in t_array:
                    u = sim.solution(x, t)
                    f.write(f"{t:.4f}," + ",".join([f"{ui:.6f}" for ui in u]) + "\n")
            messagebox.showinfo("Success", f"Exported to {filename}")
#by-mahdysp
    def export_json(self):
        """Export parameters to JSON"""
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if filename:
            data = {
                "wave": {
                    "L": self.wave_L_var.get(),
                    "c": self.wave_c_var.get(),
                    "terms": self.wave_terms_var.get(),
                    "shape": self.wave_shape_var.get(),
                    "amplitude": self.wave_amp_var.get(),
                    "velocity_type": self.wave_velocity_var.get(),
                    "velocity_amplitude": self.wave_vel_amp_var.get()
                },
                "heat": {
                    "L": self.heat_L_var.get(),
                    "alpha": self.heat_alpha_var.get(),
                    "terms": self.heat_terms_var.get(),
                    "bc_type": self.heat_bc_var.get(),
                    "init_type": self.heat_init_var.get()
                }
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Success", f"Exported to {filename}")

    def save_figure(self):
        """Save current figure"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if filename:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            messagebox.showinfo("Success", f"Saved to {filename}")

    def show_about(self):
        """Show about dialog"""
        about_text = """
Wave & Heat Equation Simulator
Version 1.0

Features:
• Wave Equation with initial shape AND velocity
• Heat Equation with Dirichlet/Neumann BC
• D'Alembert solution visualization
• Numerical time analysis
• Material comparison
• Energy conservation check

Using Fourier Series analytical solutions.

GitHub: github.com/yourusername/wave-heat-simulator
        """
        messagebox.showinfo("About", about_text)

    def show_docs(self):
        """Show documentation window"""
        docs_window = tk.Toplevel(self.root)
        docs_window.title("Documentation")
        docs_window.geometry("600x500")

        text = tk.Text(docs_window, wrap=tk.WORD, padx=15, pady=15)
        text.pack(fill=tk.BOTH, expand=True)

        docs = """
DOCUMENTATION
=============

WAVE EQUATION
-------------
The wave equation: ∂²u/∂t² = c² ∂²u/∂x²

Initial Conditions:
  • f(x) = u(x,0)  - Initial displacement
  • g(x) = ∂u/∂t(x,0) - Initial velocity

Solution:
  u(x,t) = Σ [Aₙcos(ωₙt) + Bₙsin(ωₙt)] sin(nπx/L)

Where:
  • Aₙ comes from initial shape f(x)
  • Bₙ comes from initial velocity g(x)
  • ωₙ = nπc/L

HEAT EQUATION
-------------
The heat equation: ∂u/∂t = α ∂²u/∂x²

Boundary Conditions:
  • Dirichlet: T(0,t) = T₀, T(L,t) = T₁
  • Neumann: ∂T/∂x = 0 (insulated)

Time constant: τₙ = L²/(n²π²α)

HOW TO USE
----------
1. Set parameters (L, c or α, terms)
2. Choose initial conditions
3. Click "Compute"
4. Use "Plot", "Animate", or analysis buttons
        """
        text.insert(tk.END, docs)
        text.config(state=tk.DISABLED)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
