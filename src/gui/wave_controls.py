"""Wave equation control panel"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from ..physics.wave_equation import VibratingString
from ..utils.validators import validate_wave_inputs


def create_wave_controls(app, parent):
    """Create wave equation control widgets"""
    scrollable_frame = app.create_scrollable_frame(parent)
    
    # Parameters frame
    params_frame = ttk.LabelFrame(scrollable_frame, text="Parameters", padding=10)
    params_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(params_frame, text="Length (L):").grid(row=0, column=0, sticky=tk.W, pady=2)
    app.wave_L_var = tk.DoubleVar(value=1.0)
    ttk.Entry(params_frame, textvariable=app.wave_L_var, width=15).grid(row=0, column=1, pady=2)
    
    ttk.Label(params_frame, text="Wave Speed (c):").grid(row=1, column=0, sticky=tk.W, pady=2)
    app.wave_c_var = tk.DoubleVar(value=1.0)
    ttk.Entry(params_frame, textvariable=app.wave_c_var, width=15).grid(row=1, column=1, pady=2)
    
    ttk.Label(params_frame, text="Fourier Terms:").grid(row=2, column=0, sticky=tk.W, pady=2)
    app.wave_terms_var = tk.IntVar(value=20)
    ttk.Spinbox(params_frame, from_=1, to=200, textvariable=app.wave_terms_var, width=13).grid(row=2, column=1, pady=2)
    
    ttk.Label(params_frame, text="Sim Time:").grid(row=3, column=0, sticky=tk.W, pady=2)
    app.wave_time_var = tk.DoubleVar(value=4.0)
    ttk.Entry(params_frame, textvariable=app.wave_time_var, width=15).grid(row=3, column=1, pady=2)
    
    # Initial Shape frame
    shape_frame = ttk.LabelFrame(scrollable_frame, text="Initial Shape f(x)", padding=10)
    shape_frame.pack(fill=tk.X, padx=5, pady=5)
    
    app.wave_shape_var = tk.StringVar(value="triangular")
    shapes = [("Triangular", "triangular"), ("Sinusoidal", "sinusoidal"),
              ("Plucked", "plucked"), ("Gaussian", "gaussian")]
    
    for i, (text, val) in enumerate(shapes):
        ttk.Radiobutton(shape_frame, text=text, variable=app.wave_shape_var, value=val).grid(
            row=i // 2, column=i % 2, sticky=tk.W, padx=5, pady=2)
    
    ttk.Label(shape_frame, text="Amplitude:").grid(row=2, column=0, sticky=tk.W, pady=2)
    app.wave_amp_var = tk.DoubleVar(value=0.5)
    ttk.Scale(shape_frame, from_=0.1, to=1.0, variable=app.wave_amp_var, orient=tk.HORIZONTAL).grid(
        row=2, column=1, sticky=tk.EW, pady=2)
    
    # Initial Velocity frame
    velocity_frame = ttk.LabelFrame(scrollable_frame, text="Initial Velocity g(x)", padding=10)
    velocity_frame.pack(fill=tk.X, padx=5, pady=5)
    
    app.wave_velocity_var = tk.StringVar(value="zero")
    velocities = [("Zero", "zero"), ("Sinusoidal", "sinusoidal"),
                  ("Gaussian", "gaussian"), ("Plucked", "plucked")]
    
    for i, (text, val) in enumerate(velocities): #by-mahdysp
        ttk.Radiobutton(velocity_frame, text=text, variable=app.wave_velocity_var, value=val).grid(
            row=i // 2, column=i % 2, sticky=tk.W, padx=5, pady=2)
    
    ttk.Label(velocity_frame, text="Velocity Amp:").grid(row=2, column=0, sticky=tk.W, pady=2)
    app.wave_vel_amp_var = tk.DoubleVar(value=0.0)
    ttk.Entry(velocity_frame, textvariable=app.wave_vel_amp_var, width=15).grid(row=2, column=1, pady=2)
    
    hint_label = ttk.Label(velocity_frame, text="(Set Amp > 0 to enable)", font=('Helvetica', 8))
    hint_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
    
    # Buttons frame
    btn_frame = ttk.Frame(scrollable_frame)
    btn_frame.pack(fill=tk.X, padx=5, pady=10)
    
    ttk.Button(btn_frame, text="Compute", 
               command=lambda: compute_wave(app)).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Plot", 
               command=lambda: app.plotter.plot_wave()).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Animate", 
               command=lambda: app.plotter.animate_wave()).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Stop", 
               command=app.stop_animation).pack(side=tk.LEFT, padx=2)
    
    # Analysis frame
    analysis_frame = ttk.LabelFrame(scrollable_frame, text="Analysis", padding=10)
    analysis_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Button(analysis_frame, text="Natural Modes", 
               command=lambda: app.plotter.show_wave_modes()).pack(fill=tk.X, pady=2)
    ttk.Button(analysis_frame, text="Nodal Points", 
               command=lambda: app.plotter.show_nodal_points()).pack(fill=tk.X, pady=2)
    ttk.Button(analysis_frame, text="Energy Plot", 
               command=lambda: app.plotter.show_energy_plot()).pack(fill=tk.X, pady=2)
    ttk.Button(analysis_frame, text="Fourier Spectrum", 
               command=lambda: app.plotter.show_wave_spectrum()).pack(fill=tk.X, pady=2)
    ttk.Button(analysis_frame, text="D'Alembert View", 
               command=lambda: app.plotter.show_dalembert()).pack(fill=tk.X, pady=2)
    
    # Info display
    app.wave_info_text = tk.Text(scrollable_frame, height=10, width=40, 
                                bg=app.colors['panel'], fg=app.colors['fg'])
    app.wave_info_text.pack(fill=tk.X, padx=5, pady=5)


def get_wave_initial_shape(app):
    """Get initial shape function based on user selection"""
    L = app.wave_L_var.get()
    amp = app.wave_amp_var.get()
    shape = app.wave_shape_var.get()
    
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


def get_wave_initial_velocity(app):
    """Get initial velocity function based on user selection"""
    L = app.wave_L_var.get()
    amp = app.wave_vel_amp_var.get()
    vel_type = app.wave_velocity_var.get()
    
    if vel_type == "zero" or amp == 0:
        return lambda x: np.zeros_like(x)
    elif vel_type == "sinusoidal":
        return lambda x: amp * np.sin(np.pi * x / L)
    elif vel_type == "gaussian":
        return lambda x: amp * np.exp(-50 * (x - L / 2) ** 2)
    elif vel_type == "plucked":
        return lambda x: np.where(x <= L / 2, 2 * amp * x / L, 2 * amp * (L - x) / L)
    
    return lambda x: np.zeros_like(x)


def compute_wave(app):
    """Compute wave equation solution"""
    try:
        L = app.wave_L_var.get()
        c = app.wave_c_var.get()
        num_terms = app.wave_terms_var.get()
        sim_time = app.wave_time_var.get()
        
        validate_wave_inputs(L, c, num_terms, sim_time)
        
        app.status_var.set("Computing wave solution...")
        app.root.update()
        
        app.wave_sim = VibratingString(L=L, c=c, num_terms=num_terms)
        
        f = get_wave_initial_shape(app)
        g = get_wave_initial_velocity(app)
        
        app.wave_sim.compute_coefficients(f, g)
        
        # Update info display
        app.wave_info_text.delete(1.0, tk.END)
        app.wave_info_text.insert(tk.END, "═" * 35 + "\n")
        app.wave_info_text.insert(tk.END, "  WAVE SIMULATION COMPUTED\n")
        app.wave_info_text.insert(tk.END, "═" * 35 + "\n\n")
        app.wave_info_text.insert(tk.END, f"Parameters:\n")
        app.wave_info_text.insert(tk.END, f"  L = {L}, c = {c}\n")
        app.wave_info_text.insert(tk.END, f"  Fourier Terms = {num_terms}\n")
        app.wave_info_text.insert(tk.END, f"  Period T = {2 * L / c:.4f} s\n\n")
        app.wave_info_text.insert(tk.END, f"Initial Conditions:\n")
        app.wave_info_text.insert(tk.END, f"  Shape: {app.wave_shape_var.get()}\n")
        app.wave_info_text.insert(tk.END, f"  Velocity: {app.wave_velocity_var.get()}\n")
        app.wave_info_text.insert(tk.END, f"  Vel. Amp: {app.wave_vel_amp_var.get()}\n\n")
        app.wave_info_text.insert(tk.END, "Coefficients (first 5):\n")
        for i in range(min(5, num_terms)):
            app.wave_info_text.insert(tk.END, f"  A{i + 1} = {app.wave_sim.An[i]:+.6f}\n")
            app.wave_info_text.insert(tk.END, f"  B{i + 1} = {app.wave_sim.Bn[i]:+.6f}\n")
        
        app.status_var.set("Wave solution computed successfully")
        messagebox.showinfo("Success", "Wave solution computed!")
        
    except ValueError as e:
        messagebox.showerror("Validation Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", f"Computation failed: {str(e)}")
        app.status_var.set("Error during computation")
