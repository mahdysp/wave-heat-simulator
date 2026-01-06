"""Heat equation control panel"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from ..physics.heat_equation import HeatDiffusion
from ..utils.validators import validate_heat_inputs


def create_heat_controls(app, parent):
    """Create heat equation control widgets"""
    scrollable_frame = app.create_scrollable_frame(parent)
    
    # Parameters frame
    params_frame = ttk.LabelFrame(scrollable_frame, text="Parameters", padding=10)
    params_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(params_frame, text="Length (L):").grid(row=0, column=0, sticky=tk.W, pady=2)
    app.heat_L_var = tk.DoubleVar(value=1.0)
    ttk.Entry(params_frame, textvariable=app.heat_L_var, width=15).grid(row=0, column=1, pady=2)
    
    ttk.Label(params_frame, text="Material:").grid(row=1, column=0, sticky=tk.W, pady=2)
    app.heat_material_var = tk.StringVar(value="Custom")
    material_combo = ttk.Combobox(params_frame, textvariable=app.heat_material_var,
                                  values=list(HeatDiffusion.MATERIALS.keys()), width=12)
    material_combo.grid(row=1, column=1, pady=2)
    material_combo.bind('<<ComboboxSelected>>', lambda e: on_material_change(app))
    
    ttk.Label(params_frame, text="Diffusivity (α):").grid(row=2, column=0, sticky=tk.W, pady=2)
    app.heat_alpha_var = tk.DoubleVar(value=0.01)
    ttk.Entry(params_frame, textvariable=app.heat_alpha_var, width=15).grid(row=2, column=1, pady=2)
    
    ttk.Label(params_frame, text="Fourier Terms:").grid(row=3, column=0, sticky=tk.W, pady=2)
    app.heat_terms_var = tk.IntVar(value=50)
    ttk.Spinbox(params_frame, from_=1, to=200, textvariable=app.heat_terms_var, width=13).grid(row=3, column=1, pady=2)
    
    ttk.Label(params_frame, text="Sim Time:").grid(row=4, column=0, sticky=tk.W, pady=2)
    app.heat_time_var = tk.DoubleVar(value=10.0)
    ttk.Entry(params_frame, textvariable=app.heat_time_var, width=15).grid(row=4, column=1, pady=2)
    
    # Boundary Conditions frame
    bc_frame = ttk.LabelFrame(scrollable_frame, text="Boundary Conditions", padding=10)
    bc_frame.pack(fill=tk.X, padx=5, pady=5)
    
    app.heat_bc_var = tk.StringVar(value="dirichlet")
    ttk.Radiobutton(bc_frame, text="Dirichlet (T=const)", 
                   variable=app.heat_bc_var, value="dirichlet").pack(anchor=tk.W)
    ttk.Radiobutton(bc_frame, text="Neumann (Insulated)", 
                   variable=app.heat_bc_var, value="neumann").pack(anchor=tk.W)
    
    bc_temp_frame = ttk.Frame(bc_frame)
    bc_temp_frame.pack(fill=tk.X, pady=5)
    ttk.Label(bc_temp_frame, text="T(0):").pack(side=tk.LEFT)
    app.heat_T_left_var = tk.DoubleVar(value=0)
    ttk.Entry(bc_temp_frame, textvariable=app.heat_T_left_var, width=8).pack(side=tk.LEFT, padx=5)
    ttk.Label(bc_temp_frame, text="T(L):").pack(side=tk.LEFT)
    app.heat_T_right_var = tk.DoubleVar(value=0)
    ttk.Entry(bc_temp_frame, textvariable=app.heat_T_right_var, width=8).pack(side=tk.LEFT, padx=5)
    
    # Initial Temperature frame
    init_frame = ttk.LabelFrame(scrollable_frame, text="Initial Temperature", padding=10)
    init_frame.pack(fill=tk.X, padx=5, pady=5)
    
    app.heat_init_var = tk.StringVar(value="sinusoidal")
    init_types = [("Sinusoidal", "sinusoidal"), ("Triangular", "triangular"),
                  ("Step", "step"), ("Gaussian", "gaussian"), ("Uniform", "uniform")]
    
    for i, (text, val) in enumerate(init_types):
        ttk.Radiobutton(init_frame, text=text, variable=app.heat_init_var, value=val).grid(
            row=i // 2, column=i % 2, sticky=tk.W, padx=5, pady=2)
    
    ttk.Label(init_frame, text="Max Temp:").grid(row=3, column=0, sticky=tk.W, pady=2)
    app.heat_max_temp_var = tk.DoubleVar(value=100)
    ttk.Entry(init_frame, textvariable=app.heat_max_temp_var, width=15).grid(row=3, column=1, pady=2)
    
    # Buttons frame
    btn_frame = ttk.Frame(scrollable_frame)
    btn_frame.pack(fill=tk.X, padx=5, pady=10)
    
    ttk.Button(btn_frame, text="Compute", 
               command=lambda: compute_heat(app)).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Plot", 
               command=lambda: app.plotter.plot_heat()).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Animate", 
               command=lambda: app.plotter.animate_heat()).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Stop",  #by-mahdysp
               command=app.stop_animation).pack(side=tk.LEFT, padx=2)
    
    # Analysis frame
    analysis_frame = ttk.LabelFrame(scrollable_frame, text="Analysis", padding=10)
    analysis_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Button(analysis_frame, text="Compare Materials", 
               command=lambda: app.plotter.compare_materials()).pack(fill=tk.X, pady=2)
    ttk.Button(analysis_frame, text="Steady State", 
               command=lambda: app.plotter.show_steady_state()).pack(fill=tk.X, pady=2)
    ttk.Button(analysis_frame, text="Center Temp vs Time", 
               command=lambda: app.plotter.show_center_temp()).pack(fill=tk.X, pady=2)
    ttk.Button(analysis_frame, text="Numerical Analysis", 
               command=lambda: app.plotter.show_numerical_analysis()).pack(fill=tk.X, pady=2)
    
    # Info display
    app.heat_info_text = tk.Text(scrollable_frame, height=8, width=40, 
                                bg=app.colors['panel'], fg=app.colors['fg'])
    app.heat_info_text.pack(fill=tk.X, padx=5, pady=5)


def on_material_change(app):
    """Update diffusivity when material changes"""
    material = app.heat_material_var.get()
    alpha = HeatDiffusion.MATERIALS.get(material, 0.01)
    app.heat_alpha_var.set(alpha)


def get_heat_initial_temp(app):
    """Get initial temperature function"""
    L = app.heat_L_var.get()
    T_max = app.heat_max_temp_var.get()
    init_type = app.heat_init_var.get()
    
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


def compute_heat(app):
    """Compute heat equation solution"""
    try:
        L = app.heat_L_var.get()
        alpha = app.heat_alpha_var.get()
        num_terms = app.heat_terms_var.get()
        sim_time = app.heat_time_var.get()
        bc_type = app.heat_bc_var.get()
        
        validate_heat_inputs(L, alpha, num_terms, sim_time)
        
        app.status_var.set("Computing heat solution...")
        app.root.update()
        
        app.heat_sim = HeatDiffusion(L=L, alpha=alpha, num_terms=num_terms, boundary_type=bc_type)
        
        if bc_type == 'dirichlet':
            app.heat_sim.set_boundary_temperatures(
                app.heat_T_left_var.get(), 
                app.heat_T_right_var.get()
            )
        
        f = get_heat_initial_temp(app)
        app.heat_sim.compute_coefficients(f)
        
        tau = app.heat_sim.get_decay_constant(1)
        
        # Update info display
        app.heat_info_text.delete(1.0, tk.END)
        app.heat_info_text.insert(tk.END, "═" * 35 + "\n")
        app.heat_info_text.insert(tk.END, "  HEAT SIMULATION COMPUTED\n")
        app.heat_info_text.insert(tk.END, "═" * 35 + "\n\n")
        app.heat_info_text.insert(tk.END, f"L = {L}, α = {alpha:.2e}\n")
        app.heat_info_text.insert(tk.END, f"BC: {bc_type.capitalize()}\n")
        app.heat_info_text.insert(tk.END, f"Fourier Terms = {num_terms}\n")
        app.heat_info_text.insert(tk.END, f"Time constant τ₁ = {tau:.4f} s\n")
        app.heat_info_text.insert(tk.END, f"Est. steady state: {5 * tau:.2f} s\n")
        
        app.status_var.set("Heat solution computed successfully")
        messagebox.showinfo("Success", "Heat solution computed!")
        
    except ValueError as e:
        messagebox.showerror("Validation Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", f"Computation failed: {str(e)}")
        app.status_var.set("Error during computation")
