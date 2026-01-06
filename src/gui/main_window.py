"""Main window and application controller"""

import tkinter as tk
from tkinter import ttk, messagebox
import warnings

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ..physics.wave_equation import VibratingString
from ..physics.heat_equation import HeatDiffusion
from .styles import COLORS, configure_styles
from .wave_controls import create_wave_controls
from .heat_controls import create_heat_controls
from .plotting import PlottingManager
from ..utils.export import export_to_csv, export_to_json, save_figure

warnings.filterwarnings('ignore')


class SimulationGUI:
    """Main GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Wave & Heat Equation Simulator - Complete Version")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Setup style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.colors = COLORS
        configure_styles(self.style)
        
        self.root.configure(bg=self.colors['bg'])
        
        # Simulation objects
        self.wave_sim = None
        self.heat_sim = None
        self.animation = None
        self.is_animating = False
        
        # Plotting manager
        self.plotter = PlottingManager(self)
        
        # Create GUI
        self._create_menu()
        self._create_main_layout()
        self._bind_shortcuts()
        
    def _create_menu(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export CSV (Ctrl+E)", command=self.export_csv_data)
        file_menu.add_command(label="Export JSON", command=self.export_json_data)
        file_menu.add_command(label="Save Figure (Ctrl+S)", command=self.save_figure_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_docs)
    
    def _create_main_layout(self):
        """Create main window layout"""
        # Main container
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
        
        # Wave equation tab
        self.wave_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.wave_frame, text="  Wave Equation  ")
        create_wave_controls(self, self.wave_frame)
        
        # Heat equation tab
        self.heat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.heat_frame, text="  Heat Equation  ")
        create_heat_controls(self, self.heat_frame)
        
        # Create plot area
        self._create_plot_area()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Press Space to toggle animation")
        self.status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN, 
            anchor=tk.W #by-mahdysp
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _create_plot_area(self):
        """Create matplotlib plot area"""
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#f5f5f5')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar_frame = ttk.Frame(self.right_panel)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
    
    def _bind_shortcuts(self):
        """Bind keyboard shortcuts"""
        self.root.bind('<Control-s>', lambda e: self.save_figure_dialog())
        self.root.bind('<Control-e>', lambda e: self.export_csv_data())
        self.root.bind('<Escape>', lambda e: self.stop_animation())
        self.root.bind('<space>', lambda e: self._toggle_animation())
    
    def _toggle_animation(self):
        """Toggle animation on/off with spacebar"""
        if self.animation is not None:
            self.stop_animation()
        else:
            current_tab = self.notebook.index(self.notebook.select())
            if current_tab == 0 and self.wave_sim is not None:
                self.plotter.animate_wave()
            elif current_tab == 1 and self.heat_sim is not None:
                self.plotter.animate_heat()
    
    def stop_animation(self):
        """Stop running animation"""
        if self.animation is not None:
            try:
                if hasattr(self.animation, 'event_source') and self.animation.event_source is not None:
                    self.animation.event_source.stop()
            except Exception:
                pass
            self.animation = None
        self.status_var.set("Animation stopped")
    
    def create_scrollable_frame(self, parent):
        """Create a scrollable frame with proper mouse wheel binding"""
        canvas = tk.Canvas(parent, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def configure_frame_width(event):
            canvas.itemconfig(canvas_frame, width=event.width)
        
        canvas.bind('<Configure>', configure_frame_width)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
        
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_mousewheel)
        canvas.bind("<Button-5>", _on_mousewheel)
        
        def bind_mousewheel(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel)
            widget.bind("<Button-5>", _on_mousewheel)
            for child in widget.winfo_children():
                bind_mousewheel(child)
        
        scrollable_frame.bind("<Map>", lambda e: bind_mousewheel(scrollable_frame))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return scrollable_frame
    
    # Export methods
    def export_csv_data(self):
        """Export current simulation data to CSV"""
        if self.wave_sim is None and self.heat_sim is None:
            messagebox.showwarning("Warning", "No data to export!")
            return
        
        sim = self.wave_sim if self.wave_sim else self.heat_sim
        L = self.wave_L_var.get() if self.wave_sim else self.heat_L_var.get()
        T = self.wave_time_var.get() if self.wave_sim else self.heat_time_var.get()
        export_to_csv(sim, L, T)
    
    def export_json_data(self):
        """Export parameters to JSON"""
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
        export_to_json(data)
    
    def save_figure_dialog(self):
        """Save current figure"""
        save_figure(self.fig)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Wave & Heat Equation Simulator
Version 1.0.0

Features:
• Wave Equation with initial shape AND velocity
• Heat Equation with Dirichlet/Neumann BC
• D'Alembert solution visualization
• Numerical time analysis
• Material comparison
• Energy conservation check

Keyboard Shortcuts:
• Space: Toggle animation
• Escape: Stop animation
• Ctrl+S: Save figure
• Ctrl+E: Export CSV

Using Fourier Series analytical solutions.
        """
        messagebox.showinfo("About", about_text)
    
    def show_docs(self):
        """Show documentation window"""
        docs_window = tk.Toplevel(self.root)
        docs_window.title("Documentation")
        docs_window.geometry("600x500")
        
        text = tk.Text(docs_window, wrap=tk.WORD, padx=15, pady=15)
        scrollbar = ttk.Scrollbar(docs_window, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)
        
        docs = """
DOCUMENTATION

WAVE EQUATION
The wave equation: ∂²u/∂t² = c² ∂²u/∂x²

Initial Conditions:
• f(x) = u(x,0) - Initial displacement
• g(x) = ∂u/∂t(x,0) - Initial velocity

HEAT EQUATION
The heat equation: ∂u/∂t = α ∂²u/∂x²

Boundary Conditions:
• Dirichlet: T(0,t) = T₀, T(L,t) = T₁
• Neumann: ∂T/∂x = 0 (insulated)

KEYBOARD SHORTCUTS
Space: Toggle animation
Escape: Stop animation
Ctrl+S: Save figure
Ctrl+E: Export CSV
        """
        text.insert(tk.END, docs)
        text.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle window close event"""
        self.stop_animation()
        self.root.quit()
