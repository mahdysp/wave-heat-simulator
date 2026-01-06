#!/usr/bin/env python3
"""
PDE Fourier Simulator - Main Entry Point
Interactive solver for wave and heat equations using Fourier series
"""

import sys
import tkinter as tk
from gui.main_window import SimulationGUI

def main():
    """Main entry point for the application"""
    try:
        root = tk.Tk()
        app = SimulationGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
