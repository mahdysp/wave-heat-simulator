<div align="center">

# 🌊 Wave & Heat Equation Simulator

**A powerful Python GUI application for simulating 1D wave and heat equations using Fourier series analytical solutions.**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20|%20Linux%20|%20macOS-lightgrey?style=for-the-badge)]()

<p align="center">
  <img src="screenshots/demo.gif" alt="Demo" width="700">
</p>

[Features](#-features) •
[Installation](#-installation) •
[Usage](#-usage) •
[Theory](#-theory) •
[Screenshots](#-screenshots) •
[Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Theory](#-theory)
- [Screenshots](#-screenshots)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

### 🌊 Wave Equation Solver
| Feature | Description |
|---------|-------------|
| **Initial Conditions** | Support for both displacement f(x) and velocity g(x) |
| **Shape Options** | Triangular, Sinusoidal, Plucked, Gaussian |
| **Natural Modes** | Visualize vibration modes φₙ(x) = sin(nπx/L) |
| **Nodal Points** | Display zero-displacement positions |
| **Energy Analysis** | Kinetic, potential, and total energy plots |
| **Fourier Spectrum** | View Aₙ and Bₙ coefficients |
| **D'Alembert View** | Traveling wave visualization |

### 🔥 Heat Equation Solver
| Feature | Description |
|---------|-------------|
| **Boundary Conditions** | Dirichlet (fixed T) and Neumann (insulated) |
| **Material Presets** | Copper, Aluminum, Iron, Steel, Silver, Gold, Glass, Wood |
| **Initial Temperature** | Sinusoidal, Triangular, Step, Gaussian, Uniform |
| **Steady State** | Visualize approach to equilibrium |
| **Time Analysis** | Calculate time to reach target temperature |
| **Material Comparison** | Compare thermal behavior of different materials |

### 🎨 General Features
- 🎬 **Real-time Animations** - Smooth visualization of time evolution
- 🌙 **Dark Theme GUI** - Easy on the eyes
- 💾 **Export Options** - CSV, JSON, PNG, PDF, SVG
- 📊 **Interactive Plots** - Zoom, pan, and save with matplotlib toolbar
- ⚡ **Fast Computation** - Optimized NumPy operations

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Tkinter (usually included with Python)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/wave-heat-simulator.git

# Navigate to the directory
cd wave-heat-simulator

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/gui.py
