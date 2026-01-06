# 🌊🔥 Wave & Heat Equation Simulator

A Python-based graphical simulator for the **one-dimensional Wave Equation** and **Heat Equation**,  
implemented using **analytical Fourier series solutions** and visualized through an interactive **Tkinter GUI**.

This project is designed for educational, scientific, and numerical analysis purposes, providing a clear
understanding of wave propagation and heat diffusion phenomena.

---

## ✨ Features

### 🌊 Wave Equation
- One-dimensional wave equation simulation
- Initial displacement and initial velocity support
- Analytical Fourier series solution
- Normal modes visualization
- Nodal points identification
- Kinetic, potential, and total energy analysis
- Energy conservation verification
- D’Alembert solution visualization
- Fourier coefficients spectrum (Aₙ, Bₙ)
- Time-dependent wave animation

### 🔥 Heat Equation
- One-dimensional heat equation simulation
- Dirichlet and Neumann boundary conditions
- Analytical Fourier series solution
- Temperature distribution over time
- Steady-state solution visualization
- Center-point temperature decay analysis
- Thermal time constant (τ) computation
- Material comparison (Copper, Aluminum, Steel, Glass, Wood)
- Heat diffusion animation

---

## 🧮 Mathematical Background

### 🌊 Wave Equation

∂²u/∂t² = c² ∂²u/∂x²

u(x,t) = Σ [ Aₙ cos(ωₙ t) + Bₙ sin(ωₙ t) ] sin(nπx / L)

where:
- Aₙ: coefficients from initial displacement
- Bₙ: coefficients from initial velocity
- ωₙ = nπc / L

---

### 🔥 Heat Equation

∂u/∂t = α ∂²u/∂x²

τₙ = L² / (n² π² α)

---

## 🗂 Project Structure

wave_heat_simulator/  
├── README.md  
├── LICENSE  
├── requirements.txt  
└── src/  
&nbsp;&nbsp;&nbsp;&nbsp;└── wave_heat_simulator/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── __init__.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── main.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── gui.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── wave_simulation.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── heat_simulation.py  

---

## ⚙ Requirements

- Python 3.9 or newer
- numpy
- matplotlib
- tkinter (included with standard Python installations)

pip install -r requirements.txt

---

## ▶ Running the Application

python -m src.wave_heat_simulator.main

---

## ⌨ Controls & Shortcuts

- Space → Play / Pause animation
- Escape → Stop animation
- Ctrl + S → Save current figure
- Ctrl + E → Export simulation data

---

## 📤 Export Options

- CSV export of simulation data
- JSON export of simulation parameters
- Figure export (PNG, PDF, SVG)

---

## 📊 Analysis Capabilities

- Energy conservation validation
- Thermal time constant analysis
- Steady-state convergence visualization
- Center temperature decay tracking
- Material-based diffusion comparison

---

## 🎓 Educational Use

Suitable for:
- Engineering Mathematics
- Computational Physics
- Partial Differential Equations
- Fourier Series visualization
- Scientific computing education

---

## 📜 License

MIT License

---

## 👤 Author

**Mohamad Mahdy Sobhany poor**

Python · Scientific Computing · GUI Development

⭐ If you find this project useful, consider starring it on GitHub.
