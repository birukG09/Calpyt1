# Calpyt1 - Comprehensive Python Mathematical Computing Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

**Calpyt1** is a **next-level Python framework** for **symbolic, numerical, applied, and AI-integrated calculus**, designed for **engineering, automation, robotics, physics, finance, and machine learning applications**. It combines symbolic math, numerical solvers, optimization, simulation, and interactive visualization in a single modular package.
![img alt](https://github.com/birukG09/Calpyt1/blob/24b6ebf0f889cc27b45f8b09a9f2d100187a878b/20250909_0002_Calpyt1%20Python%20Framework_simple_compose_01k4nj0gzkejktm9sw997w8r1n.png)
---

## 🌟 Features

### 1. Symbolic Calculus

* ⚡ **Derivatives:** Basic, higher-order, partial derivatives, chain rule, implicit differentiation
* ∫ **Integration:** Indefinite, definite, multiple, line & surface integrals
* 🔍 **Limits:** Evaluate limits, L’Hôpital’s rule, continuity analysis, asymptote detection
* Σ **Series:** Taylor, Maclaurin, Fourier series, convergence analysis
* 🧮 **Advanced:** Gradient, Jacobian, Hessian matrices, critical point analysis

### 2. Numerical Methods

* 📈 **Integration:** Adaptive quadrature, Simpson’s rule, trapezoidal rule, Monte Carlo, Gaussian quadrature
* 🔄 **ODE Solvers:** Euler, Runge-Kutta, adaptive methods, stiff solvers, systems of ODEs
* 🌊 **PDE Support:** Basic solver interface for partial differential equations
* 🧩 **Optimization:** Gradient descent, genetic algorithms, simulated annealing
* ⚙️ **Root Finding:** Bisection, Newton-Raphson, secant, Brent’s method, polynomial roots

### 3. Visualization

* 📊 **2D & 3D Plotting:** Function plots, surface plots, contour plots
* 🏞 **Vector Fields:** Gradient fields, phase portraits
* 🖥 **Interactive Dashboards:** Plotly-based interactive visualizations
* 🔄 **Convergence Analysis:** Visualize optimization & numerical method convergence

### 4. Engineering Applications

* 🛠 **Control Systems:** Transfer functions, step/impulse/Bode response, stability analysis
* 🤖 **Mechanical:** Kinematics, dynamics simulation
* ⚡ **Electrical:** RLC circuit simulation, signal processing
* 🌡 **Thermodynamics:** Heat transfer and optimization problems

### 5. AI / ML Integration

* 🤖 **Auto-Differentiation:** Gradient-based pipelines
* 🧠 **Physics-Informed Neural Networks (PINNs):** Hybrid symbolic-numeric learning
* ⚡ **ML Framework Integration:** Compatible with PyTorch, TensorFlow, and JAX

---

## 🎬 Visual Demonstrations 

* **2D/3D Plots**(

* **Control System Responses**
  !

* **Optimization Convergence**
  
---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/birukG09/Calpyt1.git
cd Calpyt1

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

### 📦 Usage Examples

#### Symbolic Derivative

```python
from calpyt1.mega_calculus import MegaAdvancedCalculus
x = MegaAdvancedCalculus().create_symbol('x')
f = x**3 + 2*x**2 + x
print("Derivative:", MegaAdvancedCalculus().derivative(f, x))
```

#### Numerical Integration

```python
calc = MegaAdvancedCalculus()
result = calc.numeric_integral(lambda x: x**2, 0, 5)
print("Numerical Integral:", result)
```

#### Control System Simulation

```python
import numpy as np
calc = MegaAdvancedCalculus()
t = np.linspace(0, 10, 100)
num = [1]
den = [1, 2, 1]
t_out, y_out = calc.control_system_response(num, den, t)
```

#### 3D Surface Plot

```python
x, y = calc.create_symbol('x'), calc.create_symbol('y')
func = x**2 + y**2
calc.plot_surface(func, x, y, -5, 5, -5, 5)
```

---

## 📂 Folder Structure

```
Calpyt1/
├─ calpyt1/                 # Core library
│  ├─ symbolic.py
│  ├─ numeric.py
│  ├─ engineering.py
│  ├─ robotics.py
│  ├─ visualization.py
│  └─ ai_ml.py
├─ examples/                 # Jupyter notebooks & scripts
├─ tests/                    # Unit tests
├─ requirements.txt
└─ README.md
```

---

## ⚡ Contribution

Contributions are welcome!

* Add new modules (advanced finance, CFD, robotics AI)
* Improve visualization or AI/ML features
* Optimize numerical solvers
* Add Jupyter notebooks with interactive examples

**How to contribute:**

```bash
git fork https://github.com/birukG09/Calpyt1.git
git checkout -b feature-name
# Make changes
git commit -m "Add feature XYZ"
git push origin feature-name
# Create Pull Request
```

---

## 📚 References & Resources

* [SymPy Documentation](https://www.sympy.org/en/index.html)
* [NumPy & SciPy Documentation](https://numpy.org/doc/stable/)
* [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
* [Plotly Python Docs](https://plotly.com/python/)
* [Physics-Informed Neural Networks (PINNs)](https://deepxde.readthedocs.io/en/latest/)

---

## 🛡 License

[MIT License](LICENSE)

---

### ✅ Why Calpyt1 Stands Out

* **Integrated Framework:** Symbolic + numeric + applied + AI-ready calculus
* **Multi-Domain:** Engineering, robotics, physics, finance, AI/ML
* **Modular & Extensible:** Easily add new modules and capabilities
* **Interactive Examples:** Rich notebooks and GIF visualizations ,,,,

