"""
Calpyt1 - Comprehensive Python Mathematical Computing Framework

A next-level Python framework for symbolic, numerical, applied, and AI-integrated calculus.
Supports engineering, automation, robotics, physics, finance, and machine learning applications.
"""

__version__ = "0.1.0"
__author__ = "Calpyt1 Development Team"
__email__ = "dev@calpyt1.org"

# Core imports
from .core.base import CalcEngine

# Symbolic calculus imports
from .symbolic_calculus import (
    SymbolicDerivatives,
    SymbolicIntegrals,
    SymbolicLimits,
    TaylorSeries,
)

# Numerical calculus imports
from .numerical_calculus import (
    NumericalIntegration,
    ODESolver,
    Optimization,
    RootFinding,
)

# Visualization imports
from .visualization import CalcPlotter

# Main classes for easy access
__all__ = [
    "CalcEngine",
    "SymbolicDerivatives",
    "SymbolicIntegrals", 
    "SymbolicLimits",
    "TaylorSeries",
    "NumericalIntegration",
    "ODESolver",
    "Optimization",
    "RootFinding",
    "CalcPlotter",
]

# Version info
version_info = tuple(map(int, __version__.split('.')))
