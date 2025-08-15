"""
Base classes and core functionality for Calpyt1 framework
"""

import sympy as sp
import numpy as np
from typing import Union, Optional, Any, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalcEngine:
    """
    Main calculation engine that provides unified interface to all Calpyt1 modules
    """
    
    def __init__(self, precision: int = 15):
        """
        Initialize the calculation engine
        
        Args:
            precision: Numerical precision for calculations
        """
        self.precision = precision
        self.symbol_cache = {}
        self._setup_environment()
        
        # Lazy load modules to avoid circular imports
        self._symbolic_derivatives = None
        self._symbolic_integrals = None
        self._symbolic_limits = None
        self._taylor_series = None
        self._numerical_integration = None
        self._ode_solver = None
        self._optimization = None
        self._root_finding = None
        self._plotter = None
        
        logger.info(f"CalcEngine initialized with precision: {precision}")
    
    def _setup_environment(self):
        """Setup the calculation environment"""
        # Set SymPy printing options
        sp.init_printing(use_unicode=True)
        
        # Set NumPy precision
        np.set_printoptions(precision=self.precision)
    
    def create_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """
        Create a symbolic variable with caching
        
        Args:
            name: Symbol name
            **assumptions: SymPy assumptions (real, positive, etc.)
            
        Returns:
            SymPy symbol
        """
        cache_key = (name, tuple(sorted(assumptions.items())))
        
        if cache_key not in self.symbol_cache:
            self.symbol_cache[cache_key] = sp.Symbol(name, **assumptions)
        
        return self.symbol_cache[cache_key]
    
    def create_symbols(self, names: str, **assumptions) -> tuple:
        """
        Create multiple symbolic variables
        
        Args:
            names: Space-separated symbol names
            **assumptions: SymPy assumptions
            
        Returns:
            Tuple of SymPy symbols
        """
        return sp.symbols(names, **assumptions)
    
    @property
    def symbolic_derivatives(self):
        """Lazy load symbolic derivatives module"""
        if self._symbolic_derivatives is None:
            from ..symbolic_calculus.derivatives import SymbolicDerivatives
            self._symbolic_derivatives = SymbolicDerivatives(self)
        return self._symbolic_derivatives
    
    @property
    def symbolic_integrals(self):
        """Lazy load symbolic integrals module"""
        if self._symbolic_integrals is None:
            from ..symbolic_calculus.integrals import SymbolicIntegrals
            self._symbolic_integrals = SymbolicIntegrals(self)
        return self._symbolic_integrals
    
    @property
    def symbolic_limits(self):
        """Lazy load symbolic limits module"""
        if self._symbolic_limits is None:
            from ..symbolic_calculus.limits import SymbolicLimits
            self._symbolic_limits = SymbolicLimits(self)
        return self._symbolic_limits
    
    @property
    def taylor_series(self):
        """Lazy load Taylor series module"""
        if self._taylor_series is None:
            from ..symbolic_calculus.series import TaylorSeries
            self._taylor_series = TaylorSeries(self)
        return self._taylor_series
    
    @property
    def numerical_integration(self):
        """Lazy load numerical integration module"""
        if self._numerical_integration is None:
            from ..numerical_calculus.integration import NumericalIntegration
            self._numerical_integration = NumericalIntegration(self)
        return self._numerical_integration
    
    @property
    def ode_solver(self):
        """Lazy load ODE solver module"""
        if self._ode_solver is None:
            from ..numerical_calculus.ode_solver import ODESolver
            self._ode_solver = ODESolver(self)
        return self._ode_solver
    
    @property
    def optimization(self):
        """Lazy load optimization module"""
        if self._optimization is None:
            from ..numerical_calculus.optimization import Optimization
            self._optimization = Optimization(self)
        return self._optimization
    
    @property
    def root_finding(self):
        """Lazy load root finding module"""
        if self._root_finding is None:
            from ..numerical_calculus.root_finding import RootFinding
            self._root_finding = RootFinding(self)
        return self._root_finding
    
    @property
    def plotter(self):
        """Lazy load plotting module"""
        if self._plotter is None:
            from ..visualization.plotting import CalcPlotter
            self._plotter = CalcPlotter(self)
        return self._plotter
    
    def clear_cache(self):
        """Clear symbol cache"""
        self.symbol_cache.clear()
        logger.info("Symbol cache cleared")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the calculation engine
        
        Returns:
            Dictionary with engine information
        """
        return {
            "version": "0.1.0",
            "precision": self.precision,
            "cached_symbols": len(self.symbol_cache),
            "available_modules": [
                "symbolic_derivatives",
                "symbolic_integrals", 
                "symbolic_limits",
                "taylor_series",
                "numerical_integration",
                "ode_solver",
                "optimization", 
                "root_finding",
                "plotter"
            ]
        }


class BaseModule:
    """
    Base class for all Calpyt1 modules
    """
    
    def __init__(self, engine: CalcEngine):
        """
        Initialize module with reference to calculation engine
        
        Args:
            engine: Main CalcEngine instance
        """
        self.engine = engine
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_expression(self, expr: Union[str, sp.Basic]) -> sp.Basic:
        """
        Validate and convert expression to SymPy format
        
        Args:
            expr: Expression as string or SymPy object
            
        Returns:
            SymPy expression
            
        Raises:
            ValueError: If expression is invalid
        """
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except Exception as e:
                raise ValueError(f"Invalid expression: {expr}. Error: {str(e)}")
        elif isinstance(expr, sp.Basic):
            return expr
        else:
            raise ValueError(f"Expression must be string or SymPy object, got {type(expr)}")
    
    def validate_variable(self, var: Union[str, sp.Symbol]) -> sp.Symbol:
        """
        Validate and convert variable to SymPy Symbol
        
        Args:
            var: Variable as string or Symbol
            
        Returns:
            SymPy Symbol
        """
        if isinstance(var, str):
            return self.engine.create_symbol(var)
        elif isinstance(var, sp.Symbol):
            return var
        else:
            raise ValueError(f"Variable must be string or Symbol, got {type(var)}")
