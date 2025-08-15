"""
Numerical integration module for Calpyt1 framework
"""

import numpy as np
import scipy.integrate as integrate
from scipy import special
import sympy as sp
from typing import Union, Optional, List, Dict, Any, Callable, Tuple
from ..core.base import BaseModule


class NumericalIntegration(BaseModule):
    """
    Handles numerical integration methods
    """
    
    def __init__(self, engine):
        """Initialize numerical integration module"""
        super().__init__(engine)
        self.logger.info("NumericalIntegration module initialized")
    
    def _sympify_to_callable(self, expr: Union[str, sp.Basic, Callable], 
                            variables: List[Union[str, sp.Symbol]]) -> Callable:
        """Convert SymPy expression to callable function"""
        if callable(expr):
            return expr
        
        # Convert to SymPy if string
        if isinstance(expr, str):
            expr = sp.sympify(expr)
        
        # Convert variables to symbols
        sym_vars = [self.validate_variable(var) for var in variables]
        
        # Create lambdified function
        return sp.lambdify(sym_vars, expr, modules=['numpy'])
    
    def quad(self, func: Union[str, sp.Basic, Callable],
             a: float, b: float,
             args: tuple = (),
             epsabs: float = 1.49e-8,
             epsrel: float = 1.49e-8) -> Tuple[float, float]:
        """
        Numerical integration using adaptive quadrature
        
        Args:
            func: Function to integrate (expression, SymPy expr, or callable)
            a: Lower integration limit
            b: Upper integration limit
            args: Extra arguments to pass to function
            epsabs: Absolute error tolerance
            epsrel: Relative error tolerance
            
        Returns:
            Tuple of (integral_value, estimated_error)
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, [x])
            
            result, error = integrate.quad(func, a, b, args=args, 
                                         epsabs=epsabs, epsrel=epsrel)
            
            self.logger.info(f"Computed numerical integral from {a} to {b}")
            return result, error
        except Exception as e:
            raise RuntimeError(f"Failed to compute numerical integral: {str(e)}")
    
    def dblquad(self, func: Union[str, sp.Basic, Callable],
                a: float, b: float,
                gfun: Union[float, Callable], hfun: Union[float, Callable],
                args: tuple = ()) -> Tuple[float, float]:
        """
        Double integration
        
        Args:
            func: Function to integrate f(y,x)
            a: Lower x limit
            b: Upper x limit
            gfun: Lower y limit (function of x or constant)
            hfun: Upper y limit (function of x or constant)
            args: Extra arguments
            
        Returns:
            Tuple of (integral_value, estimated_error)
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x, y = self.engine.create_symbols('x y')
                func = self._sympify_to_callable(func, [y, x])
            
            result, error = integrate.dblquad(func, a, b, gfun, hfun, args=args)
            
            self.logger.info("Computed double integral")
            return result, error
        except Exception as e:
            raise RuntimeError(f"Failed to compute double integral: {str(e)}")
    
    def tplquad(self, func: Union[str, sp.Basic, Callable],
                a: float, b: float,
                gfun: Union[float, Callable], hfun: Union[float, Callable],
                qfun: Union[float, Callable], rfun: Union[float, Callable],
                args: tuple = ()) -> Tuple[float, float]:
        """
        Triple integration
        
        Args:
            func: Function to integrate f(z,y,x)
            a, b: x limits
            gfun, hfun: y limits (functions of x or constants)
            qfun, rfun: z limits (functions of x,y or constants)
            args: Extra arguments
            
        Returns:
            Tuple of (integral_value, estimated_error)
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x, y, z = self.engine.create_symbols('x y z')
                func = self._sympify_to_callable(func, [z, y, x])
            
            result, error = integrate.tplquad(func, a, b, gfun, hfun, 
                                            qfun, rfun, args=args)
            
            self.logger.info("Computed triple integral")
            return result, error
        except Exception as e:
            raise RuntimeError(f"Failed to compute triple integral: {str(e)}")
    
    def simpson(self, func: Union[str, sp.Basic, Callable],
                a: float, b: float, n: int = 100) -> float:
        """
        Simpson's rule integration
        
        Args:
            func: Function to integrate
            a: Lower limit
            b: Upper limit
            n: Number of intervals (must be even)
            
        Returns:
            Integral approximation
        """
        if n % 2 != 0:
            n += 1  # Make even
            
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, [x])
            
            # Generate points
            x_vals = np.linspace(a, b, n+1)
            y_vals = np.array([func(x) for x in x_vals])
            
            # Apply Simpson's rule
            h = (b - a) / n
            result = h/3 * (y_vals[0] + 4*np.sum(y_vals[1::2]) + 
                           2*np.sum(y_vals[2:-1:2]) + y_vals[-1])
            
            self.logger.info(f"Computed Simpson's rule integration with {n} intervals")
            return float(result)
        except Exception as e:
            raise RuntimeError(f"Failed to compute Simpson's rule: {str(e)}")
    
    def trapezoidal(self, func: Union[str, sp.Basic, Callable],
                   a: float, b: float, n: int = 100) -> float:
        """
        Trapezoidal rule integration
        
        Args:
            func: Function to integrate
            a: Lower limit
            b: Upper limit
            n: Number of intervals
            
        Returns:
            Integral approximation
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, [x])
            
            # Generate points
            x_vals = np.linspace(a, b, n+1)
            y_vals = np.array([func(x) for x in x_vals])
            
            # Apply trapezoidal rule
            h = (b - a) / n
            result = h * (0.5*y_vals[0] + np.sum(y_vals[1:-1]) + 0.5*y_vals[-1])
            
            self.logger.info(f"Computed trapezoidal rule integration with {n} intervals")
            return float(result)
        except Exception as e:
            raise RuntimeError(f"Failed to compute trapezoidal rule: {str(e)}")
    
    def monte_carlo(self, func: Union[str, sp.Basic, Callable],
                   bounds: List[Tuple[float, float]], 
                   n_samples: int = 10000,
                   seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Monte Carlo integration for multidimensional integrals
        
        Args:
            func: Function to integrate
            bounds: List of (min, max) tuples for each dimension
            n_samples: Number of random samples
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (integral_estimate, standard_error)
        """
        if seed is not None:
            np.random.seed(seed)
        
        try:
            n_dim = len(bounds)
            
            # Convert to callable if needed
            if not callable(func):
                if n_dim == 1:
                    x = self.engine.create_symbol('x')
                    variables = [x]
                elif n_dim == 2:
                    x, y = self.engine.create_symbols('x y')
                    variables = [x, y]
                elif n_dim == 3:
                    x, y, z = self.engine.create_symbols('x y z')
                    variables = [x, y, z]
                else:
                    variables = [self.engine.create_symbol(f'x{i}') for i in range(n_dim)]
                
                func = self._sympify_to_callable(func, variables)
            
            # Generate random samples
            samples = np.zeros((n_samples, n_dim))
            volume = 1.0
            
            for i, (low, high) in enumerate(bounds):
                samples[:, i] = np.random.uniform(low, high, n_samples)
                volume *= (high - low)
            
            # Evaluate function at samples
            if n_dim == 1:
                func_values = np.array([func(sample[0]) for sample in samples])
            else:
                func_values = np.array([func(*sample) for sample in samples])
            
            # Compute estimate and error
            mean_value = np.mean(func_values)
            integral_estimate = volume * mean_value
            
            # Standard error
            variance = np.var(func_values)
            standard_error = volume * np.sqrt(variance / n_samples)
            
            self.logger.info(f"Computed Monte Carlo integration with {n_samples} samples")
            return float(integral_estimate), float(standard_error)
        except Exception as e:
            raise RuntimeError(f"Failed to compute Monte Carlo integration: {str(e)}")
    
    def gaussian_quadrature(self, func: Union[str, sp.Basic, Callable],
                           a: float = -1, b: float = 1,
                           n: int = 5, method: str = 'legendre') -> float:
        """
        Gaussian quadrature integration
        
        Args:
            func: Function to integrate
            a: Lower limit
            b: Upper limit  
            n: Number of quadrature points
            method: Quadrature method ('legendre', 'chebyshev', 'laguerre', 'hermite')
            
        Returns:
            Integral approximation
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, [x])
            
            if method == 'legendre':
                # Gauss-Legendre quadrature for [-1,1], transform to [a,b]
                points, weights = np.polynomial.legendre.leggauss(n)
                # Transform from [-1,1] to [a,b]
                transformed_points = 0.5 * (b - a) * points + 0.5 * (b + a)
                jacobian = 0.5 * (b - a)
                
            elif method == 'chebyshev':
                # Gauss-Chebyshev quadrature
                points, weights = special.roots_chebyt(n)
                transformed_points = 0.5 * (b - a) * points + 0.5 * (b + a)
                jacobian = 0.5 * (b - a)
                
            elif method == 'laguerre':
                # Gauss-Laguerre quadrature for [0,∞) with weight e^(-x)
                points, weights = special.roots_laguerre(n)
                transformed_points = points
                jacobian = 1.0
                
            elif method == 'hermite':
                # Gauss-Hermite quadrature for (-∞,∞) with weight e^(-x²)
                points, weights = special.roots_hermite(n)
                transformed_points = points
                jacobian = 1.0
                
            else:
                raise ValueError(f"Unknown quadrature method: {method}")
            
            # Evaluate function at quadrature points and compute integral
            func_values = np.array([func(x) for x in transformed_points])
            result = jacobian * np.sum(weights * func_values)
            
            self.logger.info(f"Computed {method} quadrature with {n} points")
            return float(result)
        except Exception as e:
            raise RuntimeError(f"Failed to compute Gaussian quadrature: {str(e)}")
    
    def romberg(self, func: Union[str, sp.Basic, Callable],
               a: float, b: float, n: int = 6) -> np.ndarray:
        """
        Romberg integration (Richardson extrapolation)
        
        Args:
            func: Function to integrate
            a: Lower limit
            b: Upper limit
            n: Number of Romberg iterations
            
        Returns:
            Romberg table (numpy array)
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, [x])
            
            # Initialize Romberg table
            R = np.zeros((n, n))
            
            # First column: trapezoidal rule with different step sizes
            for i in range(n):
                N = 2**i + 1  # Number of points
                h = (b - a) / (N - 1)
                x_vals = np.linspace(a, b, N)
                y_vals = np.array([func(x) for x in x_vals])
                R[i, 0] = h * (0.5*y_vals[0] + np.sum(y_vals[1:-1]) + 0.5*y_vals[-1])
            
            # Fill the rest of the table using Richardson extrapolation
            for j in range(1, n):
                for i in range(j, n):
                    R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
            
            self.logger.info(f"Computed Romberg integration with {n} iterations")
            return R
        except Exception as e:
            raise RuntimeError(f"Failed to compute Romberg integration: {str(e)}")
    
    def adaptive_integration(self, func: Union[str, sp.Basic, Callable],
                           a: float, b: float,
                           tol: float = 1e-6,
                           max_iter: int = 1000) -> Tuple[float, int]:
        """
        Adaptive integration using recursive subdivision
        
        Args:
            func: Function to integrate
            a: Lower limit
            b: Upper limit
            tol: Error tolerance
            max_iter: Maximum number of subdivisions
            
        Returns:
            Tuple of (integral_value, number_of_evaluations)
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, [x])
            
            def adaptive_quad_recursive(f, a, b, tol, eval_count=0):
                """Recursive adaptive quadrature"""
                if eval_count > max_iter:
                    return self.simpson(f, a, b, 10), eval_count
                
                # Simpson's rule for whole interval
                mid = (a + b) / 2
                S_whole = self.simpson(f, a, b, 2)
                
                # Simpson's rule for two halves
                S_left = self.simpson(f, a, mid, 2)
                S_right = self.simpson(f, mid, b, 2)
                S_halves = S_left + S_right
                
                eval_count += 10  # Approximate number of evaluations
                
                # Check error estimate
                error_est = abs(S_halves - S_whole) / 15  # Error estimate for Simpson's
                
                if error_est <= tol:
                    return S_halves, eval_count
                else:
                    # Subdivide further
                    left_result, eval_count = adaptive_quad_recursive(f, a, mid, tol/2, eval_count)
                    right_result, eval_count = adaptive_quad_recursive(f, mid, b, tol/2, eval_count)
                    return left_result + right_result, eval_count
            
            result, evaluations = adaptive_quad_recursive(func, a, b, tol)
            
            self.logger.info(f"Computed adaptive integration with {evaluations} evaluations")
            return float(result), evaluations
        except Exception as e:
            raise RuntimeError(f"Failed to compute adaptive integration: {str(e)}")
    
    def compare_methods(self, func: Union[str, sp.Basic, Callable],
                       a: float, b: float,
                       exact_value: Optional[float] = None) -> Dict[str, Dict]:
        """
        Compare different integration methods
        
        Args:
            func: Function to integrate
            a: Lower limit
            b: Upper limit
            exact_value: Known exact value for error calculation
            
        Returns:
            Dictionary with results from different methods
        """
        results = {}
        
        try:
            # Adaptive quadrature (scipy.integrate.quad)
            try:
                quad_result, quad_error = self.quad(func, a, b)
                results['adaptive_quad'] = {
                    'value': quad_result,
                    'error_estimate': quad_error,
                    'method': 'Adaptive quadrature'
                }
            except Exception as e:
                results['adaptive_quad'] = {'error': str(e)}
            
            # Simpson's rule
            try:
                simpson_result = self.simpson(func, a, b, 100)
                results['simpson'] = {
                    'value': simpson_result,
                    'method': 'Simpson\'s rule (n=100)'
                }
            except Exception as e:
                results['simpson'] = {'error': str(e)}
            
            # Trapezoidal rule
            try:
                trap_result = self.trapezoidal(func, a, b, 100)
                results['trapezoidal'] = {
                    'value': trap_result,
                    'method': 'Trapezoidal rule (n=100)'
                }
            except Exception as e:
                results['trapezoidal'] = {'error': str(e)}
            
            # Gaussian quadrature
            if a >= -1 and b <= 1:  # Only for suitable range
                try:
                    gauss_result = self.gaussian_quadrature(func, a, b, 10)
                    results['gaussian'] = {
                        'value': gauss_result,
                        'method': 'Gauss-Legendre (n=10)'
                    }
                except Exception as e:
                    results['gaussian'] = {'error': str(e)}
            
            # Calculate absolute errors if exact value is provided
            if exact_value is not None:
                for method, data in results.items():
                    if 'value' in data:
                        data['absolute_error'] = abs(data['value'] - exact_value)
                        data['relative_error'] = abs(data['value'] - exact_value) / abs(exact_value)
            
            self.logger.info("Compared integration methods")
            return results
        except Exception as e:
            return {'error': f"Failed to compare methods: {str(e)}"}
