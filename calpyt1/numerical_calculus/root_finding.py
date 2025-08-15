"""
Root finding module for Calpyt1 framework
"""

import numpy as np
import scipy.optimize as optimize
from scipy.optimize import fsolve, root, brentq, newton
import sympy as sp
from typing import Union, Optional, List, Dict, Any, Callable, Tuple
from ..core.base import BaseModule


class RootFinding(BaseModule):
    """
    Handles numerical root finding methods
    """
    
    def __init__(self, engine):
        """Initialize root finding module"""
        super().__init__(engine)
        self.logger.info("RootFinding module initialized")
    
    def _sympify_to_callable(self, expr: Union[str, sp.Basic], 
                            variables: List[str]) -> Callable:
        """Convert SymPy expression to callable function"""
        if callable(expr):
            return expr
            
        if isinstance(expr, str):
            expr = sp.sympify(expr)
        
        # Create symbols
        sym_vars = [sp.Symbol(var) for var in variables]
        
        # Create lambdified function
        return sp.lambdify(sym_vars, expr, modules=['numpy'])
    
    def bisection(self, func: Union[str, sp.Basic, Callable],
                 a: float, b: float,
                 tolerance: float = 1e-6,
                 max_iterations: int = 100) -> Dict[str, Any]:
        """
        Bisection method for root finding
        
        Args:
            func: Function to find root of
            a: Left bracket
            b: Right bracket
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            
        Returns:
            Root finding result dictionary
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, ['x'])
            
            # Check that function changes sign
            fa, fb = func(a), func(b)
            if fa * fb > 0:
                raise ValueError("Function must have opposite signs at endpoints")
            
            # Bisection iterations
            history = []
            
            for iteration in range(max_iterations):
                c = (a + b) / 2
                fc = func(c)
                
                history.append({
                    'iteration': iteration,
                    'a': a, 'b': b, 'c': c,
                    'fa': fa, 'fb': fb, 'fc': fc,
                    'interval_width': abs(b - a)
                })
                
                # Check convergence
                if abs(fc) < tolerance or abs(b - a) < tolerance:
                    break
                
                # Update interval
                if fa * fc < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
            
            result = {
                'success': abs(func(c)) < tolerance,
                'root': c,
                'function_value': func(c),
                'iterations': iteration + 1,
                'tolerance': tolerance,
                'history': history,
                'method': 'bisection'
            }
            
            self.logger.info(f"Bisection method completed in {iteration + 1} iterations")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to find root using bisection method: {str(e)}")
    
    def newton_raphson(self, func: Union[str, sp.Basic, Callable],
                      derivative: Union[str, sp.Basic, Callable, None],
                      x0: float,
                      tolerance: float = 1e-6,
                      max_iterations: int = 100) -> Dict[str, Any]:
        """
        Newton-Raphson method for root finding
        
        Args:
            func: Function to find root of
            derivative: Derivative function (if None, computed numerically)
            x0: Initial guess
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            
        Returns:
            Root finding result dictionary
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, ['x'])
            
            # Handle derivative
            if derivative is None:
                def derivative_func(x):
                    h = 1e-8
                    return (func(x + h) - func(x - h)) / (2 * h)
            elif not callable(derivative):
                derivative_func = self._sympify_to_callable(derivative, ['x'])
            else:
                derivative_func = derivative
            
            # Newton-Raphson iterations
            x = x0
            history = []
            
            for iteration in range(max_iterations):
                fx = func(x)
                fpx = derivative_func(x)
                
                history.append({
                    'iteration': iteration,
                    'x': x,
                    'f(x)': fx,
                    'f\'(x)': fpx
                })
                
                # Check for zero derivative
                if abs(fpx) < 1e-12:
                    raise RuntimeError("Derivative is zero - cannot continue")
                
                # Check convergence
                if abs(fx) < tolerance:
                    break
                
                # Newton-Raphson update
                x_new = x - fx / fpx
                
                # Check for convergence in x
                if abs(x_new - x) < tolerance:
                    break
                
                x = x_new
            
            result = {
                'success': abs(func(x)) < tolerance,
                'root': x,
                'function_value': func(x),
                'iterations': iteration + 1,
                'tolerance': tolerance,
                'history': history,
                'method': 'newton_raphson'
            }
            
            self.logger.info(f"Newton-Raphson method completed in {iteration + 1} iterations")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to find root using Newton-Raphson method: {str(e)}")
    
    def secant(self, func: Union[str, sp.Basic, Callable],
              x0: float, x1: float,
              tolerance: float = 1e-6,
              max_iterations: int = 100) -> Dict[str, Any]:
        """
        Secant method for root finding
        
        Args:
            func: Function to find root of
            x0: First initial guess
            x1: Second initial guess
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            
        Returns:
            Root finding result dictionary
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, ['x'])
            
            # Secant method iterations
            history = []
            
            for iteration in range(max_iterations):
                fx0 = func(x0)
                fx1 = func(x1)
                
                history.append({
                    'iteration': iteration,
                    'x0': x0, 'x1': x1,
                    'f(x0)': fx0, 'f(x1)': fx1
                })
                
                # Check convergence
                if abs(fx1) < tolerance:
                    break
                
                # Check for zero denominator
                if abs(fx1 - fx0) < 1e-12:
                    raise RuntimeError("Function values are too close - cannot continue")
                
                # Secant update
                x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
                
                # Check convergence in x
                if abs(x_new - x1) < tolerance:
                    break
                
                x0, x1 = x1, x_new
            
            result = {
                'success': abs(func(x1)) < tolerance,
                'root': x1,
                'function_value': func(x1),
                'iterations': iteration + 1,
                'tolerance': tolerance,
                'history': history,
                'method': 'secant'
            }
            
            self.logger.info(f"Secant method completed in {iteration + 1} iterations")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to find root using secant method: {str(e)}")
    
    def brent(self, func: Union[str, sp.Basic, Callable],
             a: float, b: float,
             tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Brent's method for root finding (combines bisection, secant, and inverse quadratic)
        
        Args:
            func: Function to find root of
            a: Left bracket
            b: Right bracket
            tolerance: Convergence tolerance
            
        Returns:
            Root finding result dictionary
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, ['x'])
            
            # Use scipy's Brent method
            root = brentq(func, a, b, xtol=tolerance)
            
            result = {
                'success': True,
                'root': root,
                'function_value': func(root),
                'tolerance': tolerance,
                'method': 'brent',
                'bracket': [a, b]
            }
            
            self.logger.info("Brent's method completed successfully")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to find root using Brent's method: {str(e)}")
    
    def fixed_point(self, func: Union[str, sp.Basic, Callable],
                   x0: float,
                   tolerance: float = 1e-6,
                   max_iterations: int = 100) -> Dict[str, Any]:
        """
        Fixed point iteration: find x such that x = g(x)
        
        Args:
            func: Function g(x) for fixed point iteration
            x0: Initial guess
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            
        Returns:
            Fixed point result dictionary
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, ['x'])
            
            # Fixed point iterations
            x = x0
            history = []
            
            for iteration in range(max_iterations):
                x_new = func(x)
                
                history.append({
                    'iteration': iteration,
                    'x': x,
                    'g(x)': x_new,
                    'error': abs(x_new - x)
                })
                
                # Check convergence
                if abs(x_new - x) < tolerance:
                    break
                
                x = x_new
            
            result = {
                'success': abs(func(x) - x) < tolerance,
                'fixed_point': x,
                'error': abs(func(x) - x),
                'iterations': iteration + 1,
                'tolerance': tolerance,
                'history': history,
                'method': 'fixed_point'
            }
            
            self.logger.info(f"Fixed point iteration completed in {iteration + 1} iterations")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to find fixed point: {str(e)}")
    
    def solve_system(self, equations: List[Union[str, sp.Basic, Callable]],
                    x0: Union[List[float], np.ndarray],
                    method: str = 'hybr',
                    tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Solve system of nonlinear equations
        
        Args:
            equations: List of equations to solve (should equal zero)
            x0: Initial guess
            method: Solution method ('hybr', 'lm', 'broyden1', 'broyden2', etc.)
            tolerance: Convergence tolerance
            
        Returns:
            System solution result dictionary
        """
        try:
            x0 = np.array(x0)
            n_vars = len(x0)
            n_eqs = len(equations)
            
            if n_vars != n_eqs:
                raise ValueError("Number of variables must equal number of equations")
            
            variables = [f'x{i}' for i in range(n_vars)]
            
            # Convert equations to callable functions
            equation_funcs = []
            for eq in equations:
                if not callable(eq):
                    eq_func = self._sympify_to_callable(eq, variables)
                else:
                    eq_func = eq
                equation_funcs.append(eq_func)
            
            def system_func(x):
                """System function that returns vector of equation values"""
                return np.array([eq(*x) for eq in equation_funcs])
            
            # Solve the system
            solution = root(system_func, x0, method=method, tol=tolerance)
            
            result = {
                'success': solution.success,
                'x': solution.x.tolist(),
                'residual': np.linalg.norm(solution.fun),
                'function_values': solution.fun.tolist(),
                'iterations': getattr(solution, 'nit', None),
                'function_evaluations': solution.nfev,
                'method': method,
                'message': solution.message
            }
            
            self.logger.info(f"System of equations solved using {method} method")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to solve system of equations: {str(e)}")
    
    def polynomial_roots(self, coefficients: List[Union[float, int, complex]]) -> Dict[str, Any]:
        """
        Find all roots of a polynomial
        
        Args:
            coefficients: Polynomial coefficients [an, an-1, ..., a1, a0]
                         for polynomial an*x^n + ... + a1*x + a0
            
        Returns:
            Polynomial roots result dictionary
        """
        try:
            # Use numpy to find roots
            roots = np.roots(coefficients)
            
            # Separate real and complex roots
            real_roots = []
            complex_roots = []
            
            for root in roots:
                if np.isreal(root) and abs(np.imag(root)) < 1e-12:
                    real_roots.append(float(np.real(root)))
                else:
                    complex_roots.append(complex(root))
            
            result = {
                'success': True,
                'all_roots': roots.tolist(),
                'real_roots': real_roots,
                'complex_roots': complex_roots,
                'n_real_roots': len(real_roots),
                'n_complex_roots': len(complex_roots),
                'polynomial_degree': len(coefficients) - 1,
                'coefficients': coefficients
            }
            
            self.logger.info(f"Found {len(roots)} roots of polynomial (degree {len(coefficients)-1})")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to find polynomial roots: {str(e)}")
    
    def find_all_roots(self, func: Union[str, sp.Basic, Callable],
                      search_range: Tuple[float, float],
                      n_intervals: int = 100,
                      tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Find all roots in a given range by subdividing and checking sign changes
        
        Args:
            func: Function to find roots of
            search_range: Range to search for roots (a, b)
            n_intervals: Number of intervals to subdivide range
            tolerance: Tolerance for root finding
            
        Returns:
            All roots result dictionary
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, ['x'])
            
            a, b = search_range
            x_vals = np.linspace(a, b, n_intervals + 1)
            
            roots = []
            intervals_with_roots = []
            
            # Check each interval for sign changes
            for i in range(n_intervals):
                x1, x2 = x_vals[i], x_vals[i + 1]
                
                try:
                    f1, f2 = func(x1), func(x2)
                    
                    # Check for sign change
                    if f1 * f2 < 0:
                        # Use Brent's method to find root in this interval
                        try:
                            root = brentq(func, x1, x2, xtol=tolerance)
                            roots.append(root)
                            intervals_with_roots.append((x1, x2))
                        except:
                            # If Brent fails, try bisection
                            try:
                                bisection_result = self.bisection(func, x1, x2, tolerance)
                                if bisection_result['success']:
                                    roots.append(bisection_result['root'])
                                    intervals_with_roots.append((x1, x2))
                            except:
                                pass
                    
                    # Check for exact zero
                    elif abs(f1) < tolerance:
                        if not any(abs(root - x1) < tolerance for root in roots):
                            roots.append(x1)
                    elif abs(f2) < tolerance and i == n_intervals - 1:
                        if not any(abs(root - x2) < tolerance for root in roots):
                            roots.append(x2)
                
                except:
                    # Skip intervals where function evaluation fails
                    continue
            
            # Remove duplicate roots
            unique_roots = []
            for root in roots:
                if not any(abs(root - existing) < tolerance for existing in unique_roots):
                    unique_roots.append(root)
            
            result = {
                'success': True,
                'roots': unique_roots,
                'n_roots': len(unique_roots),
                'search_range': search_range,
                'intervals_checked': n_intervals,
                'intervals_with_roots': intervals_with_roots,
                'tolerance': tolerance
            }
            
            self.logger.info(f"Found {len(unique_roots)} roots in range {search_range}")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to find all roots: {str(e)}")
    
    def compare_methods(self, func: Union[str, sp.Basic, Callable],
                       initial_data: Dict[str, Any],
                       tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Compare different root finding methods
        
        Args:
            func: Function to find root of
            initial_data: Dictionary with method-specific initial data
                         {'bracket': [a, b], 'guess': x0, 'second_guess': x1}
            tolerance: Convergence tolerance
            
        Returns:
            Comparison of different methods
        """
        results = {}
        
        try:
            # Bisection method (requires bracket)
            if 'bracket' in initial_data:
                try:
                    a, b = initial_data['bracket']
                    bisection_result = self.bisection(func, a, b, tolerance)
                    results['bisection'] = {
                        'result': bisection_result,
                        'converged': bisection_result['success'],
                        'iterations': bisection_result['iterations'],
                        'final_error': abs(bisection_result['function_value'])
                    }
                except Exception as e:
                    results['bisection'] = {'error': str(e)}
            
            # Newton-Raphson method (requires guess)
            if 'guess' in initial_data:
                try:
                    x0 = initial_data['guess']
                    newton_result = self.newton_raphson(func, None, x0, tolerance)
                    results['newton_raphson'] = {
                        'result': newton_result,
                        'converged': newton_result['success'],
                        'iterations': newton_result['iterations'],
                        'final_error': abs(newton_result['function_value'])
                    }
                except Exception as e:
                    results['newton_raphson'] = {'error': str(e)}
            
            # Secant method (requires two guesses)
            if 'guess' in initial_data and 'second_guess' in initial_data:
                try:
                    x0 = initial_data['guess']
                    x1 = initial_data['second_guess']
                    secant_result = self.secant(func, x0, x1, tolerance)
                    results['secant'] = {
                        'result': secant_result,
                        'converged': secant_result['success'],
                        'iterations': secant_result['iterations'],
                        'final_error': abs(secant_result['function_value'])
                    }
                except Exception as e:
                    results['secant'] = {'error': str(e)}
            
            # Brent's method (requires bracket)
            if 'bracket' in initial_data:
                try:
                    a, b = initial_data['bracket']
                    brent_result = self.brent(func, a, b, tolerance)
                    results['brent'] = {
                        'result': brent_result,
                        'converged': brent_result['success'],
                        'iterations': 'N/A (scipy implementation)',
                        'final_error': abs(brent_result['function_value'])
                    }
                except Exception as e:
                    results['brent'] = {'error': str(e)}
            
            self.logger.info(f"Compared {len(results)} root finding methods")
            return results
        except Exception as e:
            return {'error': f"Failed to compare methods: {str(e)}"}
