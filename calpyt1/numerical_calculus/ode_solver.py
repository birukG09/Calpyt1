"""
ODE solver module for Calpyt1 framework
"""

import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
import sympy as sp
from typing import Union, Optional, List, Dict, Any, Callable, Tuple
from ..core.base import BaseModule


class ODESolver(BaseModule):
    """
    Handles ordinary differential equation solving
    """
    
    def __init__(self, engine):
        """Initialize ODE solver module"""
        super().__init__(engine)
        self.logger.info("ODESolver module initialized")
    
    def _sympify_to_callable(self, expr: Union[str, sp.Basic], 
                            variables: List[str]) -> Callable:
        """Convert SymPy expression to callable function"""
        if isinstance(expr, str):
            expr = sp.sympify(expr)
        
        # Create symbols
        sym_vars = [sp.Symbol(var) for var in variables]
        
        # Create lambdified function
        return sp.lambdify(sym_vars, expr, modules=['numpy'])
    
    def solve_ivp_wrapper(self, func: Union[str, sp.Basic, Callable],
                         t_span: Tuple[float, float],
                         y0: Union[float, List[float], np.ndarray],
                         method: str = 'RK45',
                         t_eval: Optional[np.ndarray] = None,
                         dense_output: bool = False,
                         **kwargs) -> Dict[str, Any]:
        """
        Solve initial value problem using scipy.integrate.solve_ivp
        
        Args:
            func: Right-hand side function dy/dt = f(t, y)
            t_span: Integration interval (t0, tf)
            y0: Initial condition(s)
            method: Integration method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA')
            t_eval: Points where solution is evaluated
            dense_output: Whether to compute continuous solution
            **kwargs: Additional arguments for solve_ivp
            
        Returns:
            Dictionary with solution results
        """
        try:
            # Convert to numpy array if needed
            if isinstance(y0, (int, float)):
                y0 = np.array([y0])
            elif isinstance(y0, list):
                y0 = np.array(y0)
            
            # Convert function to callable if needed
            if not callable(func):
                # Assume func is dy/dt = f(t, y) where y can be vector
                t, y = sp.symbols('t y')
                if isinstance(func, str):
                    func = sp.sympify(func)
                
                # For systems, we need to handle multiple variables
                if len(y0) == 1:
                    func_callable = self._sympify_to_callable(func, ['t', 'y'])
                    def ode_func(t, y):
                        return np.array([func_callable(t, y[0])])
                else:
                    # For systems, assume func is a list of expressions
                    if not isinstance(func, list):
                        raise ValueError("For systems of ODEs, provide list of expressions")
                    
                    func_callables = []
                    for i, f in enumerate(func):
                        vars_list = ['t'] + [f'y{j}' for j in range(len(y0))]
                        func_callables.append(self._sympify_to_callable(f, vars_list))
                    
                    def ode_func(t, y):
                        result = []
                        for f_callable in func_callables:
                            args = [t] + list(y)
                            result.append(f_callable(*args))
                        return np.array(result)
            else:
                ode_func = func
            
            # Solve the ODE
            solution = solve_ivp(ode_func, t_span, y0, method=method,
                               t_eval=t_eval, dense_output=dense_output, **kwargs)
            
            result = {
                'success': solution.success,
                'message': solution.message,
                't': solution.t,
                'y': solution.y,
                'nfev': solution.nfev,
                'njev': getattr(solution, 'njev', None),
                'nlu': getattr(solution, 'nlu', None),
                'sol': solution.sol if dense_output else None
            }
            
            self.logger.info(f"Solved ODE using {method} method")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to solve ODE: {str(e)}")
    
    def euler_method(self, func: Union[str, sp.Basic, Callable],
                    t0: float, y0: float, tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using Euler's method
        
        Args:
            func: Right-hand side function dy/dt = f(t, y)
            t0: Initial time
            y0: Initial value
            tf: Final time
            h: Step size
            
        Returns:
            Tuple of (t_values, y_values)
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                func = self._sympify_to_callable(func, ['t', 'y'])
            
            # Create arrays
            t_values = np.arange(t0, tf + h, h)
            y_values = np.zeros_like(t_values)
            y_values[0] = y0
            
            # Euler's method: y_{n+1} = y_n + h * f(t_n, y_n)
            for i in range(len(t_values) - 1):
                y_values[i+1] = y_values[i] + h * func(t_values[i], y_values[i])
            
            self.logger.info(f"Solved ODE using Euler's method with step size {h}")
            return t_values, y_values
        except Exception as e:
            raise RuntimeError(f"Failed to solve ODE with Euler's method: {str(e)}")
    
    def runge_kutta_4(self, func: Union[str, sp.Basic, Callable],
                     t0: float, y0: float, tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using 4th order Runge-Kutta method
        
        Args:
            func: Right-hand side function dy/dt = f(t, y)
            t0: Initial time
            y0: Initial value
            tf: Final time
            h: Step size
            
        Returns:
            Tuple of (t_values, y_values)
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                func = self._sympify_to_callable(func, ['t', 'y'])
            
            # Create arrays
            t_values = np.arange(t0, tf + h, h)
            y_values = np.zeros_like(t_values)
            y_values[0] = y0
            
            # RK4 method
            for i in range(len(t_values) - 1):
                t_i = t_values[i]
                y_i = y_values[i]
                
                k1 = h * func(t_i, y_i)
                k2 = h * func(t_i + h/2, y_i + k1/2)
                k3 = h * func(t_i + h/2, y_i + k2/2)
                k4 = h * func(t_i + h, y_i + k3)
                
                y_values[i+1] = y_i + (k1 + 2*k2 + 2*k3 + k4) / 6
            
            self.logger.info(f"Solved ODE using RK4 method with step size {h}")
            return t_values, y_values
        except Exception as e:
            raise RuntimeError(f"Failed to solve ODE with RK4 method: {str(e)}")
    
    def solve_system(self, system: List[Union[str, sp.Basic]],
                    t_span: Tuple[float, float],
                    y0: List[float],
                    method: str = 'RK45',
                    t_eval: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Solve system of ODEs
        
        Args:
            system: List of differential equations [dy1/dt, dy2/dt, ...]
            t_span: Time interval (t0, tf)
            y0: Initial conditions [y1(0), y2(0), ...]
            method: Integration method
            t_eval: Points where solution is evaluated
            
        Returns:
            Solution dictionary
        """
        try:
            n_eqs = len(system)
            if len(y0) != n_eqs:
                raise ValueError("Number of initial conditions must match number of equations")
            
            # Create variable names
            var_names = ['t'] + [f'y{i}' for i in range(n_eqs)]
            
            # Convert each equation to callable
            func_list = []
            for eq in system:
                if isinstance(eq, str):
                    eq = sp.sympify(eq)
                func_list.append(self._sympify_to_callable(eq, var_names))
            
            def system_func(t, y):
                """System function for solve_ivp"""
                args = [t] + list(y)
                return np.array([f(*args) for f in func_list])
            
            # Solve the system
            solution = solve_ivp(system_func, t_span, y0, method=method, t_eval=t_eval)
            
            result = {
                'success': solution.success,
                'message': solution.message,
                't': solution.t,
                'y': solution.y,  # Each row is a solution component
                'nfev': solution.nfev
            }
            
            self.logger.info(f"Solved system of {n_eqs} ODEs using {method}")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to solve ODE system: {str(e)}")
    
    def solve_second_order(self, equation: Union[str, sp.Basic],
                          t_span: Tuple[float, float],
                          y0: float, dy0: float,
                          method: str = 'RK45') -> Dict[str, Any]:
        """
        Solve second-order ODE by converting to first-order system
        
        Args:
            equation: Second-order ODE in form d2y/dt2 = f(t, y, dy/dt)
            t_span: Time interval
            y0: Initial position
            dy0: Initial velocity
            method: Integration method
            
        Returns:
            Solution dictionary
        """
        try:
            # Convert second-order ODE to system of first-order ODEs
            # Let z1 = y, z2 = dy/dt
            # Then dz1/dt = z2, dz2/dt = f(t, z1, z2)
            
            if isinstance(equation, str):
                equation = sp.sympify(equation)
            
            # Create system
            eq1 = "y1"  # dz1/dt = z2 (where y0=z1, y1=z2)
            eq2 = equation  # dz2/dt = f(t, y0, y1)
            
            system = [eq1, eq2]
            initial_conditions = [y0, dy0]
            
            # Solve the system
            result = self.solve_system(system, t_span, initial_conditions, method)
            
            if result['success']:
                result['position'] = result['y'][0]  # y(t)
                result['velocity'] = result['y'][1]  # dy/dt(t)
            
            self.logger.info("Solved second-order ODE")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to solve second-order ODE: {str(e)}")
    
    def boundary_value_problem(self, equation: Union[str, sp.Basic],
                              t_span: Tuple[float, float],
                              boundary_conditions: Dict[str, float],
                              n_points: int = 100) -> Dict[str, Any]:
        """
        Solve boundary value problem using shooting method
        
        Args:
            equation: Second-order ODE
            t_span: Time interval
            boundary_conditions: {'y_left': value, 'y_right': value}
            n_points: Number of discretization points
            
        Returns:
            Solution dictionary
        """
        try:
            from scipy.optimize import fsolve
            
            t0, tf = t_span
            y_left = boundary_conditions.get('y_left')
            y_right = boundary_conditions.get('y_right')
            
            if y_left is None or y_right is None:
                raise ValueError("Must specify both y_left and y_right boundary conditions")
            
            def shooting_residual(dy0_guess):
                """Residual function for shooting method"""
                try:
                    # Solve IVP with guessed initial derivative
                    result = self.solve_second_order(equation, t_span, y_left, dy0_guess[0])
                    
                    if result['success']:
                        # Check if final value matches right boundary condition
                        y_final = result['position'][-1]
                        return [y_final - y_right]
                    else:
                        return [1e6]  # Large residual if solution failed
                except:
                    return [1e6]
            
            # Initial guess for derivative at left boundary
            dy0_initial_guess = [0.0]
            
            # Solve for correct initial derivative
            dy0_solution = fsolve(shooting_residual, dy0_initial_guess)
            
            # Solve with correct initial derivative
            final_result = self.solve_second_order(equation, t_span, y_left, dy0_solution[0])
            
            final_result['boundary_conditions'] = boundary_conditions
            final_result['initial_derivative'] = dy0_solution[0]
            
            self.logger.info("Solved boundary value problem using shooting method")
            return final_result
        except Exception as e:
            raise RuntimeError(f"Failed to solve boundary value problem: {str(e)}")
    
    def stability_analysis(self, equilibria: List[float],
                          jacobian_func: Union[str, sp.Basic, Callable],
                          parameter_values: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Analyze stability of equilibrium points
        
        Args:
            equilibria: List of equilibrium points
            jacobian_func: Jacobian matrix function
            parameter_values: Parameter values to substitute
            
        Returns:
            Stability analysis results
        """
        try:
            results = {}
            
            for i, eq_point in enumerate(equilibria):
                eq_name = f"equilibrium_{i}"
                
                # Evaluate Jacobian at equilibrium point
                if callable(jacobian_func):
                    if parameter_values:
                        # Assume jacobian_func takes equilibrium point and parameters
                        J_at_eq = jacobian_func(eq_point, **parameter_values)
                    else:
                        J_at_eq = jacobian_func(eq_point)
                else:
                    # SymPy expression - substitute equilibrium point
                    if isinstance(jacobian_func, str):
                        jacobian_func = sp.sympify(jacobian_func)
                    
                    # Create symbol for state variable
                    y = sp.Symbol('y')
                    J_at_eq = jacobian_func.subs(y, eq_point)
                    
                    # Substitute parameter values if provided
                    if parameter_values:
                        for param, value in parameter_values.items():
                            param_symbol = sp.Symbol(param)
                            J_at_eq = J_at_eq.subs(param_symbol, value)
                    
                    # Convert to numerical value
                    J_at_eq = float(J_at_eq)
                
                # Analyze eigenvalues (for 1D case, just the value itself)
                if isinstance(J_at_eq, (int, float)):
                    eigenvalues = [J_at_eq]
                else:
                    # For matrices, compute eigenvalues
                    eigenvalues = np.linalg.eigvals(J_at_eq)
                
                # Determine stability
                max_real_part = max(np.real(eigenvalues))
                
                if max_real_part < 0:
                    stability = "stable"
                elif max_real_part > 0:
                    stability = "unstable"
                else:
                    stability = "marginal"
                
                results[eq_name] = {
                    'point': eq_point,
                    'jacobian': J_at_eq,
                    'eigenvalues': eigenvalues.tolist() if hasattr(eigenvalues, 'tolist') else eigenvalues,
                    'stability': stability,
                    'max_real_eigenvalue': max_real_part
                }
            
            self.logger.info(f"Analyzed stability of {len(equilibria)} equilibrium points")
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to analyze stability: {str(e)}")
    
    def phase_portrait_data(self, system: List[Union[str, sp.Basic]],
                           y_range: Tuple[float, float],
                           dy_range: Tuple[float, float],
                           grid_density: int = 20) -> Dict[str, np.ndarray]:
        """
        Generate data for phase portrait plotting
        
        Args:
            system: System of 2 first-order ODEs [dy1/dt, dy2/dt]
            y_range: Range for first variable (y1_min, y1_max)
            dy_range: Range for second variable (y2_min, y2_max)
            grid_density: Number of grid points in each direction
            
        Returns:
            Dictionary with grid data for plotting
        """
        try:
            if len(system) != 2:
                raise ValueError("Phase portrait requires exactly 2 ODEs")
            
            # Create grid
            y1_vals = np.linspace(y_range[0], y_range[1], grid_density)
            y2_vals = np.linspace(dy_range[0], dy_range[1], grid_density)
            Y1, Y2 = np.meshgrid(y1_vals, y2_vals)
            
            # Convert system to callable functions
            var_names = ['t', 'y0', 'y1']  # t, y1, y2
            func1 = self._sympify_to_callable(system[0], var_names)
            func2 = self._sympify_to_callable(system[1], var_names)
            
            # Compute derivatives at each grid point
            DY1 = np.zeros_like(Y1)
            DY2 = np.zeros_like(Y2)
            
            for i in range(grid_density):
                for j in range(grid_density):
                    y1, y2 = Y1[i, j], Y2[i, j]
                    # Note: t is not used in autonomous systems, set to 0
                    DY1[i, j] = func1(0, y1, y2)
                    DY2[i, j] = func2(0, y1, y2)
            
            # Normalize arrows for better visualization
            M = np.sqrt(DY1**2 + DY2**2)
            M[M == 0] = 1  # Avoid division by zero
            DY1_norm = DY1 / M
            DY2_norm = DY2 / M
            
            result = {
                'Y1': Y1,
                'Y2': Y2,
                'DY1': DY1,
                'DY2': DY2,
                'DY1_normalized': DY1_norm,
                'DY2_normalized': DY2_norm,
                'magnitude': M
            }
            
            self.logger.info(f"Generated phase portrait data with {grid_density}x{grid_density} grid")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to generate phase portrait data: {str(e)}")
