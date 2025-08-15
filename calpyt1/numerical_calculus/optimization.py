"""
Optimization module for Calpyt1 framework
"""

import numpy as np
import scipy.optimize as optimize
from scipy.optimize import minimize, differential_evolution, basinhopping
import sympy as sp
from typing import Union, Optional, List, Dict, Any, Callable, Tuple
from ..core.base import BaseModule


class Optimization(BaseModule):
    """
    Handles numerical optimization problems
    """
    
    def __init__(self, engine):
        """Initialize optimization module"""
        super().__init__(engine)
        self.logger.info("Optimization module initialized")
    
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
    
    def minimize_scalar(self, func: Union[str, sp.Basic, Callable],
                       bounds: Optional[Tuple[float, float]] = None,
                       method: str = 'brent',
                       **kwargs) -> Dict[str, Any]:
        """
        Minimize scalar function
        
        Args:
            func: Function to minimize
            bounds: Bounds for optimization (a, b)
            method: Optimization method ('brent', 'bounded', 'golden')
            **kwargs: Additional arguments for scipy.optimize.minimize_scalar
            
        Returns:
            Optimization result dictionary
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, ['x'])
            
            # Perform optimization
            if bounds is not None:
                result = optimize.minimize_scalar(func, bounds=bounds, method=method, **kwargs)
            else:
                result = optimize.minimize_scalar(func, method=method, **kwargs)
            
            # Format result
            output = {
                'success': result.success,
                'x': result.x,
                'fun': result.fun,
                'nfev': result.nfev,
                'message': result.message,
                'method': method
            }
            
            self.logger.info(f"Scalar minimization completed using {method} method")
            return output
        except Exception as e:
            raise RuntimeError(f"Failed to minimize scalar function: {str(e)}")
    
    def minimize_multivariable(self, func: Union[str, sp.Basic, Callable],
                              x0: Union[List[float], np.ndarray],
                              method: str = 'BFGS',
                              bounds: Optional[List[Tuple[float, float]]] = None,
                              constraints: Optional[List[Dict]] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Minimize multivariable function
        
        Args:
            func: Function to minimize f(x1, x2, ..., xn)
            x0: Initial guess
            method: Optimization method ('BFGS', 'L-BFGS-B', 'Newton-CG', 'trust-constr', etc.)
            bounds: Bounds for each variable [(min1, max1), (min2, max2), ...]
            constraints: List of constraint dictionaries
            **kwargs: Additional arguments for scipy.optimize.minimize
            
        Returns:
            Optimization result dictionary
        """
        try:
            x0 = np.array(x0)
            n_vars = len(x0)
            
            # Convert to callable if needed
            if not callable(func):
                variables = [f'x{i}' for i in range(n_vars)]
                func = self._sympify_to_callable(func, variables)
            
            # Perform optimization
            result = optimize.minimize(func, x0, method=method, bounds=bounds,
                                     constraints=constraints, **kwargs)
            
            # Format result
            output = {
                'success': result.success,
                'x': result.x.tolist(),
                'fun': result.fun,
                'nfev': result.nfev,
                'njev': getattr(result, 'njev', None),
                'nhev': getattr(result, 'nhev', None),
                'nit': result.nit,
                'message': result.message,
                'method': method
            }
            
            self.logger.info(f"Multivariable minimization completed using {method} method")
            return output
        except Exception as e:
            raise RuntimeError(f"Failed to minimize multivariable function: {str(e)}")
    
    def gradient_descent(self, func: Union[str, sp.Basic, Callable],
                        gradient: Union[str, sp.Basic, Callable, None],
                        x0: Union[List[float], np.ndarray],
                        learning_rate: float = 0.01,
                        max_iterations: int = 1000,
                        tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Custom gradient descent implementation
        
        Args:
            func: Function to minimize
            gradient: Gradient function (if None, computed numerically)
            x0: Initial point
            learning_rate: Step size
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Optimization result dictionary
        """
        try:
            x0 = np.array(x0)
            n_vars = len(x0)
            variables = [f'x{i}' for i in range(n_vars)]
            
            # Convert function to callable
            if not callable(func):
                func = self._sympify_to_callable(func, variables)
            
            # Convert gradient to callable or compute numerically
            if gradient is None:
                def gradient_func(x):
                    # Numerical gradient using finite differences
                    h = 1e-8
                    grad = np.zeros_like(x)
                    for i in range(len(x)):
                        x_plus = x.copy()
                        x_minus = x.copy()
                        x_plus[i] += h
                        x_minus[i] -= h
                        grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
                    return grad
            elif not callable(gradient):
                gradient_func = self._sympify_to_callable(gradient, variables)
            else:
                gradient_func = gradient
            
            # Gradient descent iterations
            x = x0.copy()
            history = {'x': [x.copy()], 'f': [func(x)]}
            
            for iteration in range(max_iterations):
                grad = gradient_func(x)
                x_new = x - learning_rate * grad
                
                # Check convergence
                if np.linalg.norm(x_new - x) < tolerance:
                    break
                
                x = x_new
                history['x'].append(x.copy())
                history['f'].append(func(x))
            
            result = {
                'success': iteration < max_iterations - 1,
                'x': x.tolist(),
                'fun': func(x),
                'nit': iteration + 1,
                'history': history,
                'learning_rate': learning_rate,
                'message': 'Converged' if iteration < max_iterations - 1 else 'Max iterations reached'
            }
            
            self.logger.info(f"Gradient descent completed in {iteration + 1} iterations")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to perform gradient descent: {str(e)}")
    
    def genetic_algorithm(self, func: Union[str, sp.Basic, Callable],
                         bounds: List[Tuple[float, float]],
                         population_size: int = 50,
                         max_generations: int = 100,
                         mutation_rate: float = 0.01,
                         **kwargs) -> Dict[str, Any]:
        """
        Genetic algorithm optimization using differential evolution
        
        Args:
            func: Function to minimize
            bounds: Bounds for each variable
            population_size: Size of population
            max_generations: Maximum number of generations
            mutation_rate: Mutation rate
            **kwargs: Additional arguments for differential_evolution
            
        Returns:
            Optimization result dictionary
        """
        try:
            n_vars = len(bounds)
            variables = [f'x{i}' for i in range(n_vars)]
            
            # Convert to callable if needed
            if not callable(func):
                func = self._sympify_to_callable(func, variables)
            
            # Use scipy's differential evolution
            result = differential_evolution(func, bounds, 
                                          popsize=population_size,
                                          maxiter=max_generations,
                                          **kwargs)
            
            output = {
                'success': result.success,
                'x': result.x.tolist(),
                'fun': result.fun,
                'nfev': result.nfev,
                'nit': result.nit,
                'message': result.message,
                'population_size': population_size,
                'max_generations': max_generations
            }
            
            self.logger.info(f"Genetic algorithm completed in {result.nit} generations")
            return output
        except Exception as e:
            raise RuntimeError(f"Failed to perform genetic algorithm: {str(e)}")
    
    def simulated_annealing(self, func: Union[str, sp.Basic, Callable],
                           x0: Union[List[float], np.ndarray],
                           temperature: float = 1.0,
                           cooling_rate: float = 0.95,
                           min_temperature: float = 1e-8,
                           max_iterations: int = 1000) -> Dict[str, Any]:
        """
        Simulated annealing optimization using basin hopping
        
        Args:
            func: Function to minimize
            x0: Initial point
            temperature: Initial temperature
            cooling_rate: Temperature reduction factor
            min_temperature: Minimum temperature
            max_iterations: Maximum iterations
            
        Returns:
            Optimization result dictionary
        """
        try:
            x0 = np.array(x0)
            n_vars = len(x0)
            variables = [f'x{i}' for i in range(n_vars)]
            
            # Convert to callable if needed
            if not callable(func):
                func = self._sympify_to_callable(func, variables)
            
            # Custom step function for temperature-based acceptance
            class TemperatureBasedStep:
                def __init__(self, temperature, cooling_rate):
                    self.temperature = temperature
                    self.cooling_rate = cooling_rate
                
                def __call__(self, x):
                    # Random step
                    step_size = self.temperature * 0.1
                    return x + np.random.normal(0, step_size, size=len(x))
            
            # Custom acceptance function
            class TemperatureBasedAccept:
                def __init__(self, temperature, cooling_rate, min_temp):
                    self.temperature = temperature
                    self.cooling_rate = cooling_rate
                    self.min_temp = min_temp
                
                def __call__(self, f_new, x_new, f_old, x_old):
                    if f_new < f_old:
                        accept = True
                    else:
                        # Metropolis criterion
                        delta = f_new - f_old
                        prob = np.exp(-delta / max(self.temperature, self.min_temp))
                        accept = np.random.random() < prob
                    
                    # Cool down
                    self.temperature *= self.cooling_rate
                    return accept
            
            # Set up step and acceptance functions
            step_function = TemperatureBasedStep(temperature, cooling_rate)
            accept_function = TemperatureBasedAccept(temperature, cooling_rate, min_temperature)
            
            # Use basin hopping with custom functions
            result = basinhopping(func, x0, 
                                niter=max_iterations,
                                T=temperature,
                                take_step=step_function,
                                accept_test=accept_function)
            
            output = {
                'success': True,  # Basin hopping doesn't have success attribute
                'x': result.x.tolist(),
                'fun': result.fun,
                'nfev': result.nfev,
                'nit': result.nit,
                'message': 'Simulated annealing completed',
                'initial_temperature': temperature,
                'final_temperature': accept_function.temperature
            }
            
            self.logger.info(f"Simulated annealing completed in {result.nit} iterations")
            return output
        except Exception as e:
            raise RuntimeError(f"Failed to perform simulated annealing: {str(e)}")
    
    def constrained_optimization(self, objective: Union[str, sp.Basic, Callable],
                                x0: Union[List[float], np.ndarray],
                                equality_constraints: Optional[List[Union[str, sp.Basic]]] = None,
                                inequality_constraints: Optional[List[Union[str, sp.Basic]]] = None,
                                bounds: Optional[List[Tuple[float, float]]] = None,
                                method: str = 'trust-constr') -> Dict[str, Any]:
        """
        Solve constrained optimization problem
        
        Args:
            objective: Objective function to minimize
            x0: Initial point
            equality_constraints: List of equality constraint functions g(x) = 0
            inequality_constraints: List of inequality constraint functions h(x) >= 0
            bounds: Variable bounds
            method: Optimization method
            
        Returns:
            Optimization result dictionary
        """
        try:
            x0 = np.array(x0)
            n_vars = len(x0)
            variables = [f'x{i}' for i in range(n_vars)]
            
            # Convert objective to callable
            if not callable(objective):
                objective_func = self._sympify_to_callable(objective, variables)
            else:
                objective_func = objective
            
            # Prepare constraints
            constraints = []
            
            # Equality constraints
            if equality_constraints:
                for eq_constraint in equality_constraints:
                    if not callable(eq_constraint):
                        eq_func = self._sympify_to_callable(eq_constraint, variables)
                    else:
                        eq_func = eq_constraint
                    
                    constraints.append({'type': 'eq', 'fun': eq_func})
            
            # Inequality constraints
            if inequality_constraints:
                for ineq_constraint in inequality_constraints:
                    if not callable(ineq_constraint):
                        ineq_func = self._sympify_to_callable(ineq_constraint, variables)
                    else:
                        ineq_func = ineq_constraint
                    
                    constraints.append({'type': 'ineq', 'fun': ineq_func})
            
            # Solve constrained optimization
            result = optimize.minimize(objective_func, x0, 
                                     method=method,
                                     bounds=bounds,
                                     constraints=constraints if constraints else None)
            
            output = {
                'success': result.success,
                'x': result.x.tolist(),
                'fun': result.fun,
                'nfev': result.nfev,
                'njev': getattr(result, 'njev', None),
                'nit': result.nit,
                'message': result.message,
                'method': method,
                'n_equality_constraints': len(equality_constraints) if equality_constraints else 0,
                'n_inequality_constraints': len(inequality_constraints) if inequality_constraints else 0
            }
            
            self.logger.info(f"Constrained optimization completed using {method}")
            return output
        except Exception as e:
            raise RuntimeError(f"Failed to solve constrained optimization: {str(e)}")
    
    def multi_objective_optimization(self, objectives: List[Union[str, sp.Basic, Callable]],
                                   x0: Union[List[float], np.ndarray],
                                   weights: Optional[List[float]] = None,
                                   method: str = 'weighted_sum') -> Dict[str, Any]:
        """
        Multi-objective optimization
        
        Args:
            objectives: List of objective functions
            x0: Initial point
            weights: Weights for each objective (for weighted sum method)
            method: Multi-objective method ('weighted_sum', 'pareto_front')
            
        Returns:
            Optimization result dictionary
        """
        try:
            x0 = np.array(x0)
            n_vars = len(x0)
            n_objectives = len(objectives)
            variables = [f'x{i}' for i in range(n_vars)]
            
            # Convert objectives to callable functions
            objective_funcs = []
            for obj in objectives:
                if not callable(obj):
                    obj_func = self._sympify_to_callable(obj, variables)
                else:
                    obj_func = obj
                objective_funcs.append(obj_func)
            
            if method == 'weighted_sum':
                # Weighted sum approach
                if weights is None:
                    weights = [1.0 / n_objectives] * n_objectives
                elif len(weights) != n_objectives:
                    raise ValueError("Number of weights must match number of objectives")
                
                def combined_objective(x):
                    return sum(w * f(x) for w, f in zip(weights, objective_funcs))
                
                result = optimize.minimize(combined_objective, x0)
                
                # Evaluate all objectives at solution
                objective_values = [f(result.x) for f in objective_funcs]
                
                output = {
                    'success': result.success,
                    'x': result.x.tolist(),
                    'objective_values': objective_values,
                    'weighted_sum': result.fun,
                    'weights': weights,
                    'method': method,
                    'nfev': result.nfev,
                    'message': result.message
                }
                
            elif method == 'pareto_front':
                # Simple Pareto front approximation using multiple weighted sums
                n_points = 10
                pareto_solutions = []
                
                for i in range(n_points):
                    # Generate different weight combinations
                    w1 = i / (n_points - 1)
                    w2 = 1 - w1
                    current_weights = [w1, w2] + [0] * (n_objectives - 2)
                    
                    def combined_objective(x):
                        return sum(w * f(x) for w, f in zip(current_weights, objective_funcs))
                    
                    result = optimize.minimize(combined_objective, x0)
                    
                    if result.success:
                        obj_values = [f(result.x) for f in objective_funcs]
                        pareto_solutions.append({
                            'x': result.x.tolist(),
                            'objectives': obj_values,
                            'weights': current_weights.copy()
                        })
                
                output = {
                    'success': len(pareto_solutions) > 0,
                    'pareto_solutions': pareto_solutions,
                    'n_solutions': len(pareto_solutions),
                    'method': method,
                    'message': f'Found {len(pareto_solutions)} Pareto solutions'
                }
            
            else:
                raise ValueError(f"Unknown multi-objective method: {method}")
            
            self.logger.info(f"Multi-objective optimization completed using {method}")
            return output
        except Exception as e:
            raise RuntimeError(f"Failed to perform multi-objective optimization: {str(e)}")
    
    def line_search(self, func: Union[str, sp.Basic, Callable],
                   gradient: Union[str, sp.Basic, Callable],
                   x: Union[List[float], np.ndarray],
                   direction: Union[List[float], np.ndarray],
                   method: str = 'wolfe') -> Dict[str, Any]:
        """
        Perform line search to find optimal step size
        
        Args:
            func: Objective function
            gradient: Gradient function
            x: Current point
            direction: Search direction
            method: Line search method ('wolfe', 'armijo', 'exact')
            
        Returns:
            Line search result dictionary
        """
        try:
            x = np.array(x)
            direction = np.array(direction)
            n_vars = len(x)
            variables = [f'x{i}' for i in range(n_vars)]
            
            # Convert functions to callable
            if not callable(func):
                func = self._sympify_to_callable(func, variables)
            if not callable(gradient):
                gradient = self._sympify_to_callable(gradient, variables)
            
            if method == 'exact':
                # Exact line search by minimizing along the direction
                def line_func(alpha):
                    return func(x + alpha * direction)
                
                alpha_result = optimize.minimize_scalar(line_func)
                optimal_alpha = alpha_result.x
                
            elif method in ['wolfe', 'armijo']:
                # Use scipy's line search
                from scipy.optimize.linesearch import line_search_wolfe2
                
                def func_and_grad(x_new):
                    return func(x_new), gradient(x_new)
                
                alpha_result = line_search_wolfe2(func, func_and_grad, x, direction)
                optimal_alpha = alpha_result[0] if alpha_result[0] is not None else 1.0
                
            else:
                raise ValueError(f"Unknown line search method: {method}")
            
            # Compute new point and function value
            x_new = x + optimal_alpha * direction
            f_new = func(x_new)
            
            result = {
                'success': True,
                'alpha': optimal_alpha,
                'x_new': x_new.tolist(),
                'f_new': f_new,
                'method': method,
                'direction': direction.tolist()
            }
            
            self.logger.info(f"Line search completed using {method} method")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to perform line search: {str(e)}")
