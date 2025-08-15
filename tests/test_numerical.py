"""
Unit tests for numerical calculus module
"""

import pytest
import numpy as np
from calpyt1 import CalcEngine


class TestNumericalIntegration:
    """Test cases for numerical integration"""
    
    def setup_method(self):
        """Setup test fixture"""
        self.engine = CalcEngine()
    
    def test_quad_integration(self):
        """Test adaptive quadrature integration"""
        # Test polynomial integration
        result, error = self.engine.numerical_integration.quad("x**2", 0, 1)
        expected = 1/3
        assert abs(result - expected) < 1e-10
        assert error < 1e-10
        
        # Test trigonometric integration
        result, error = self.engine.numerical_integration.quad("sin(x)", 0, np.pi)
        expected = 2.0
        assert abs(result - expected) < 1e-10
    
    def test_simpson_integration(self):
        """Test Simpson's rule integration"""
        result = self.engine.numerical_integration.simpson("x**2", 0, 1, 1000)
        expected = 1/3
        assert abs(result - expected) < 1e-6
        
        # Test with fewer points (should be less accurate)
        result_few = self.engine.numerical_integration.simpson("x**2", 0, 1, 10)
        assert abs(result_few - expected) > abs(result - expected)
    
    def test_trapezoidal_integration(self):
        """Test trapezoidal rule integration"""
        result = self.engine.numerical_integration.trapezoidal("x**2", 0, 1, 1000)
        expected = 1/3
        assert abs(result - expected) < 1e-4
    
    def test_monte_carlo_integration(self):
        """Test Monte Carlo integration"""
        # Test 1D integration
        result, error = self.engine.numerical_integration.monte_carlo("1", [(0, 2)], 10000, seed=42)
        expected = 2.0  # Integral of 1 from 0 to 2
        assert abs(result - expected) < 0.1  # Monte Carlo has higher error
        
        # Test 2D integration
        result, error = self.engine.numerical_integration.monte_carlo("1", [(0, 1), (0, 1)], 10000, seed=42)
        expected = 1.0  # Area of unit square
        assert abs(result - expected) < 0.1
    
    def test_gaussian_quadrature(self):
        """Test Gaussian quadrature"""
        # Test on [-1, 1] interval (natural for Gauss-Legendre)
        result = self.engine.numerical_integration.gaussian_quadrature("x**2", -1, 1, 5, "legendre")
        expected = 2/3  # Integral of x^2 from -1 to 1
        assert abs(result - expected) < 1e-10
    
    def test_compare_methods(self):
        """Test method comparison functionality"""
        exact_value = 1/3
        comparison = self.engine.numerical_integration.compare_methods("x**2", 0, 1, exact_value)
        
        # Check that comparison contains expected methods
        assert 'adaptive_quad' in comparison
        assert 'simpson' in comparison
        assert 'trapezoidal' in comparison
        
        # Check that all methods give reasonable results
        for method, result in comparison.items():
            if 'error' not in result:
                assert abs(result['value'] - exact_value) < 1e-3


class TestODESolver:
    """Test cases for ODE solving"""
    
    def setup_method(self):
        """Setup test fixture"""
        self.engine = CalcEngine()
    
    def test_euler_method(self):
        """Test Euler's method"""
        # Test dy/dt = -y, y(0) = 1
        # Analytical solution: y(t) = exp(-t)
        t_vals, y_vals = self.engine.ode_solver.euler_method("-y", 0, 1, 1, 0.01)
        
        # Check final value
        analytical_final = np.exp(-1)
        assert abs(y_vals[-1] - analytical_final) < 0.1  # Euler is not very accurate
    
    def test_rk4_method(self):
        """Test 4th order Runge-Kutta method"""
        # Test dy/dt = -y, y(0) = 1
        t_vals, y_vals = self.engine.ode_solver.runge_kutta_4("-y", 0, 1, 1, 0.01)
        
        # Check final value (RK4 should be more accurate than Euler)
        analytical_final = np.exp(-1)
        assert abs(y_vals[-1] - analytical_final) < 1e-4
    
    def test_solve_ivp_wrapper(self):
        """Test solve_ivp wrapper"""
        result = self.engine.ode_solver.solve_ivp_wrapper("-y", (0, 1), 1, method='RK45')
        
        assert result['success'] == True
        assert len(result['t']) > 1
        assert len(result['y'][0]) == len(result['t'])
        
        # Check final value
        analytical_final = np.exp(-1)
        assert abs(result['y'][0][-1] - analytical_final) < 1e-6
    
    def test_solve_system(self):
        """Test system of ODEs"""
        # Simple system: dx/dt = y, dy/dt = -x (harmonic oscillator)
        system = ["y1", "-y0"]
        t_span = (0, np.pi/2)
        y0 = [1, 0]  # x(0) = 1, y(0) = 0
        
        result = self.engine.ode_solver.solve_system(system, t_span, y0)
        
        assert result['success'] == True
        
        # At t = π/2, analytical solution is x = 0, y = 1
        final_x = result['y'][0][-1]
        final_y = result['y'][1][-1]
        
        assert abs(final_x) < 1e-3
        assert abs(final_y - 1) < 1e-3
    
    def test_second_order_ode(self):
        """Test second-order ODE conversion"""
        # Test d²y/dt² = -y (simple harmonic oscillator)
        # Initial conditions: y(0) = 1, dy/dt(0) = 0
        result = self.engine.ode_solver.solve_second_order("-y0", (0, np.pi), 1, 0)
        
        assert result['success'] == True
        
        # At t = π, analytical solution is y = -1
        final_position = result['position'][-1]
        assert abs(final_position + 1) < 1e-3


class TestOptimization:
    """Test cases for optimization"""
    
    def setup_method(self):
        """Setup test fixture"""
        self.engine = CalcEngine()
    
    def test_minimize_scalar(self):
        """Test scalar function minimization"""
        # Minimize x^2 - 4x + 3 (minimum at x = 2)
        result = self.engine.optimization.minimize_scalar("x**2 - 4*x + 3", bounds=(0, 5))
        
        assert result['success'] == True
        assert abs(result['x'] - 2) < 1e-6
        assert abs(result['fun'] + 1) < 1e-6  # f(2) = -1
    
    def test_minimize_multivariable(self):
        """Test multivariable function minimization"""
        # Minimize x^2 + y^2 (minimum at (0, 0))
        result = self.engine.optimization.minimize_multivariable("x0**2 + x1**2", [1, 1])
        
        assert result['success'] == True
        assert abs(result['x'][0]) < 1e-6
        assert abs(result['x'][1]) < 1e-6
        assert abs(result['fun']) < 1e-10
    
    def test_gradient_descent(self):
        """Test gradient descent implementation"""
        # Minimize x^2 + y^2
        result = self.engine.optimization.gradient_descent(
            "x0**2 + x1**2", None, [1, 1], learning_rate=0.1, max_iterations=100
        )
        
        assert result['success'] == True
        assert abs(result['x'][0]) < 1e-3
        assert abs(result['x'][1]) < 1e-3
        assert result['fun'] < 1e-6
    
    def test_genetic_algorithm(self):
        """Test genetic algorithm"""
        # Simple quadratic function
        result = self.engine.optimization.genetic_algorithm(
            "x0**2 + x1**2", [(-2, 2), (-2, 2)], population_size=20, max_generations=50
        )
        
        assert result['success'] == True
        assert abs(result['fun']) < 0.1  # GA may not be as precise
    
    def test_constrained_optimization(self):
        """Test constrained optimization"""
        # Minimize x^2 + y^2 subject to x + y = 1
        result = self.engine.optimization.constrained_optimization(
            "x0**2 + x1**2", [0.5, 0.5], equality_constraints=["x0 + x1 - 1"]
        )
        
        assert result['success'] == True
        
        # Optimal solution should be x = y = 0.5
        assert abs(result['x'][0] - 0.5) < 1e-3
        assert abs(result['x'][1] - 0.5) < 1e-3
        
        # Check constraint satisfaction
        constraint_value = result['x'][0] + result['x'][1] - 1
        assert abs(constraint_value) < 1e-6
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization"""
        objectives = ["x0**2", "x1**2"]
        result = self.engine.optimization.multi_objective_optimization(
            objectives, [1, 1], method='weighted_sum'
        )
        
        assert result['success'] == True
        assert len(result['objective_values']) == 2


class TestRootFinding:
    """Test cases for root finding"""
    
    def setup_method(self):
        """Setup test fixture"""
        self.engine = CalcEngine()
    
    def test_bisection_method(self):
        """Test bisection method"""
        # Find root of x^2 - 4 (roots at ±2)
        result = self.engine.root_finding.bisection("x**2 - 4", 1, 3, tolerance=1e-6)
        
        assert result['success'] == True
        assert abs(result['root'] - 2) < 1e-6
        assert abs(result['function_value']) < 1e-6
    
    def test_newton_raphson(self):
        """Test Newton-Raphson method"""
        # Find root of x^2 - 4
        result = self.engine.root_finding.newton_raphson("x**2 - 4", None, 1.5, tolerance=1e-10)
        
        assert result['success'] == True
        assert abs(result['root'] - 2) < 1e-10
    
    def test_secant_method(self):
        """Test secant method"""
        # Find root of x^2 - 4
        result = self.engine.root_finding.secant("x**2 - 4", 1.5, 2.5, tolerance=1e-8)
        
        assert result['success'] == True
        assert abs(result['root'] - 2) < 1e-8
    
    def test_brent_method(self):
        """Test Brent's method"""
        # Find root of x^2 - 4
        result = self.engine.root_finding.brent("x**2 - 4", 1, 3, tolerance=1e-10)
        
        assert result['success'] == True
        assert abs(result['root'] - 2) < 1e-10
    
    def test_solve_system(self):
        """Test system of nonlinear equations"""
        # Solve x^2 + y^2 = 4, x - y = 0 (solutions at (√2, √2) and (-√2, -√2))
        equations = ["x0**2 + x1**2 - 4", "x0 - x1"]
        result = self.engine.root_finding.solve_system(equations, [1, 1])
        
        assert result['success'] == True
        
        # Check that solution satisfies equations
        x, y = result['x']
        assert abs(x**2 + y**2 - 4) < 1e-6
        assert abs(x - y) < 1e-6
        
        # Check that it's one of the expected solutions
        sqrt2 = np.sqrt(2)
        is_positive_solution = abs(x - sqrt2) < 1e-3 and abs(y - sqrt2) < 1e-3
        is_negative_solution = abs(x + sqrt2) < 1e-3 and abs(y + sqrt2) < 1e-3
        assert is_positive_solution or is_negative_solution
    
    def test_polynomial_roots(self):
        """Test polynomial root finding"""
        # Test x^2 - 3x + 2 = (x-1)(x-2)
        coeffs = [1, -3, 2]
        result = self.engine.root_finding.polynomial_roots(coeffs)
        
        assert result['success'] == True
        assert result['n_real_roots'] == 2
        
        roots = sorted(result['real_roots'])
        assert abs(roots[0] - 1) < 1e-10
        assert abs(roots[1] - 2) < 1e-10
    
    def test_find_all_roots(self):
        """Test finding all roots in a range"""
        # Function with multiple roots: (x-1)(x-2)(x-3)
        func = "(x-1)*(x-2)*(x-3)"
        result = self.engine.root_finding.find_all_roots(func, (0, 4), n_intervals=100)
        
        assert result['success'] == True
        assert result['n_roots'] == 3
        
        roots = sorted(result['roots'])
        assert abs(roots[0] - 1) < 1e-6
        assert abs(roots[1] - 2) < 1e-6
        assert abs(roots[2] - 3) < 1e-6
    
    def test_compare_methods(self):
        """Test method comparison"""
        initial_data = {
            'bracket': [1, 3],
            'guess': 2,
            'second_guess': 1.5
        }
        
        comparison = self.engine.root_finding.compare_methods("x**2 - 4", initial_data)
        
        # Check that multiple methods are tested
        assert len(comparison) >= 3
        
        # Check that at least some methods converged
        successful_methods = [method for method, result in comparison.items() 
                             if not 'error' in result and result.get('converged', False)]
        assert len(successful_methods) >= 2


class TestNumericalIntegration:
    """Integration tests for numerical methods"""
    
    def setup_method(self):
        """Setup test fixture"""
        self.engine = CalcEngine()
    
    def test_integration_ode_consistency(self):
        """Test consistency between integration and ODE solving"""
        # If dy/dt = f(t), then y(t) = y(0) + ∫f(t)dt from 0 to t
        
        # Test with f(t) = t (so y(t) = y(0) + t²/2)
        func = "t"
        
        # Solve ODE: dy/dt = t, y(0) = 0
        ode_result = self.engine.ode_solver.solve_ivp_wrapper(func, (0, 2), 0)
        
        # Integrate f(t) = t from 0 to 2
        integral_result, _ = self.engine.numerical_integration.quad(func, 0, 2)
        
        assert ode_result['success']
        
        # Final value from ODE should match integral result
        final_ode_value = ode_result['y'][0][-1]
        assert abs(final_ode_value - integral_result) < 1e-6
    
    def test_optimization_root_finding_consistency(self):
        """Test consistency between optimization and root finding"""
        # Critical points of f(x) are roots of f'(x)
        
        func = "x**3 - 3*x**2 + 2*x"
        derivative = "3*x**2 - 6*x + 2"
        
        # Find minimum using optimization
        opt_result = self.engine.optimization.minimize_scalar(func, bounds=(0, 3))
        
        # Find roots of derivative
        root_result = self.engine.root_finding.find_all_roots(derivative, (0, 3))
        
        assert opt_result['success']
        assert root_result['success']
        
        # The minimum should be at one of the critical points
        min_x = opt_result['x']
        min_distance_to_root = min(abs(min_x - root) for root in root_result['roots'])
        assert min_distance_to_root < 1e-6
    
    def test_numerical_symbolic_consistency(self):
        """Test that numerical methods agree with symbolic results when possible"""
        # Compare numerical and symbolic integration
        
        # Function that can be integrated symbolically
        func = "x**2"
        
        # Numerical integration
        numerical_result, _ = self.engine.numerical_integration.quad(func, 0, 1)
        
        # The exact result should be 1/3
        exact_result = 1/3
        
        assert abs(numerical_result - exact_result) < 1e-10
        
        # Test with symbolic derivative verification
        # If F'(x) = f(x), then ∫f(x)dx from a to b = F(b) - F(a)
        # For f(x) = x^2, F(x) = x^3/3
        
        F_b = (1**3) / 3
        F_a = (0**3) / 3
        symbolic_result = F_b - F_a
        
        assert abs(numerical_result - symbolic_result) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
