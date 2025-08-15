"""
Numerical calculus examples for Calpyt1 framework
"""

import numpy as np
from calpyt1 import CalcEngine
import sympy as sp


def numerical_integration_examples():
    """Examples of numerical integration"""
    print("=" * 60)
    print("NUMERICAL INTEGRATION EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Basic numerical integration
    print("\n1. Basic Numerical Integration:")
    print("-" * 30)
    
    examples = [
        ("x**2", 0, 1),
        ("sin(x)", 0, np.pi),
        ("exp(-x**2)", -2, 2),
        ("1/(1+x**2)", -5, 5)
    ]
    
    for expr, a, b in examples:
        result, error = engine.numerical_integration.quad(expr, a, b)
        print(f"∫[{a:.1f} to {b:.1f}] ({expr}) dx = {result:.6f} ± {error:.2e}")
    
    # Compare integration methods
    print("\n2. Method Comparison:")
    print("-" * 30)
    
    expr = "sin(x)"
    a, b = 0, np.pi
    exact_value = 2.0  # Exact value of ∫[0 to π] sin(x) dx
    
    comparison = engine.numerical_integration.compare_methods(expr, a, b, exact_value)
    
    for method, result in comparison.items():
        if 'error' not in result:
            value = result['value']
            abs_error = result.get('absolute_error', 'N/A')
            print(f"{method:15s}: {value:.8f} (error: {abs_error})")
    
    # Monte Carlo integration
    print("\n3. Monte Carlo Integration:")
    print("-" * 30)
    
    expr = "x**2 + y**2"
    bounds = [(-1, 1), (-1, 1)]  # Unit square
    result, error = engine.numerical_integration.monte_carlo(expr, bounds, 10000)
    print(f"∫∫ ({expr}) dxdy over unit square = {result:.6f} ± {error:.6f}")


def ode_solver_examples():
    """Examples of ODE solving"""
    print("\n" + "=" * 60)
    print("ODE SOLVER EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Simple first-order ODE
    print("\n1. First-order ODE:")
    print("-" * 30)
    
    # dy/dt = -2*y + 1, y(0) = 0
    func = "-2*y + 1"
    t_span = (0, 5)
    y0 = 0
    
    result = engine.ode_solver.solve_ivp_wrapper(func, t_span, y0)
    
    if result['success']:
        print(f"Solved: dy/dt = {func}, y(0) = {y0}")
        print(f"Solution at t = {result['t'][-1]:.2f}: y = {result['y'][0][-1]:.6f}")
        print(f"Function evaluations: {result['nfev']}")
    
    # Compare numerical methods
    print("\n2. Method Comparison (Euler vs RK4):")
    print("-" * 30)
    
    func = "-y"  # dy/dt = -y, analytical solution: y = e^(-t)
    t0, tf, h = 0, 2, 0.1
    y0 = 1
    
    # Euler method
    t_euler, y_euler = engine.ode_solver.euler_method(func, t0, y0, tf, h)
    
    # RK4 method
    t_rk4, y_rk4 = engine.ode_solver.runge_kutta_4(func, t0, y0, tf, h)
    
    # Analytical solution at final time
    analytical = np.exp(-tf)
    
    print(f"Analytical solution at t={tf}: {analytical:.6f}")
    print(f"Euler method result: {y_euler[-1]:.6f} (error: {abs(y_euler[-1] - analytical):.2e})")
    print(f"RK4 method result: {y_rk4[-1]:.6f} (error: {abs(y_rk4[-1] - analytical):.2e})")
    
    # System of ODEs
    print("\n3. System of ODEs (Predator-Prey):")
    print("-" * 30)
    
    # Lotka-Volterra equations
    # dx/dt = a*x - b*x*y
    # dy/dt = -c*y + d*x*y
    system = [
        "x - 0.1*x*y0",  # dx/dt (x=y0, y=y1 in system notation)
        "-1.5*y1 + 0.075*y0*y1"  # dy/dt
    ]
    
    t_span = (0, 15)
    y0 = [10, 5]  # Initial: 10 prey, 5 predators
    
    result = engine.ode_solver.solve_system(system, t_span, y0)
    
    if result['success']:
        print("Solved Lotka-Volterra system")
        print(f"Final populations: Prey = {result['y'][0][-1]:.2f}, Predators = {result['y'][1][-1]:.2f}")


def optimization_examples():
    """Examples of optimization"""
    print("\n" + "=" * 60)
    print("OPTIMIZATION EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Scalar optimization
    print("\n1. Scalar Optimization:")
    print("-" * 30)
    
    examples = [
        ("x**2 - 4*x + 3", (-5, 5)),
        ("sin(x) + 0.1*x**2", (-10, 10)),
        ("exp(x) - 2*x", (-2, 3))
    ]
    
    for expr, bounds in examples:
        result = engine.optimization.minimize_scalar(expr, bounds=bounds)
        print(f"Min of {expr} in {bounds}: x = {result['x']:.6f}, f = {result['fun']:.6f}")
    
    # Multivariable optimization
    print("\n2. Multivariable Optimization:")
    print("-" * 30)
    
    examples = [
        ("x0**2 + x1**2", [1, 1]),
        ("(x0-1)**2 + (x1-2)**2", [0, 0]),
        ("x0**2 + 2*x1**2 + x0*x1", [1, 1])
    ]
    
    for expr, x0 in examples:
        result = engine.optimization.minimize_multivariable(expr, x0)
        if result['success']:
            print(f"Min of {expr}: x = {result['x']}, f = {result['fun']:.6f}")
    
    # Constrained optimization
    print("\n3. Constrained Optimization:")
    print("-" * 30)
    
    objective = "x0**2 + x1**2"
    equality_constraints = ["x0 + x1 - 1"]  # x + y = 1
    x0 = [0.5, 0.5]
    
    result = engine.optimization.constrained_optimization(
        objective, x0, equality_constraints=equality_constraints
    )
    
    if result['success']:
        print(f"Min of {objective} subject to x+y=1:")
        print(f"  x = {result['x']}, f = {result['fun']:.6f}")
    
    # Gradient descent
    print("\n4. Gradient Descent:")
    print("-" * 30)
    
    func = "x0**2 + 2*x1**2"
    x0 = [2, 2]
    
    result = engine.optimization.gradient_descent(func, None, x0, learning_rate=0.1, max_iterations=100)
    
    if result['success']:
        print(f"Gradient descent on {func}:")
        print(f"  Final point: {result['x']}")
        print(f"  Final value: {result['fun']:.6f}")
        print(f"  Iterations: {result['nit']}")


def root_finding_examples():
    """Examples of root finding"""
    print("\n" + "=" * 60)
    print("ROOT FINDING EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Single root finding
    print("\n1. Single Root Finding:")
    print("-" * 30)
    
    examples = [
        ("x**2 - 4", [-3, 3]),
        ("sin(x)", [3, 4]),
        ("exp(x) - 2", [0, 2]),
        ("x**3 - x - 1", [1, 2])
    ]
    
    for expr, bracket in examples:
        result = engine.root_finding.brent(expr, bracket[0], bracket[1])
        if result['success']:
            print(f"Root of {expr} in {bracket}: x = {result['root']:.6f}")
    
    # Method comparison
    print("\n2. Method Comparison:")
    print("-" * 30)
    
    expr = "x**2 - 2"  # Root at x = √2 ≈ 1.414213562
    initial_data = {
        'bracket': [1, 2],
        'guess': 1.5,
        'second_guess': 1.4
    }
    
    comparison = engine.root_finding.compare_methods(expr, initial_data)
    
    exact_root = np.sqrt(2)
    print(f"Exact root: {exact_root:.8f}")
    
    for method, result in comparison.items():
        if 'error' not in result and result['converged']:
            root = result['result']['root']
            error = abs(root - exact_root)
            iterations = result['iterations']
            print(f"{method:15s}: {root:.8f} (error: {error:.2e}, iter: {iterations})")
    
    # System of equations
    print("\n3. System of Nonlinear Equations:")
    print("-" * 30)
    
    # Solve: x^2 + y^2 = 4, x - y = 0
    equations = ["x0**2 + x1**2 - 4", "x0 - x1"]
    x0 = [1, 1]
    
    result = engine.root_finding.solve_system(equations, x0)
    
    if result['success']:
        print("Solved system: x² + y² = 4, x - y = 0")
        print(f"Solution: x = {result['x'][0]:.6f}, y = {result['x'][1]:.6f}")
        print(f"Residual: {result['residual']:.2e}")
    
    # Polynomial roots
    print("\n4. Polynomial Roots:")
    print("-" * 30)
    
    # x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
    coefficients = [1, -6, 11, -6]
    result = engine.root_finding.polynomial_roots(coefficients)
    
    print(f"Roots of polynomial with coefficients {coefficients}:")
    print(f"Real roots: {result['real_roots']}")
    if result['complex_roots']:
        print(f"Complex roots: {result['complex_roots']}")


def comprehensive_numerical_example():
    """Comprehensive numerical example"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE NUMERICAL EXAMPLE")
    print("=" * 60)
    
    engine = CalcEngine()
    
    print("\nAnalyzing the function f(x) = x³ - 6x² + 9x + 1")
    
    # 1. Find critical points numerically
    print("\n1. Finding Critical Points:")
    print("-" * 30)
    
    # f'(x) = 3x² - 12x + 9
    derivative = "3*x**2 - 12*x + 9"
    
    # Find roots of derivative
    roots = engine.root_finding.find_all_roots(derivative, (-1, 5), 100)
    print(f"Critical points: {roots['roots']}")
    
    # 2. Evaluate function at critical points
    print("\n2. Function Values at Critical Points:")
    print("-" * 30)
    
    def f(x):
        return x**3 - 6*x**2 + 9*x + 1
    
    for root in roots['roots']:
        value = f(root)
        print(f"f({root:.6f}) = {value:.6f}")
    
    # 3. Find minimum numerically
    print("\n3. Numerical Minimization:")
    print("-" * 30)
    
    func = "x**3 - 6*x**2 + 9*x + 1"
    result = engine.optimization.minimize_scalar(func, bounds=(0, 4))
    
    if result['success']:
        print(f"Minimum at x = {result['x']:.6f}, f = {result['fun']:.6f}")
    
    # 4. Integrate the function
    print("\n4. Numerical Integration:")
    print("-" * 30)
    
    integral, error = engine.numerical_integration.quad(func, 0, 3)
    print(f"∫[0 to 3] f(x) dx = {integral:.6f} ± {error:.2e}")
    
    # 5. Solve related ODE
    print("\n5. Related ODE (dy/dt = f(t)):")
    print("-" * 30)
    
    ode_func = "t**3 - 6*t**2 + 9*t + 1"
    t_span = (0, 2)
    y0 = 0
    
    ode_result = engine.ode_solver.solve_ivp_wrapper(ode_func, t_span, y0)
    
    if ode_result['success']:
        print(f"Solution at t = {ode_result['t'][-1]:.2f}: y = {ode_result['y'][0][-1]:.6f}")


def main():
    """Run all numerical examples"""
    print("CALPYT1 NUMERICAL CALCULUS EXAMPLES")
    print("=" * 80)
    
    try:
        numerical_integration_examples()
        ode_solver_examples()
        optimization_examples()
        root_finding_examples()
        comprehensive_numerical_example()
        
        print("\n" + "=" * 80)
        print("ALL NUMERICAL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError in numerical examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
