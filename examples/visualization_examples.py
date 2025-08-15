"""
Visualization examples for Calpyt1 framework
"""

import numpy as np
from calpyt1 import CalcEngine
import sympy as sp


def basic_2d_plotting():
    """Examples of basic 2D plotting"""
    print("=" * 60)
    print("2D PLOTTING EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Single function plot
    print("\n1. Single Function Plot:")
    print("-" * 30)
    
    functions = [
        ("sin(x)", "Sine Function"),
        ("x**2", "Quadratic Function"),
        ("exp(-x**2)", "Gaussian Function"),
        ("tan(x)", "Tangent Function")
    ]
    
    for expr, title in functions:
        print(f"Plotting: {title}")
        result = engine.plotter.plot_2d(
            expr, 
            x_range=(-2*np.pi, 2*np.pi),
            title=title,
            n_points=1000
        )
        if result['success']:
            print(f"  ✓ Successfully plotted {len(result['x_data'])} points")
    
    # Multiple functions comparison
    print("\n2. Multiple Functions Comparison:")
    print("-" * 30)
    
    functions = {
        "Linear": "x",
        "Quadratic": "x**2", 
        "Cubic": "x**3",
        "Exponential": "exp(x/2)"
    }
    
    result = engine.plotter.plot_multiple_functions(
        functions,
        x_range=(-3, 3),
        title="Function Comparison"
    )
    
    if result['success']:
        print(f"  ✓ Successfully plotted {result['n_functions']} functions")


def surface_3d_plotting():
    """Examples of 3D surface plotting"""
    print("\n" + "=" * 60)
    print("3D SURFACE PLOTTING EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Basic 3D surfaces
    print("\n1. Basic 3D Surfaces:")
    print("-" * 30)
    
    surfaces = [
        ("x**2 + y**2", "Paraboloid"),
        ("sin(x)*cos(y)", "Sine-Cosine Surface"),
        ("exp(-(x**2 + y**2))", "Gaussian Bell"),
        ("x**2 - y**2", "Hyperbolic Paraboloid (Saddle)")
    ]
    
    for expr, title in surfaces:
        print(f"Plotting: {title}")
        result = engine.plotter.plot_3d_surface(
            expr,
            x_range=(-3, 3),
            y_range=(-3, 3),
            title=title,
            n_points=50
        )
        if result['success']:
            print(f"  ✓ Successfully created 3D surface")
    
    # Interactive 3D plot
    print("\n2. Interactive 3D Plot:")
    print("-" * 30)
    
    expr = "sin(sqrt(x**2 + y**2))"
    print(f"Creating interactive plot for: {expr}")
    
    fig = engine.plotter.plot_interactive_3d(
        expr,
        x_range=(-5, 5),
        y_range=(-5, 5),
        title="Interactive Ripple Function"
    )
    
    print("  ✓ Interactive 3D plot created (use fig.show() to display)")


def gradient_field_plotting():
    """Examples of gradient field visualization"""
    print("\n" + "=" * 60)
    print("GRADIENT FIELD PLOTTING EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Gradient fields
    print("\n1. Gradient Field Visualizations:")
    print("-" * 30)
    
    functions = [
        ("x**2 + y**2", "Quadratic Bowl"),
        ("x**2 - y**2", "Saddle Point"),
        ("sin(x) + cos(y)", "Sine-Cosine Field"),
        ("exp(-(x**2 + y**2))", "Gaussian Peak")
    ]
    
    for expr, title in functions:
        print(f"Plotting gradient field: {title}")
        result = engine.plotter.plot_gradient_field(
            expr,
            x_range=(-3, 3),
            y_range=(-3, 3),
            title=f"Gradient Field: {title}",
            n_points=15
        )
        if result['success']:
            print(f"  ✓ Successfully plotted gradient field")


def phase_portrait_examples():
    """Examples of phase portrait visualization"""
    print("\n" + "=" * 60)
    print("PHASE PORTRAIT EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Linear systems
    print("\n1. Linear System Phase Portraits:")
    print("-" * 30)
    
    systems = [
        (["y1", "-y0"], "Simple Harmonic Oscillator"),
        (["-0.5*y0 + y1", "-y0 - 0.5*y1"], "Damped Oscillator"),
        (["y0", "-y1"], "Spiral"),
        (["0.1*y0", "-0.1*y1"], "Saddle Point")
    ]
    
    for system, title in systems:
        print(f"Plotting: {title}")
        
        # Define some trajectories
        trajectories = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]
        
        result = engine.plotter.plot_phase_portrait(
            system,
            x_range=(-3, 3),
            y_range=(-3, 3),
            trajectories=trajectories[:4],  # Use fewer trajectories for clarity
            title=f"Phase Portrait: {title}"
        )
        
        if result['success']:
            print(f"  ✓ Successfully plotted phase portrait")
    
    # Nonlinear systems
    print("\n2. Nonlinear System Phase Portraits:")
    print("-" * 30)
    
    # Pendulum equation: d²θ/dt² + sin(θ) = 0
    # Convert to system: dx/dt = y, dy/dt = -sin(x)
    pendulum_system = ["y1", "-sin(y0)"]
    
    print("Plotting: Nonlinear Pendulum")
    
    trajectories = [
        (0.1, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (0, 2), (1, 1), (2, 1)
    ]
    
    result = engine.plotter.plot_phase_portrait(
        pendulum_system,
        x_range=(-4, 4),
        y_range=(-3, 3),
        trajectories=trajectories[:6],
        title="Phase Portrait: Nonlinear Pendulum"
    )
    
    if result['success']:
        print("  ✓ Successfully plotted nonlinear phase portrait")


def convergence_plotting():
    """Examples of convergence visualization"""
    print("\n" + "=" * 60)
    print("CONVERGENCE PLOTTING EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Optimization convergence
    print("\n1. Optimization Convergence:")
    print("-" * 30)
    
    # Generate convergence data for different methods
    func = "x0**2 + 2*x1**2"
    x0 = [2, 3]
    
    # Gradient descent with different learning rates
    convergence_data = {}
    
    learning_rates = [0.01, 0.1, 0.5]
    
    for lr in learning_rates:
        result = engine.optimization.gradient_descent(
            func, None, x0.copy(), learning_rate=lr, max_iterations=50
        )
        
        if result['success'] and 'history' in result:
            convergence_data[f'Learning Rate {lr}'] = result['history']['f']
    
    if convergence_data:
        result = engine.plotter.plot_convergence(
            convergence_data,
            title="Gradient Descent Convergence",
            ylabel="Function Value",
            log_scale=True
        )
        
        if result['success']:
            print(f"  ✓ Successfully plotted convergence for {result['n_series']} methods")
    
    # Root finding convergence
    print("\n2. Root Finding Convergence:")
    print("-" * 30)
    
    # Demonstrate convergence of different root finding methods
    func = "x**2 - 2"
    
    # Bisection method convergence
    bisection_result = engine.root_finding.bisection(func, 1, 2, tolerance=1e-10, max_iterations=50)
    
    if bisection_result['success'] and 'history' in bisection_result:
        # Extract convergence data
        convergence_data = {
            'Bisection Method': [abs(h['fc']) for h in bisection_result['history']]
        }
        
        result = engine.plotter.plot_convergence(
            convergence_data,
            title="Root Finding Convergence",
            ylabel="Function Value (absolute)",
            log_scale=True
        )
        
        if result['success']:
            print("  ✓ Successfully plotted root finding convergence")


def mathematical_dashboard():
    """Example of creating a mathematical dashboard"""
    print("\n" + "=" * 60)
    print("MATHEMATICAL DASHBOARD EXAMPLE")
    print("=" * 60)
    
    engine = CalcEngine()
    
    print("\n1. Creating Interactive Dashboard:")
    print("-" * 30)
    
    # Define plots for dashboard
    plots_config = [
        {
            'type': '2d',
            'function': 'sin(x)',
            'x_range': (-2*np.pi, 2*np.pi),
            'title': 'Sine Function',
            'name': 'sin(x)'
        },
        {
            'type': '2d', 
            'function': 'x**2',
            'x_range': (-3, 3),
            'title': 'Quadratic Function',
            'name': 'x²'
        },
        {
            'type': '3d',
            'function': 'x**2 + y**2',
            'x_range': (-2, 2),
            'y_range': (-2, 2),
            'title': 'Paraboloid',
            'name': 'x² + y²',
            'n_points': 30
        },
        {
            'type': '2d',
            'function': 'exp(-x**2)',
            'x_range': (-3, 3),
            'title': 'Gaussian',
            'name': 'e^(-x²)'
        }
    ]
    
    fig = engine.plotter.create_dashboard(
        plots_config,
        title="Calpyt1 Mathematical Dashboard"
    )
    
    print("  ✓ Interactive dashboard created")
    print("  Use fig.show() to display the dashboard in a web browser")


def advanced_visualization_examples():
    """Advanced visualization examples"""
    print("\n" + "=" * 60)
    print("ADVANCED VISUALIZATION EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Contour plots with gradients
    print("\n1. Function Analysis with Gradient:")
    print("-" * 30)
    
    func = "x**2 + 2*y**2 + x*y"
    print(f"Analyzing function: {func}")
    
    # Find critical points
    critical_points = engine.symbolic_derivatives.get_critical_points(func, ['x', 'y'])
    print(f"Critical points: {critical_points}")
    
    # Plot gradient field
    result = engine.plotter.plot_gradient_field(
        func,
        x_range=(-3, 3),
        y_range=(-3, 3),
        title="Function Analysis with Gradient Field"
    )
    
    if result['success']:
        print("  ✓ Successfully plotted function analysis")
    
    # Taylor series visualization
    print("\n2. Taylor Series Approximation:")
    print("-" * 30)
    
    base_func = "sin(x)"
    functions = {'Original': base_func}
    
    # Add Taylor series approximations
    for order in [1, 3, 5, 7]:
        taylor = engine.taylor_series.taylor_series(base_func, 'x', 0, order)
        functions[f'Taylor O({order})'] = str(taylor)
    
    result = engine.plotter.plot_multiple_functions(
        functions,
        x_range=(-2*np.pi, 2*np.pi),
        title="Taylor Series Approximations of sin(x)"
    )
    
    if result['success']:
        print("  ✓ Successfully plotted Taylor series comparison")
    
    # Optimization landscape
    print("\n3. Optimization Landscape:")
    print("-" * 30)
    
    # Function with multiple local minima
    func = "sin(5*x)*exp(-x**2) + 0.1*x**2"
    
    # Plot the function
    result = engine.plotter.plot_2d(
        func,
        x_range=(-3, 3),
        title="Optimization Landscape: Multiple Local Minima",
        n_points=1000
    )
    
    if result['success']:
        print("  ✓ Successfully plotted optimization landscape")
    
    # Find and display local minima
    roots_result = engine.root_finding.find_all_roots(
        "5*cos(5*x)*exp(-x**2) - 2*x*sin(5*x)*exp(-x**2) + 0.2*x",  # derivative
        (-3, 3),
        100
    )
    
    if roots_result['success']:
        print(f"  Found {roots_result['n_roots']} critical points")


def comprehensive_visualization_example():
    """Comprehensive example combining multiple visualization techniques"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    engine = CalcEngine()
    
    print("\nAnalyzing the Rosenbrock Function: f(x,y) = (1-x)² + 100(y-x²)²")
    print("-" * 70)
    
    rosenbrock = "(1-x)**2 + 100*(y-x**2)**2"
    
    # 1. 3D surface plot
    print("\n1. 3D Surface Visualization:")
    print("-" * 30)
    
    result = engine.plotter.plot_3d_surface(
        rosenbrock,
        x_range=(-2, 2),
        y_range=(-1, 3),
        title="Rosenbrock Function (Banana Function)",
        n_points=50
    )
    
    if result['success']:
        print("  ✓ 3D surface plot created")
    
    # 2. Gradient field
    print("\n2. Gradient Field Analysis:")
    print("-" * 30)
    
    result = engine.plotter.plot_gradient_field(
        rosenbrock,
        x_range=(-1.5, 1.5),
        y_range=(-0.5, 2),
        title="Rosenbrock Function: Gradient Field",
        n_points=12
    )
    
    if result['success']:
        print("  ✓ Gradient field visualization created")
    
    # 3. Optimization path
    print("\n3. Optimization Path:")
    print("-" * 30)
    
    # Find minimum using optimization
    opt_result = engine.optimization.minimize_multivariable(
        "x0**2 + 100*(x1-x0**2)**2",  # Standard form for optimizer
        [-1, 0],
        method='BFGS'
    )
    
    if opt_result['success']:
        print(f"  Optimal point: ({opt_result['x'][0]:.6f}, {opt_result['x'][1]:.6f})")
        print(f"  Function value: {opt_result['fun']:.6f}")
        print(f"  Function evaluations: {opt_result['nfev']}")
    
    # 4. Interactive dashboard
    print("\n4. Interactive Dashboard:")
    print("-" * 30)
    
    dashboard_plots = [
        {
            'type': '3d',
            'function': rosenbrock,
            'x_range': (-2, 2),
            'y_range': (-1, 3),
            'title': 'Rosenbrock 3D Surface',
            'name': 'Surface',
            'n_points': 40
        },
        {
            'type': '2d',
            'function': rosenbrock.replace('y', '1'),  # Cross-section at y=1
            'x_range': (-2, 2),
            'title': 'Cross-section at y=1',
            'name': 'f(x,1)'
        }
    ]
    
    fig = engine.plotter.create_dashboard(
        dashboard_plots,
        title="Rosenbrock Function Analysis Dashboard"
    )
    
    print("  ✓ Interactive dashboard created")
    
    print("\n" + "=" * 70)
    print("Rosenbrock function analysis complete!")
    print("The Rosenbrock function is a classic optimization test case")
    print("with a global minimum at (1,1) inside a narrow, curved valley.")
    print("=" * 70)


def main():
    """Run all visualization examples"""
    print("CALPYT1 VISUALIZATION EXAMPLES")
    print("=" * 80)
    
    try:
        basic_2d_plotting()
        surface_3d_plotting()
        gradient_field_plotting()
        phase_portrait_examples()
        convergence_plotting()
        mathematical_dashboard()
        advanced_visualization_examples()
        comprehensive_visualization_example()
        
        print("\n" + "=" * 80)
        print("ALL VISUALIZATION EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNote: Some plots may require plt.show() or fig.show() to display")
        print("Run individual functions for step-by-step visualization")
        
    except Exception as e:
        print(f"\nError in visualization examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
