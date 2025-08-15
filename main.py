#!/usr/bin/env python3
"""
Calpyt1 - Comprehensive Python Mathematical Computing Framework
Main demonstration script and application entry point

This script demonstrates the key capabilities of Calpyt1 across all modules
and provides an interactive interface for exploring the framework.
"""

import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt
from calpyt1 import CalcEngine
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import time


def print_banner():
    """Print the Calpyt1 banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                          CALPYT1                              ║
    ║         Comprehensive Python Mathematical Computing          ║
    ║                        Framework v0.1.0                      ║
    ║                                                               ║
    ║  Symbolic • Numerical • Visualization • Engineering          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demonstrate_symbolic_calculus(engine):
    """Demonstrate symbolic calculus capabilities"""
    print("\n" + "="*60)
    print("SYMBOLIC CALCULUS DEMONSTRATION")
    print("="*60)
    
    try:
        # Derivatives
        print("\n1. Symbolic Derivatives:")
        print("-" * 30)
        
        examples = [
            ("x**3 + 2*x**2 + x + 1", "x"),
            ("sin(x)*cos(x)", "x"),
            ("exp(x**2)", "x"),
        ]
        
        for expr, var in examples:
            result = engine.symbolic_derivatives.derivative(expr, var)
            print(f"d/d{var}({expr}) = {result}")
        
        # Gradient example
        print("\n2. Gradient Calculation:")
        print("-" * 30)
        expr = "x**2 + 2*y**2 + x*y"
        gradient = engine.symbolic_derivatives.gradient(expr, ["x", "y"])
        print(f"∇({expr}) = {gradient}")
        
        # Integration
        print("\n3. Symbolic Integration:")
        print("-" * 30)
        
        integrals = [
            ("x**2", "x"),
            ("sin(x)", "x"),
            ("1/(1+x**2)", "x"),
        ]
        
        for expr, var in integrals:
            result = engine.symbolic_integrals.indefinite_integral(expr, var)
            print(f"∫({expr}) d{var} = {result}")
        
        # Definite integral
        result = engine.symbolic_integrals.definite_integral("x**2", "x", 0, 1)
        print(f"∫[0 to 1] x² dx = {result}")
        
        # Limits
        print("\n4. Symbolic Limits:")
        print("-" * 30)
        
        limit_examples = [
            ("sin(x)/x", "x", 0),
            ("(1+1/x)**x", "x", "oo"),
            ("(x**2-1)/(x-1)", "x", 1),
        ]
        
        for expr, var, point in limit_examples:
            result = engine.symbolic_limits.limit(expr, var, point)
            print(f"lim[{var}→{point}] ({expr}) = {result}")
        
        # Taylor series
        print("\n5. Taylor Series:")
        print("-" * 30)
        
        series_examples = [
            ("exp(x)", "x", 0, 4),
            ("sin(x)", "x", 0, 5),
            ("cos(x)", "x", 0, 4),
        ]
        
        for expr, var, point, order in series_examples:
            result = engine.taylor_series.taylor_series(expr, var, point, order)
            print(f"{expr} ≈ {result} (around {var}={point})")
        
        print("\n✓ Symbolic calculus demonstration completed successfully!")
        
    except Exception as e:
        print(f"✗ Error in symbolic calculus demonstration: {str(e)}")
        traceback.print_exc()


def demonstrate_numerical_methods(engine):
    """Demonstrate numerical methods capabilities"""
    print("\n" + "="*60)
    print("NUMERICAL METHODS DEMONSTRATION")
    print("="*60)
    
    try:
        # Numerical integration
        print("\n1. Numerical Integration:")
        print("-" * 35)
        
        # Compare different methods
        func = "exp(-x**2)"
        a, b = -2, 2
        
        print(f"Integrating {func} from {a} to {b}:")
        
        # Adaptive quadrature
        result, error = engine.numerical_integration.quad(func, a, b)
        print(f"  Adaptive quadrature: {result:.8f} ± {error:.2e}")
        
        # Simpson's rule
        result_simpson = engine.numerical_integration.simpson(func, a, b, 1000)
        print(f"  Simpson's rule:      {result_simpson:.8f}")
        
        # Monte Carlo
        result_mc, error_mc = engine.numerical_integration.monte_carlo(func, [(a, b)], 10000, seed=42)
        print(f"  Monte Carlo:         {result_mc:.8f} ± {error_mc:.4f}")
        
        # ODE solving
        print("\n2. ODE Solving:")
        print("-" * 20)
        
        # Simple ODE: dy/dt = -y, y(0) = 1
        print("Solving dy/dt = -y, y(0) = 1")
        
        result = engine.ode_solver.solve_ivp_wrapper("-y", (0, 2), 1.0, method='RK45')
        
        if result['success']:
            analytical = np.exp(-2)
            numerical = result['y'][0][-1]
            error = abs(numerical - analytical)
            
            print(f"  Analytical solution at t=2: {analytical:.8f}")
            print(f"  Numerical solution:         {numerical:.8f}")
            print(f"  Error:                      {error:.2e}")
            print(f"  Function evaluations:       {result['nfev']}")
        
        # System of ODEs: Harmonic oscillator
        print("\n  System: Harmonic oscillator dx/dt = y, dy/dt = -x")
        
        system = ["y1", "-y0"]
        result = engine.ode_solver.solve_system(system, (0, np.pi), [1, 0])
        
        if result['success']:
            final_x = result['y'][0][-1]
            final_y = result['y'][1][-1]
            print(f"  Final position: x = {final_x:.6f} (expected: -1)")
            print(f"  Final velocity: y = {final_y:.6f} (expected: 0)")
        
        # Optimization
        print("\n3. Optimization:")
        print("-" * 20)
        
        # Scalar optimization
        print("Minimizing f(x) = x² - 4x + 3")
        result = engine.optimization.minimize_scalar("x**2 - 4*x + 3", bounds=(0, 5))
        
        if result['success']:
            print(f"  Minimum at x = {result['x']:.6f}")
            print(f"  Function value = {result['fun']:.6f}")
            print(f"  Function evaluations = {result['nfev']}")
        
        # Multivariable optimization
        print("\n  Minimizing f(x,y) = (x-1)² + (y-2)²")
        result = engine.optimization.minimize_multivariable("(x0-1)**2 + (x1-2)**2", [0, 0])
        
        if result['success']:
            print(f"  Minimum at ({result['x'][0]:.6f}, {result['x'][1]:.6f})")
            print(f"  Function value = {result['fun']:.6f}")
        
        # Root finding
        print("\n4. Root Finding:")
        print("-" * 20)
        
        # Single root
        print("Finding root of f(x) = x³ - 2x - 5")
        result = engine.root_finding.brent("x**3 - 2*x - 5", 2, 3)
        
        if result['success']:
            print(f"  Root at x = {result['root']:.8f}")
            print(f"  Function value = {result['function_value']:.2e}")
        
        # System of equations
        print("\n  Solving system: x² + y² = 4, x - y = 0")
        equations = ["x0**2 + x1**2 - 4", "x0 - x1"]
        result = engine.root_finding.solve_system(equations, [1, 1])
        
        if result['success']:
            x, y = result['x']
            print(f"  Solution: x = {x:.6f}, y = {y:.6f}")
            print(f"  Residual = {result['residual']:.2e}")
        
        print("\n✓ Numerical methods demonstration completed successfully!")
        
    except Exception as e:
        print(f"✗ Error in numerical methods demonstration: {str(e)}")
        traceback.print_exc()


def demonstrate_visualization(engine):
    """Demonstrate visualization capabilities"""
    print("\n" + "="*60)
    print("VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    try:
        # 2D plotting
        print("\n1. 2D Function Plotting:")
        print("-" * 30)
        
        functions = {
            "Sine": "sin(x)",
            "Cosine": "cos(x)",
            "Exponential": "exp(-x**2)"
        }
        
        result = engine.plotter.plot_multiple_functions(
            functions, 
            x_range=(-2*np.pi, 2*np.pi),
            title="Multiple Function Comparison"
        )
        
        if result['success']:
            print(f"✓ Plotted {result['n_functions']} functions successfully")
        
        # 3D surface plotting
        print("\n2. 3D Surface Plotting:")
        print("-" * 30)
        
        result = engine.plotter.plot_3d_surface(
            "x**2 + y**2",
            x_range=(-3, 3),
            y_range=(-3, 3),
            title="Paraboloid: f(x,y) = x² + y²"
        )
        
        if result['success']:
            print("✓ 3D surface plot created successfully")
        
        # Gradient field
        print("\n3. Gradient Field Visualization:")
        print("-" * 40)
        
        result = engine.plotter.plot_gradient_field(
            "x**2 + y**2",
            x_range=(-2, 2),
            y_range=(-2, 2),
            title="Gradient Field of f(x,y) = x² + y²"
        )
        
        if result['success']:
            print("✓ Gradient field plot created successfully")
        
        print("\n✓ Visualization demonstration completed successfully!")
        print("  Note: Plots are displayed in separate windows")
        
    except Exception as e:
        print(f"✗ Error in visualization demonstration: {str(e)}")
        traceback.print_exc()


def demonstrate_real_world_application(engine):
    """Demonstrate a real-world application"""
    print("\n" + "="*60)
    print("REAL-WORLD APPLICATION: PROJECTILE MOTION ANALYSIS")
    print("="*60)
    
    try:
        # Physical parameters
        h0 = 100  # Initial height (m)
        v0 = 30   # Initial velocity (m/s)
        g = 9.81  # Gravity (m/s²)
        
        print(f"\nProjectile Parameters:")
        print(f"  Initial height: {h0} m")
        print(f"  Initial velocity: {v0} m/s")
        print(f"  Gravity: {g} m/s²")
        
        # Position function
        height_func = f"{h0} + {v0}*t - {g/2}*t**2"
        print(f"\nHeight function: h(t) = {height_func}")
        
        # Symbolic analysis
        print("\n1. Symbolic Analysis:")
        print("-" * 25)
        
        # Velocity (first derivative)
        velocity = engine.symbolic_derivatives.derivative(height_func, "t")
        print(f"  Velocity: v(t) = dh/dt = {velocity}")
        
        # Acceleration (second derivative)
        acceleration = engine.symbolic_derivatives.derivative(height_func, "t", 2)
        print(f"  Acceleration: a(t) = dv/dt = {acceleration}")
        
        # Numerical analysis
        print("\n2. Numerical Analysis:")
        print("-" * 25)
        
        # Time to reach maximum height (v = 0)
        max_time_result = engine.root_finding.brent(str(velocity), 0, 10)
        if max_time_result['success']:
            t_max = max_time_result['root']
            print(f"  Time to max height: {t_max:.3f} seconds")
            
            # Maximum height
            h_max = h0 + v0*t_max - (g/2)*t_max**2
            print(f"  Maximum height: {h_max:.2f} meters")
        
        # Time to hit ground (h = 0)
        ground_time_result = engine.root_finding.brent(height_func, 0, 15)
        if ground_time_result['success']:
            t_ground = ground_time_result['root']
            print(f"  Time to hit ground: {t_ground:.3f} seconds")
            
            # Impact velocity
            v_impact = v0 - g*t_ground
            print(f"  Impact velocity: {v_impact:.2f} m/s")
        
        # Optimization problem
        print("\n3. Optimization Problem:")
        print("-" * 30)
        print("  Find angle for maximum range (simplified)")
        
        # For projectile motion, range R = v²sin(2θ)/g
        # Maximum occurs at θ = 45°
        v_launch = 20  # Launch speed
        optimal_angle = 45  # degrees
        max_range = (v_launch**2) / g
        
        print(f"  Launch speed: {v_launch} m/s")
        print(f"  Optimal angle: {optimal_angle}°")
        print(f"  Maximum range: {max_range:.2f} meters")
        
        # Visualization
        print("\n4. Trajectory Visualization:")
        print("-" * 35)
        
        # Create trajectory plot
        try:
            # Time points
            if 't_ground' in locals():
                t_points = np.linspace(0, t_ground, 100)
                h_points = h0 + v0*t_points - (g/2)*t_points**2
                
                plt.figure(figsize=(10, 6))
                plt.plot(t_points, h_points, 'b-', linewidth=2, label='Height')
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Ground')
                
                if 't_max' in locals():
                    plt.plot(t_max, h_max, 'ro', markersize=8, label=f'Max height ({h_max:.1f}m)')
                
                plt.xlabel('Time (seconds)')
                plt.ylabel('Height (meters)')
                plt.title('Projectile Motion: Height vs Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
                
                print("  ✓ Trajectory plot created")
        except Exception as e:
            print(f"  ✗ Plotting error: {str(e)}")
        
        print("\n✓ Real-world application demonstration completed!")
        
    except Exception as e:
        print(f"✗ Error in real-world application: {str(e)}")
        traceback.print_exc()


def interactive_mode(engine):
    """Interactive mode for exploring Calpyt1"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\nEnter mathematical expressions to explore Calpyt1 capabilities.")
    print("Commands:")
    print("  derivative <expr> <var>     - Compute derivative")
    print("  integral <expr> <var>       - Compute integral") 
    print("  limit <expr> <var> <point>  - Compute limit")
    print("  solve <expr> <var>          - Find roots")
    print("  plot <expr>                 - Plot function")
    print("  help                        - Show this help")
    print("  exit                        - Exit interactive mode")
    print("\nExample: derivative x**2+sin(x) x")
    
    while True:
        try:
            user_input = input("\ncalpyt1> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  derivative <expr> <var>     - e.g., derivative x**2+sin(x) x")
                print("  integral <expr> <var>       - e.g., integral x**2 x")
                print("  limit <expr> <var> <point>  - e.g., limit sin(x)/x x 0")
                print("  solve <expr> <var>          - e.g., solve x**2-4 x")
                print("  plot <expr>                 - e.g., plot sin(x)")
                continue
            
            parts = user_input.split()
            
            if len(parts) < 2:
                print("Invalid command. Type 'help' for usage.")
                continue
            
            command = parts[0].lower()
            
            if command == 'derivative' and len(parts) >= 3:
                expr = parts[1]
                var = parts[2]
                try:
                    result = engine.symbolic_derivatives.derivative(expr, var)
                    print(f"d/d{var}({expr}) = {result}")
                except Exception as e:
                    print(f"Error: {str(e)}")
                    
            elif command == 'integral' and len(parts) >= 3:
                expr = parts[1]
                var = parts[2]
                try:
                    result = engine.symbolic_integrals.indefinite_integral(expr, var)
                    print(f"∫({expr}) d{var} = {result}")
                except Exception as e:
                    print(f"Error: {str(e)}")
                    
            elif command == 'limit' and len(parts) >= 4:
                expr = parts[1]
                var = parts[2]
                point = parts[3]
                try:
                    # Convert point to appropriate type
                    if point.lower() in ['oo', 'inf']:
                        point = float('inf')
                    elif point.lower() in ['-oo', '-inf']:
                        point = float('-inf')
                    else:
                        point = float(point)
                    
                    result = engine.symbolic_limits.limit(expr, var, point)
                    print(f"lim[{var}→{parts[3]}] ({expr}) = {result}")
                except Exception as e:
                    print(f"Error: {str(e)}")
                    
            elif command == 'solve' and len(parts) >= 3:
                expr = parts[1]
                var = parts[2]
                try:
                    # Try to find roots in a reasonable range
                    result = engine.root_finding.find_all_roots(expr, (-10, 10), 100)
                    if result['success'] and result['roots']:
                        print(f"Roots of {expr} = 0: {result['roots']}")
                    else:
                        print("No roots found in range [-10, 10]")
                except Exception as e:
                    print(f"Error: {str(e)}")
                    
            elif command == 'plot' and len(parts) >= 2:
                expr = parts[1]
                try:
                    result = engine.plotter.plot_2d(expr, x_range=(-10, 10))
                    if result['success']:
                        print(f"Plot of {expr} created successfully")
                    else:
                        print("Failed to create plot")
                except Exception as e:
                    print(f"Error: {str(e)}")
                    
            else:
                print("Unknown command or incorrect syntax. Type 'help' for usage.")
                
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")


def run_comprehensive_demo():
    """Run the complete Calpyt1 demonstration"""
    print_banner()
    
    print("\nInitializing Calpyt1 Engine...")
    try:
        engine = CalcEngine(precision=15)
        print("✓ Engine initialized successfully!")
        
        # Display engine info
        info = engine.get_info()
        print(f"\nEngine Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"✗ Failed to initialize engine: {str(e)}")
        return 1
    
    # Run demonstrations
    try:
        demonstrate_symbolic_calculus(engine)
        demonstrate_numerical_methods(engine)
        demonstrate_visualization(engine)
        demonstrate_real_world_application(engine)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\n🎉 All demonstrations completed successfully!")
        print("\nNext steps:")
        print("  • Explore the Jupyter notebooks in notebooks/")
        print("  • Try the CLI interface: calpyt1 --help")
        print("  • Run the test suite: pytest tests/")
        print("  • Check out examples/ for more applications")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Demonstration failed: {str(e)}")
        traceback.print_exc()
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Calpyt1 Mathematical Computing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full demonstration
  python main.py --interactive      # Interactive mode
  python main.py --symbolic-only    # Only symbolic calculus demo
  python main.py --numerical-only   # Only numerical methods demo
  python main.py --viz-only         # Only visualization demo
  python main.py --version          # Show version information
        """
    )
    
    parser.add_argument('--demo', '-d', action='store_true',
                       help='Run comprehensive demo')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--symbolic-only', action='store_true',
                       help='Run only symbolic calculus demonstration')
    parser.add_argument('--numerical-only', action='store_true',
                       help='Run only numerical methods demonstration')
    parser.add_argument('--viz-only', action='store_true',
                       help='Run only visualization demonstration')
    parser.add_argument('--version', action='version', version='Calpyt1 v0.1.0')
    parser.add_argument('--no-banner', action='store_true',
                       help='Suppress banner display')
    
    args = parser.parse_args()
    
    # Show banner unless suppressed
    if not args.no_banner:
        print_banner()
    
    # Initialize engine
    try:
        engine = CalcEngine()
        if not args.no_banner:
            print("✓ Calpyt1 engine initialized successfully!")
    except Exception as e:
        print(f"✗ Failed to initialize Calpyt1 engine: {str(e)}")
        return 1
    
    # Handle different modes
    try:
        if args.demo:
            return run_comprehensive_demo()
        elif args.interactive:
            interactive_mode(engine)
        elif args.symbolic_only:
            demonstrate_symbolic_calculus(engine)
        elif args.numerical_only:
            demonstrate_numerical_methods(engine)
        elif args.viz_only:
            demonstrate_visualization(engine)
        else:
            # Run full demonstration
            return run_comprehensive_demo()
            
        return 0
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 0
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
