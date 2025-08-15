"""
Symbolic calculus examples for Calpyt1 framework
"""

import numpy as np
from calpyt1 import CalcEngine
import sympy as sp


def derivatives_examples():
    """Examples of symbolic derivatives"""
    print("=" * 60)
    print("SYMBOLIC DERIVATIVES EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Basic derivatives
    print("\n1. Basic Derivatives:")
    print("-" * 30)
    
    examples = [
        ("x**2", "x"),
        ("sin(x)", "x"),
        ("exp(x)", "x"),
        ("log(x)", "x"),
        ("x**3 + 2*x**2 + x + 1", "x")
    ]
    
    for expr, var in examples:
        result = engine.symbolic_derivatives.derivative(expr, var)
        print(f"d/d{var}({expr}) = {result}")
    
    # Higher order derivatives
    print("\n2. Higher Order Derivatives:")
    print("-" * 30)
    
    expr = "x**4 - 3*x**3 + 2*x**2 - x + 5"
    for order in range(1, 5):
        result = engine.symbolic_derivatives.derivative(expr, "x", order)
        print(f"d^{order}/dx^{order}({expr}) = {result}")
    
    # Partial derivatives
    print("\n3. Partial Derivatives:")
    print("-" * 30)
    
    expr = "x**2 + y**2 + x*y"
    dx = engine.symbolic_derivatives.derivative(expr, "x")
    dy = engine.symbolic_derivatives.derivative(expr, "y")
    print(f"∂/∂x({expr}) = {dx}")
    print(f"∂/∂y({expr}) = {dy}")
    
    # Gradient
    print("\n4. Gradient:")
    print("-" * 30)
    
    expr = "x**2 + 2*y**2 + 3*z**2"
    gradient = engine.symbolic_derivatives.gradient(expr, ["x", "y", "z"])
    print(f"∇({expr}) = {gradient}")
    
    # Chain rule
    print("\n5. Chain Rule:")
    print("-" * 30)
    
    outer = "u**2"
    inner = "sin(x)"
    result = engine.symbolic_derivatives.chain_rule(outer, inner, "x")
    print(f"Chain rule for ({outer}) ∘ ({inner}): {result}")


def integrals_examples():
    """Examples of symbolic integration"""
    print("\n" + "=" * 60)
    print("SYMBOLIC INTEGRALS EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Indefinite integrals
    print("\n1. Indefinite Integrals:")
    print("-" * 30)
    
    examples = [
        ("x**2", "x"),
        ("sin(x)", "x"),
        ("exp(x)", "x"),
        ("1/x", "x"),
        ("x*exp(x)", "x")
    ]
    
    for expr, var in examples:
        result = engine.symbolic_integrals.indefinite_integral(expr, var)
        print(f"∫({expr}) d{var} = {result}")
    
    # Definite integrals
    print("\n2. Definite Integrals:")
    print("-" * 30)
    
    examples = [
        ("x**2", "x", 0, 1),
        ("sin(x)", "x", 0, sp.pi),
        ("exp(-x)", "x", 0, sp.oo),
        ("1/(1+x**2)", "x", -sp.oo, sp.oo)
    ]
    
    for expr, var, a, b in examples:
        result = engine.symbolic_integrals.definite_integral(expr, var, a, b)
        print(f"∫[{a} to {b}] ({expr}) d{var} = {result}")
    
    # Integration by parts
    print("\n3. Integration by Parts:")
    print("-" * 30)
    
    u = "x"
    dv = "exp(x)"
    result = engine.symbolic_integrals.integration_by_parts(u, dv, "x")
    print(f"∫({u})({dv}) dx = {result}")


def limits_examples():
    """Examples of symbolic limits"""
    print("\n" + "=" * 60)
    print("SYMBOLIC LIMITS EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Basic limits
    print("\n1. Basic Limits:")
    print("-" * 30)
    
    examples = [
        ("sin(x)/x", "x", 0),
        ("(1+1/x)**x", "x", sp.oo),
        ("(x**2-1)/(x-1)", "x", 1),
        ("exp(x)", "x", sp.oo),
        ("1/x", "x", 0)
    ]
    
    for expr, var, point in examples:
        result = engine.symbolic_limits.limit(expr, var, point)
        print(f"lim[{var}→{point}] ({expr}) = {result}")
    
    # One-sided limits
    print("\n2. One-sided Limits:")
    print("-" * 30)
    
    expr = "1/x"
    left_limit = engine.symbolic_limits.left_limit(expr, "x", 0)
    right_limit = engine.symbolic_limits.right_limit(expr, "x", 0)
    print(f"lim[x→0⁻] ({expr}) = {left_limit}")
    print(f"lim[x→0⁺] ({expr}) = {right_limit}")
    
    # L'Hôpital's rule
    print("\n3. L'Hôpital's Rule:")
    print("-" * 30)
    
    numerator = "sin(x)"
    denominator = "x"
    result = engine.symbolic_limits.l_hopital_rule(numerator, denominator, "x", 0)
    print(f"L'Hôpital: lim[x→0] ({numerator})/({denominator}) = {result}")


def series_examples():
    """Examples of series expansions"""
    print("\n" + "=" * 60)
    print("SERIES EXPANSIONS EXAMPLES")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Taylor series
    print("\n1. Taylor Series:")
    print("-" * 30)
    
    examples = [
        ("exp(x)", "x", 0, 5),
        ("sin(x)", "x", 0, 7),
        ("cos(x)", "x", 0, 6),
        ("log(1+x)", "x", 0, 5),
        ("1/(1-x)", "x", 0, 5)
    ]
    
    for expr, var, point, order in examples:
        result = engine.taylor_series.taylor_series(expr, var, point, order)
        print(f"Taylor series of {expr} around {point} (order {order}):")
        print(f"  {result}")
    
    # Fourier series coefficients
    print("\n2. Fourier Series Coefficients:")
    print("-" * 30)
    
    # Square wave example
    expr = "1"  # Simple constant function for demo
    coeffs = engine.taylor_series.fourier_series_coefficients(expr, "t", 2*sp.pi, 3)
    print(f"Fourier coefficients for {expr}:")
    print(f"  a0 = {coeffs['a0']}")
    print(f"  an = {coeffs['an']}")
    print(f"  bn = {coeffs['bn']}")


def comprehensive_example():
    """Comprehensive example combining multiple operations"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EXAMPLE")
    print("=" * 60)
    
    engine = CalcEngine()
    
    # Define a function
    f = "x**3 - 6*x**2 + 9*x + 1"
    print(f"\nAnalyzing function: f(x) = {f}")
    
    # First derivative
    f_prime = engine.symbolic_derivatives.derivative(f, "x")
    print(f"f'(x) = {f_prime}")
    
    # Second derivative
    f_double_prime = engine.symbolic_derivatives.derivative(f, "x", 2)
    print(f"f''(x) = {f_double_prime}")
    
    # Critical points
    critical_points = engine.symbolic_derivatives.get_critical_points(f, ["x"])
    print(f"Critical points: {critical_points}")
    
    # Indefinite integral
    integral = engine.symbolic_integrals.indefinite_integral(f, "x")
    print(f"∫f(x)dx = {integral}")
    
    # Definite integral
    definite = engine.symbolic_integrals.definite_integral(f, "x", 0, 3)
    print(f"∫[0 to 3] f(x)dx = {definite}")
    
    # Taylor series around x=1
    taylor = engine.taylor_series.taylor_series(f, "x", 1, 4)
    print(f"Taylor series around x=1: {taylor}")


def main():
    """Run all symbolic examples"""
    print("CALPYT1 SYMBOLIC CALCULUS EXAMPLES")
    print("=" * 80)
    
    try:
        derivatives_examples()
        integrals_examples()
        limits_examples()
        series_examples()
        comprehensive_example()
        
        print("\n" + "=" * 80)
        print("ALL SYMBOLIC EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError in symbolic examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
