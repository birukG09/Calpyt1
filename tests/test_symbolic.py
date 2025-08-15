"""
Unit tests for symbolic calculus module
"""

import pytest
import numpy as np
import sympy as sp
from calpyt1 import CalcEngine


class TestSymbolicDerivatives:
    """Test cases for symbolic derivatives"""
    
    def setup_method(self):
        """Setup test fixture"""
        self.engine = CalcEngine()
    
    def test_basic_derivatives(self):
        """Test basic derivative calculations"""
        # Test polynomial derivative
        result = self.engine.symbolic_derivatives.derivative("x**2", "x")
        expected = sp.sympify("2*x")
        assert result.equals(expected)
        
        # Test trigonometric derivative
        result = self.engine.symbolic_derivatives.derivative("sin(x)", "x")
        expected = sp.cos(sp.Symbol('x'))
        assert result.equals(expected)
        
        # Test exponential derivative
        result = self.engine.symbolic_derivatives.derivative("exp(x)", "x")
        expected = sp.exp(sp.Symbol('x'))
        assert result.equals(expected)
    
    def test_higher_order_derivatives(self):
        """Test higher order derivatives"""
        # Second derivative of x^4
        result = self.engine.symbolic_derivatives.derivative("x**4", "x", 2)
        expected = sp.sympify("12*x**2")
        assert result.equals(expected)
        
        # Third derivative of x^3
        result = self.engine.symbolic_derivatives.derivative("x**3", "x", 3)
        expected = sp.sympify("6")
        assert result.equals(expected)
    
    def test_partial_derivatives(self):
        """Test partial derivatives"""
        # Partial derivative with respect to x
        result = self.engine.symbolic_derivatives.derivative("x**2 + y**2", "x")
        expected = sp.sympify("2*x")
        assert result.equals(expected)
        
        # Partial derivative with respect to y
        result = self.engine.symbolic_derivatives.derivative("x**2 + y**2", "y")
        expected = sp.sympify("2*y")
        assert result.equals(expected)
    
    def test_gradient(self):
        """Test gradient calculation"""
        expr = "x**2 + y**2"
        gradient = self.engine.symbolic_derivatives.gradient(expr, ["x", "y"])
        
        expected_x = sp.sympify("2*x")
        expected_y = sp.sympify("2*y")
        
        assert gradient[0].equals(expected_x)
        assert gradient[1].equals(expected_y)
    
    def test_jacobian(self):
        """Test Jacobian matrix calculation"""
        functions = ["x**2 + y", "x + y**2"]
        variables = ["x", "y"]
        
        jacobian = self.engine.symbolic_derivatives.jacobian(functions, variables)
        
        # Check dimensions
        assert jacobian.shape == (2, 2)
        
        # Check individual elements
        assert jacobian[0, 0].equals(sp.sympify("2*x"))
        assert jacobian[0, 1].equals(sp.sympify("1"))
        assert jacobian[1, 0].equals(sp.sympify("1"))
        assert jacobian[1, 1].equals(sp.sympify("2*y"))
    
    def test_hessian(self):
        """Test Hessian matrix calculation"""
        expr = "x**2 + y**2 + x*y"
        variables = ["x", "y"]
        
        hessian = self.engine.symbolic_derivatives.hessian(expr, variables)
        
        # Check dimensions
        assert hessian.shape == (2, 2)
        
        # Check elements
        assert hessian[0, 0].equals(sp.sympify("2"))
        assert hessian[0, 1].equals(sp.sympify("1"))
        assert hessian[1, 0].equals(sp.sympify("1"))
        assert hessian[1, 1].equals(sp.sympify("2"))
    
    def test_chain_rule(self):
        """Test chain rule application"""
        outer = "u**2"
        inner = "sin(x)"
        
        result = self.engine.symbolic_derivatives.chain_rule(outer, inner, "x")
        expected = sp.sympify("2*sin(x)*cos(x)")
        
        assert result.equals(expected)
    
    def test_implicit_derivative(self):
        """Test implicit differentiation"""
        # Circle equation: x^2 + y^2 = 1
        equation = "x**2 + y**2 - 1"
        
        result = self.engine.symbolic_derivatives.implicit_derivative(equation, "y", "x")
        expected = sp.sympify("-x/y")
        
        assert result.equals(expected)
    
    def test_critical_points(self):
        """Test critical point finding"""
        expr = "x**2 - 4*x + 3"
        
        critical_points = self.engine.symbolic_derivatives.get_critical_points(expr, ["x"])
        
        assert len(critical_points) == 1
        assert critical_points[0]['x'] == 2


class TestSymbolicIntegrals:
    """Test cases for symbolic integration"""
    
    def setup_method(self):
        """Setup test fixture"""
        self.engine = CalcEngine()
    
    def test_indefinite_integrals(self):
        """Test indefinite integration"""
        # Integral of x^2
        result = self.engine.symbolic_integrals.indefinite_integral("x**2", "x")
        expected = sp.sympify("x**3/3")
        # Check if derivative of result equals original function
        assert sp.diff(result, sp.Symbol('x')).equals(sp.sympify("x**2"))
        
        # Integral of sin(x)
        result = self.engine.symbolic_integrals.indefinite_integral("sin(x)", "x")
        assert sp.diff(result, sp.Symbol('x')).equals(sp.sin(sp.Symbol('x')))
    
    def test_definite_integrals(self):
        """Test definite integration"""
        # Integral of x^2 from 0 to 1
        result = self.engine.symbolic_integrals.definite_integral("x**2", "x", 0, 1)
        expected = sp.Rational(1, 3)
        assert result.equals(expected)
        
        # Integral of sin(x) from 0 to pi
        result = self.engine.symbolic_integrals.definite_integral("sin(x)", "x", 0, sp.pi)
        expected = 2
        assert abs(result - expected) < 1e-10
    
    def test_integration_by_parts(self):
        """Test integration by parts"""
        u = "x"
        dv = "exp(x)"
        
        result = self.engine.symbolic_integrals.integration_by_parts(u, dv, "x")
        
        # The result should be x*exp(x) - exp(x) (plus constant)
        # Check by differentiating
        derivative = sp.diff(result, sp.Symbol('x'))
        expected = sp.sympify("x*exp(x)")
        assert derivative.equals(expected)
    
    def test_improper_integrals(self):
        """Test improper integrals"""
        # Integral of exp(-x) from 0 to infinity
        result = self.engine.symbolic_integrals.improper_integral("exp(-x)", "x", 0, sp.oo)
        expected = 1
        assert result.equals(expected)
    
    def test_integration_techniques(self):
        """Test integration technique suggestions"""
        # Test polynomial
        techniques = self.engine.symbolic_integrals.get_integration_techniques("x**2", "x")
        assert techniques['direct_integration'] == True
        assert "Polynomial" in str(techniques['suggestions'])
        
        # Test product (integration by parts candidate)
        techniques = self.engine.symbolic_integrals.get_integration_techniques("x*sin(x)", "x")
        assert techniques['integration_by_parts'] == True


class TestSymbolicLimits:
    """Test cases for symbolic limits"""
    
    def setup_method(self):
        """Setup test fixture"""
        self.engine = CalcEngine()
    
    def test_basic_limits(self):
        """Test basic limit calculations"""
        # Classic limit: sin(x)/x as x -> 0
        result = self.engine.symbolic_limits.limit("sin(x)/x", "x", 0)
        expected = 1
        assert result == expected
        
        # Limit of polynomial
        result = self.engine.symbolic_limits.limit("x**2 + 1", "x", 2)
        expected = 5
        assert result == expected
    
    def test_limits_at_infinity(self):
        """Test limits at infinity"""
        result = self.engine.symbolic_limits.limit_at_infinity("1/x", "x", "+")
        expected = 0
        assert result == expected
        
        result = self.engine.symbolic_limits.limit_at_infinity("x**2", "x", "+")
        expected = sp.oo
        assert result == expected
    
    def test_one_sided_limits(self):
        """Test one-sided limits"""
        # Left limit of 1/x at x=0
        result = self.engine.symbolic_limits.left_limit("1/x", "x", 0)
        expected = -sp.oo
        assert result == expected
        
        # Right limit of 1/x at x=0
        result = self.engine.symbolic_limits.right_limit("1/x", "x", 0)
        expected = sp.oo
        assert result == expected
    
    def test_lhopital_rule(self):
        """Test L'Hôpital's rule"""
        # 0/0 form: sin(x)/x
        result = self.engine.symbolic_limits.l_hopital_rule("sin(x)", "x", "x", 0)
        expected = 1
        assert result == expected
        
        # ∞/∞ form: x/exp(x)
        result = self.engine.symbolic_limits.l_hopital_rule("x", "exp(x)", "x", sp.oo)
        expected = 0
        assert result == expected
    
    def test_continuity_check(self):
        """Test continuity analysis"""
        # Continuous function
        continuity = self.engine.symbolic_limits.continuity_check("x**2", "x", 1)
        assert continuity['continuous'] == True
        assert continuity['type'] == "continuous"
        
        # Discontinuous function
        continuity = self.engine.symbolic_limits.continuity_check("1/x", "x", 0)
        assert continuity['continuous'] == False
        assert continuity['type'] == "infinite_discontinuity"


class TestTaylorSeries:
    """Test cases for Taylor series"""
    
    def setup_method(self):
        """Setup test fixture"""
        self.engine = CalcEngine()
    
    def test_taylor_series(self):
        """Test Taylor series expansion"""
        # Taylor series of exp(x) around 0
        result = self.engine.taylor_series.taylor_series("exp(x)", "x", 0, 3)
        expected = sp.sympify("1 + x + x**2/2 + x**3/6")
        assert result.equals(expected)
        
        # Taylor series of sin(x) around 0
        result = self.engine.taylor_series.taylor_series("sin(x)", "x", 0, 5)
        expected = sp.sympify("x - x**3/6 + x**5/120")
        assert result.equals(expected)
    
    def test_maclaurin_series(self):
        """Test Maclaurin series (Taylor around 0)"""
        result = self.engine.taylor_series.maclaurin_series("cos(x)", "x", 4)
        expected = sp.sympify("1 - x**2/2 + x**4/24")
        assert result.equals(expected)
    
    def test_series_convergence(self):
        """Test series convergence tests"""
        # Ratio test for geometric series
        convergence = self.engine.taylor_series.series_convergence_test("1/2**n", "n", "ratio")
        assert convergence['convergent'] == True
        assert "converges" in convergence['conclusion']
        
        # Ratio test for divergent series
        convergence = self.engine.taylor_series.series_convergence_test("2**n", "n", "ratio")
        assert convergence['convergent'] == False
        assert "diverges" in convergence['conclusion']
    
    def test_power_series(self):
        """Test power series creation"""
        coefficients = [1, 1, 1, 1]  # 1 + x + x^2 + x^3
        result = self.engine.taylor_series.power_series(coefficients, "x", 0, 4)
        expected = sp.sympify("1 + x + x**2 + x**3")
        assert result.equals(expected)
    
    def test_fourier_coefficients(self):
        """Test Fourier series coefficients"""
        # Simple constant function
        coeffs = self.engine.taylor_series.fourier_series_coefficients("1", "t", 2*sp.pi, 2)
        
        # For constant function, a0 should be 2, an and bn should be 0
        assert coeffs['a0'] == 2
        assert all(an == 0 for an in coeffs['an'])
        assert all(bn == 0 for bn in coeffs['bn'])


class TestSymbolicIntegration:
    """Integration tests for symbolic calculus components"""
    
    def setup_method(self):
        """Setup test fixture"""
        self.engine = CalcEngine()
    
    def test_calculus_fundamental_theorem(self):
        """Test fundamental theorem of calculus"""
        # d/dx ∫f(t)dt = f(x)
        expr = "x**3"
        
        # Integrate then differentiate
        integral = self.engine.symbolic_integrals.indefinite_integral(expr, "x")
        derivative = self.engine.symbolic_derivatives.derivative(integral, "x")
        
        # Should get back original expression (up to constant)
        original = sp.sympify(expr)
        assert derivative.equals(original)
    
    def test_derivative_integral_relationship(self):
        """Test relationship between derivatives and integrals"""
        # Test that ∫f'(x)dx = f(x) + C
        original_func = "sin(x)"
        
        # Take derivative
        derivative = self.engine.symbolic_derivatives.derivative(original_func, "x")
        
        # Integrate the derivative
        integral = self.engine.symbolic_integrals.indefinite_integral(str(derivative), "x")
        
        # Check that we get back to original function (up to constant)
        original = sp.sympify(original_func)
        diff = sp.simplify(integral - original)
        
        # Difference should be a constant (no x dependence)
        assert diff.is_constant()
    
    def test_limit_derivative_relationship(self):
        """Test limit definition of derivative"""
        func = "x**2"
        x = sp.Symbol('x')
        h = sp.Symbol('h')
        
        # Limit definition: f'(x) = lim(h->0) [f(x+h) - f(x)]/h
        f_x_plus_h = sp.sympify(func).subs(x, x + h)
        f_x = sp.sympify(func)
        
        difference_quotient = (f_x_plus_h - f_x) / h
        limit_result = self.engine.symbolic_limits.limit(str(difference_quotient), "h", 0)
        
        # Compare with direct derivative
        direct_derivative = self.engine.symbolic_derivatives.derivative(func, "x")
        
        assert limit_result.equals(direct_derivative)
    
    def test_series_limit_relationship(self):
        """Test relationship between series and limits"""
        # Test that Taylor series converges to function at center
        func = "exp(x)"
        center = 0
        
        # High order Taylor series
        series = self.engine.taylor_series.taylor_series(func, "x", center, 10)
        
        # Evaluate both at center point
        func_at_center = sp.sympify(func).subs(sp.Symbol('x'), center)
        series_at_center = series.subs(sp.Symbol('x'), center)
        
        assert abs(func_at_center - series_at_center) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
