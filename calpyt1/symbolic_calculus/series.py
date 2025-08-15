"""
Series expansion module for Calpyt1 framework
"""

import sympy as sp
from typing import Union, Optional, List, Dict, Any
from ..core.base import BaseModule


class TaylorSeries(BaseModule):
    """
    Handles Taylor series and other series expansions
    """
    
    def __init__(self, engine):
        """Initialize Taylor series module"""
        super().__init__(engine)
        self.logger.info("TaylorSeries module initialized")
    
    def taylor_series(self, expr: Union[str, sp.Basic], 
                     var: Union[str, sp.Symbol],
                     point: Union[float, int, sp.Basic] = 0,
                     order: int = 5) -> sp.Basic:
        """
        Compute Taylor series expansion around a point
        
        Args:
            expr: Expression to expand
            var: Variable to expand around
            point: Point of expansion (default: 0 for Maclaurin series)
            order: Order of expansion
            
        Returns:
            Taylor series expansion
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        try:
            series = sp.series(expr, var, point, n=order+1)
            # Remove the O() term for cleaner output
            series = series.removeO()
            
            self.logger.info(f"Computed Taylor series of order {order} around {point}")
            return series
        except Exception as e:
            raise RuntimeError(f"Failed to compute Taylor series: {str(e)}")
    
    def maclaurin_series(self, expr: Union[str, sp.Basic], 
                        var: Union[str, sp.Symbol],
                        order: int = 5) -> sp.Basic:
        """
        Compute Maclaurin series (Taylor series around 0)
        
        Args:
            expr: Expression to expand
            var: Variable to expand around
            order: Order of expansion
            
        Returns:
            Maclaurin series expansion
        """
        return self.taylor_series(expr, var, point=0, order=order)
    
    def laurent_series(self, expr: Union[str, sp.Basic],
                      var: Union[str, sp.Symbol],
                      point: Union[float, int, sp.Basic] = 0,
                      order: int = 5) -> sp.Basic:
        """
        Compute Laurent series expansion (includes negative powers)
        
        Args:
            expr: Expression to expand
            var: Variable to expand around
            point: Point of expansion
            order: Order of expansion
            
        Returns:
            Laurent series expansion
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        try:
            # Use SymPy's series with negative order terms
            series = sp.series(expr, var, point, n=order+1)
            
            self.logger.info(f"Computed Laurent series of order {order} around {point}")
            return series
        except Exception as e:
            raise RuntimeError(f"Failed to compute Laurent series: {str(e)}")
    
    def power_series(self, coefficients: List[Union[float, int, sp.Basic]],
                    var: Union[str, sp.Symbol],
                    center: Union[float, int, sp.Basic] = 0,
                    max_terms: int = 10) -> sp.Basic:
        """
        Create power series from coefficients
        
        Args:
            coefficients: List of coefficients [a0, a1, a2, ...]
            var: Variable
            center: Center point of series
            max_terms: Maximum number of terms to include
            
        Returns:
            Power series expression
        """
        var = self.validate_variable(var)
        
        try:
            terms_to_use = min(len(coefficients), max_terms)
            
            series = sum(coefficients[n] * (var - center)**n 
                        for n in range(terms_to_use))
            
            self.logger.info(f"Created power series with {terms_to_use} terms")
            return series
        except Exception as e:
            raise RuntimeError(f"Failed to create power series: {str(e)}")
    
    def fourier_series_coefficients(self, expr: Union[str, sp.Basic],
                                   var: Union[str, sp.Symbol],
                                   period: Union[float, int] = 2*sp.pi,
                                   n_terms: int = 5) -> Dict[str, List]:
        """
        Compute Fourier series coefficients
        
        Args:
            expr: Periodic function to expand
            var: Variable (typically time t or angle θ)
            period: Period of the function
            n_terms: Number of harmonic terms
            
        Returns:
            Dictionary with a0, an, bn coefficients
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        try:
            # Fourier series: f(x) = a0/2 + Σ[an*cos(nωx) + bn*sin(nωx)]
            # where ω = 2π/period
            
            omega = 2 * sp.pi / period
            
            # Compute a0 coefficient
            a0 = (2/period) * sp.integrate(expr, (var, -period/2, period/2))
            
            # Compute an coefficients (cosine terms)
            an_coeffs = []
            for n in range(1, n_terms + 1):
                an = (2/period) * sp.integrate(expr * sp.cos(n * omega * var), 
                                              (var, -period/2, period/2))
                an_coeffs.append(an)
            
            # Compute bn coefficients (sine terms)
            bn_coeffs = []
            for n in range(1, n_terms + 1):
                bn = (2/period) * sp.integrate(expr * sp.sin(n * omega * var), 
                                              (var, -period/2, period/2))
                bn_coeffs.append(bn)
            
            result = {
                "a0": a0,
                "an": an_coeffs,
                "bn": bn_coeffs,
                "period": period,
                "omega": omega
            }
            
            self.logger.info(f"Computed Fourier coefficients with {n_terms} harmonics")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute Fourier coefficients: {str(e)}")
    
    def fourier_series(self, expr: Union[str, sp.Basic],
                      var: Union[str, sp.Symbol],
                      period: Union[float, int] = 2*sp.pi,
                      n_terms: int = 5) -> sp.Basic:
        """
        Compute Fourier series expansion
        
        Args:
            expr: Periodic function to expand
            var: Variable
            period: Period of function
            n_terms: Number of harmonic terms
            
        Returns:
            Fourier series approximation
        """
        coeffs = self.fourier_series_coefficients(expr, var, period, n_terms)
        
        try:
            # Build the series
            series = coeffs["a0"] / 2
            
            omega = coeffs["omega"]
            
            for n in range(n_terms):
                series += coeffs["an"][n] * sp.cos((n+1) * omega * var)
                series += coeffs["bn"][n] * sp.sin((n+1) * omega * var)
            
            self.logger.info(f"Built Fourier series with {n_terms} harmonics")
            return series
        except Exception as e:
            raise RuntimeError(f"Failed to build Fourier series: {str(e)}")
    
    def series_convergence_test(self, series_term: Union[str, sp.Basic],
                               var: Union[str, sp.Symbol],
                               test_type: str = "ratio") -> Dict[str, Any]:
        """
        Test convergence of infinite series
        
        Args:
            series_term: General term of the series a_n
            var: Series index variable (usually n)
            test_type: Type of test ('ratio', 'root', 'integral', 'comparison')
            
        Returns:
            Dictionary with convergence test results
        """
        series_term = self.validate_expression(series_term)
        var = self.validate_variable(var)
        
        result = {
            "test_type": test_type,
            "convergent": None,
            "test_value": None,
            "conclusion": ""
        }
        
        try:
            if test_type.lower() == "ratio":
                # Ratio test: lim |a_{n+1}/a_n|
                next_term = series_term.subs(var, var + 1)
                ratio = sp.Abs(next_term / series_term)
                limit_ratio = sp.limit(ratio, var, sp.oo)
                
                result["test_value"] = limit_ratio
                
                if limit_ratio < 1:
                    result["convergent"] = True
                    result["conclusion"] = "Series converges (ratio test)"
                elif limit_ratio > 1:
                    result["convergent"] = False
                    result["conclusion"] = "Series diverges (ratio test)"
                else:
                    result["conclusion"] = "Ratio test inconclusive"
                    
            elif test_type.lower() == "root":
                # Root test: lim |a_n|^(1/n)
                root_expr = sp.Abs(series_term)**(1/var)
                limit_root = sp.limit(root_expr, var, sp.oo)
                
                result["test_value"] = limit_root
                
                if limit_root < 1:
                    result["convergent"] = True
                    result["conclusion"] = "Series converges (root test)"
                elif limit_root > 1:
                    result["convergent"] = False
                    result["conclusion"] = "Series diverges (root test)"
                else:
                    result["conclusion"] = "Root test inconclusive"
                    
            elif test_type.lower() == "integral":
                # Integral test (for positive decreasing functions)
                # Convert series variable to continuous variable
                continuous_func = series_term
                integral_result = sp.integrate(continuous_func, (var, 1, sp.oo))
                
                result["test_value"] = integral_result
                
                if integral_result.is_finite:
                    result["convergent"] = True
                    result["conclusion"] = "Series converges (integral test)"
                else:
                    result["convergent"] = False
                    result["conclusion"] = "Series diverges (integral test)"
                    
            else:
                result["conclusion"] = f"Test type '{test_type}' not implemented"
            
            self.logger.info(f"Applied {test_type} convergence test")
            return result
            
        except Exception as e:
            result["conclusion"] = f"Error in convergence test: {str(e)}"
            return result
    
    def radius_of_convergence(self, coefficients: List[Union[float, int, sp.Basic]],
                             var: Union[str, sp.Symbol]) -> Dict[str, Any]:
        """
        Find radius of convergence for power series
        
        Args:
            coefficients: Series coefficients [a0, a1, a2, ...]
            var: Variable
            
        Returns:
            Dictionary with radius and interval of convergence
        """
        var = self.validate_variable(var)
        
        result = {
            "radius": None,
            "interval": None,
            "method": ""
        }
        
        try:
            # Use ratio test for radius of convergence
            # R = lim |a_n / a_{n+1}|
            
            if len(coefficients) < 2:
                result["radius"] = sp.oo
                result["method"] = "Polynomial (finite terms)"
                return result
            
            # Find pattern in ratios
            ratios = []
            for i in range(len(coefficients) - 1):
                if coefficients[i+1] != 0:
                    ratio = sp.Abs(coefficients[i] / coefficients[i+1])
                    ratios.append(ratio)
            
            if ratios:
                # Take limit of ratios (assuming they approach a limit)
                # For general case, we approximate with the last few ratios
                if len(ratios) >= 3:
                    # Use the last ratio as approximation
                    radius = ratios[-1]
                else:
                    radius = ratios[0]
                
                result["radius"] = radius
                result["method"] = "Ratio test approximation"
                
                # Interval of convergence: (-R, R) typically
                result["interval"] = f"({-radius}, {radius})"
            else:
                result["radius"] = sp.oo
                result["method"] = "All coefficients zero except possibly first"
            
            self.logger.info("Computed radius of convergence")
            return result
            
        except Exception as e:
            result["method"] = f"Error: {str(e)}"
            return result
    
    def series_manipulation(self, series1: Union[str, sp.Basic],
                           series2: Union[str, sp.Basic],
                           operation: str,
                           var: Union[str, sp.Symbol],
                           order: int = 5) -> sp.Basic:
        """
        Perform operations on power series
        
        Args:
            series1: First series
            series2: Second series (for binary operations)
            operation: 'add', 'subtract', 'multiply', 'divide', 'compose'
            var: Variable
            order: Order to truncate result
            
        Returns:
            Result of series operation
        """
        series1 = self.validate_expression(series1)
        var = self.validate_variable(var)
        
        if operation in ['add', 'subtract', 'multiply', 'divide', 'compose']:
            series2 = self.validate_expression(series2)
        
        try:
            if operation == "add":
                result = sp.series(series1 + series2, var, n=order+1).removeO()
            elif operation == "subtract":
                result = sp.series(series1 - series2, var, n=order+1).removeO()
            elif operation == "multiply":
                result = sp.series(series1 * series2, var, n=order+1).removeO()
            elif operation == "divide":
                result = sp.series(series1 / series2, var, n=order+1).removeO()
            elif operation == "compose":
                # Composition: series1(series2(var))
                result = sp.series(series1.subs(var, series2), var, n=order+1).removeO()
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            self.logger.info(f"Performed series {operation}")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to perform series operation: {str(e)}")
    
    def asymptotic_expansion(self, expr: Union[str, sp.Basic],
                            var: Union[str, sp.Symbol],
                            point: Union[float, int, sp.Basic] = sp.oo,
                            order: int = 3) -> sp.Basic:
        """
        Compute asymptotic expansion for large values
        
        Args:
            expr: Expression to expand
            var: Variable
            point: Point to expand around (usually ∞)
            order: Order of expansion
            
        Returns:
            Asymptotic expansion
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        if point == float('inf'):
            point = sp.oo
        elif point == float('-inf'):
            point = -sp.oo
        
        try:
            # For expansion around infinity, often substitute var = 1/t and expand around t=0
            if point == sp.oo:
                t = self.engine.create_symbol('t_asymp')
                substituted = expr.subs(var, 1/t)
                expansion = sp.series(substituted, t, 0, n=order+1)
                # Substitute back
                result = expansion.subs(t, 1/var).removeO()
            else:
                result = sp.series(expr, var, point, n=order+1).removeO()
            
            self.logger.info(f"Computed asymptotic expansion around {point}")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute asymptotic expansion: {str(e)}")
