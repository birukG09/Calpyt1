"""
Symbolic limits module for Calpyt1 framework
"""

import sympy as sp
from typing import Union, Optional, List, Dict, Any
from ..core.base import BaseModule


class SymbolicLimits(BaseModule):
    """
    Handles symbolic limit calculations
    """
    
    def __init__(self, engine):
        """Initialize symbolic limits module"""
        super().__init__(engine)
        self.logger.info("SymbolicLimits module initialized")
    
    def limit(self, expr: Union[str, sp.Basic], 
             var: Union[str, sp.Symbol],
             point: Union[float, int, sp.Basic],
             direction: str = "+-") -> sp.Basic:
        """
        Compute limit of expression as variable approaches a point
        
        Args:
            expr: Expression to find limit of
            var: Variable approaching the limit
            point: Point that variable approaches (can be oo, -oo)
            direction: Direction of approach ('+', '-', or '+-' for both sides)
            
        Returns:
            Limit value
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        # Handle infinity
        if point == float('inf'):
            point = sp.oo
        elif point == float('-inf'):
            point = -sp.oo
        
        try:
            if direction == "+-":
                # Two-sided limit
                result = sp.limit(expr, var, point)
            elif direction == "+":
                # Right-sided limit
                result = sp.limit(expr, var, point, '+')
            elif direction == "-":
                # Left-sided limit
                result = sp.limit(expr, var, point, '-')
            else:
                raise ValueError("Direction must be '+', '-', or '+-'")
            
            self.logger.info(f"Computed limit as {var} approaches {point}")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute limit: {str(e)}")
    
    def left_limit(self, expr: Union[str, sp.Basic], 
                  var: Union[str, sp.Symbol],
                  point: Union[float, int, sp.Basic]) -> sp.Basic:
        """
        Compute left-sided (one-sided) limit
        
        Args:
            expr: Expression to find limit of
            var: Variable approaching the limit
            point: Point that variable approaches
            
        Returns:
            Left-sided limit value
        """
        return self.limit(expr, var, point, direction="-")
    
    def right_limit(self, expr: Union[str, sp.Basic], 
                   var: Union[str, sp.Symbol],
                   point: Union[float, int, sp.Basic]) -> sp.Basic:
        """
        Compute right-sided (one-sided) limit
        
        Args:
            expr: Expression to find limit of
            var: Variable approaching the limit
            point: Point that variable approaches
            
        Returns:
            Right-sided limit value
        """
        return self.limit(expr, var, point, direction="+")
    
    def limit_at_infinity(self, expr: Union[str, sp.Basic], 
                         var: Union[str, sp.Symbol],
                         direction: str = "+") -> sp.Basic:
        """
        Compute limit as variable approaches infinity
        
        Args:
            expr: Expression to find limit of
            var: Variable approaching infinity
            direction: '+' for +∞, '-' for -∞
            
        Returns:
            Limit at infinity
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        if direction == "+":
            point = sp.oo
        elif direction == "-":
            point = -sp.oo
        else:
            raise ValueError("Direction must be '+' or '-' for infinity limits")
        
        try:
            result = sp.limit(expr, var, point)
            self.logger.info(f"Computed limit at {direction}infinity")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute limit at infinity: {str(e)}")
    
    def multivariable_limit(self, expr: Union[str, sp.Basic],
                           approach_dict: Dict[Union[str, sp.Symbol], Union[float, int, sp.Basic]]) -> sp.Basic:
        """
        Compute limit for multivariable functions
        
        Args:
            expr: Multivariable expression
            approach_dict: Dictionary {variable: approach_point}
            
        Returns:
            Multivariable limit
        """
        expr = self.validate_expression(expr)
        
        # Validate approach dictionary
        validated_approaches = {}
        for var, point in approach_dict.items():
            var = self.validate_variable(var)
            if point == float('inf'):
                point = sp.oo
            elif point == float('-inf'):
                point = -sp.oo
            validated_approaches[var] = point
        
        try:
            result = expr
            # Apply limits sequentially (note: order may matter for multivariable limits)
            for var, point in validated_approaches.items():
                result = sp.limit(result, var, point)
            
            self.logger.info(f"Computed multivariable limit for {len(approach_dict)} variables")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute multivariable limit: {str(e)}")
    
    def l_hopital_rule(self, numerator: Union[str, sp.Basic],
                      denominator: Union[str, sp.Basic],
                      var: Union[str, sp.Symbol],
                      point: Union[float, int, sp.Basic],
                      max_iterations: int = 5) -> sp.Basic:
        """
        Apply L'Hôpital's rule for indeterminate forms
        
        Args:
            numerator: Numerator expression
            denominator: Denominator expression  
            var: Variable approaching the limit
            point: Point of approach
            max_iterations: Maximum number of L'Hôpital applications
            
        Returns:
            Limit using L'Hôpital's rule
        """
        numerator = self.validate_expression(numerator)
        denominator = self.validate_expression(denominator)
        var = self.validate_variable(var)
        
        if point == float('inf'):
            point = sp.oo
        elif point == float('-inf'):
            point = -sp.oo
        
        try:
            # Check if we have an indeterminate form
            num_limit = sp.limit(numerator, var, point)
            den_limit = sp.limit(denominator, var, point)
            
            # Check for 0/0 or ∞/∞ forms
            indeterminate_forms = [
                (0, 0),
                (sp.oo, sp.oo),
                (-sp.oo, -sp.oo),
                (sp.oo, -sp.oo),
                (-sp.oo, sp.oo)
            ]
            
            current_num = numerator
            current_den = denominator
            iterations = 0
            
            while iterations < max_iterations:
                num_lim = sp.limit(current_num, var, point)
                den_lim = sp.limit(current_den, var, point)
                
                if (num_lim, den_lim) not in indeterminate_forms:
                    # Not an indeterminate form, compute regular limit
                    if den_lim != 0:
                        result = num_lim / den_lim
                    else:
                        result = sp.oo if num_lim > 0 else -sp.oo
                    break
                
                # Apply L'Hôpital's rule: differentiate numerator and denominator
                current_num = sp.diff(current_num, var)
                current_den = sp.diff(current_den, var)
                iterations += 1
            else:
                # Max iterations reached, compute limit of final expression
                result = sp.limit(current_num / current_den, var, point)
            
            self.logger.info(f"Applied L'Hôpital's rule ({iterations} iterations)")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to apply L'Hôpital's rule: {str(e)}")
    
    def series_limit(self, expr: Union[str, sp.Basic],
                    var: Union[str, sp.Symbol],
                    point: Union[float, int, sp.Basic],
                    order: int = 6) -> sp.Basic:
        """
        Compute limit using series expansion
        
        Args:
            expr: Expression to find limit of
            var: Variable approaching the limit
            point: Point of approach
            order: Order of series expansion
            
        Returns:
            Limit using series expansion
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        if point == float('inf'):
            point = sp.oo
        elif point == float('-inf'):
            point = -sp.oo
        
        try:
            # Expand expression as series around the point
            series_expansion = sp.series(expr, var, point, n=order)
            
            # Remove higher order terms
            series_expansion = series_expansion.removeO()
            
            # Compute limit of series
            result = sp.limit(series_expansion, var, point)
            
            self.logger.info(f"Computed limit using series expansion (order {order})")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute limit using series: {str(e)}")
    
    def squeeze_theorem_check(self, expr: Union[str, sp.Basic],
                             lower_bound: Union[str, sp.Basic],
                             upper_bound: Union[str, sp.Basic],
                             var: Union[str, sp.Symbol],
                             point: Union[float, int, sp.Basic]) -> Dict[str, Any]:
        """
        Check conditions for squeeze theorem application
        
        Args:
            expr: Expression to find limit of
            lower_bound: Lower bounding function
            upper_bound: Upper bounding function
            var: Variable approaching the limit
            point: Point of approach
            
        Returns:
            Dictionary with squeeze theorem analysis
        """
        expr = self.validate_expression(expr)
        lower_bound = self.validate_expression(lower_bound)
        upper_bound = self.validate_expression(upper_bound)
        var = self.validate_variable(var)
        
        if point == float('inf'):
            point = sp.oo
        elif point == float('-inf'):
            point = -sp.oo
        
        result = {
            "applicable": False,
            "lower_limit": None,
            "upper_limit": None,
            "function_limit": None,
            "bounds_equal": False,
            "message": ""
        }
        
        try:
            # Compute limits of bounds
            lower_limit = sp.limit(lower_bound, var, point)
            upper_limit = sp.limit(upper_bound, var, point)
            
            result["lower_limit"] = lower_limit
            result["upper_limit"] = upper_limit
            
            # Check if bounds have equal limits
            if lower_limit == upper_limit:
                result["bounds_equal"] = True
                result["function_limit"] = lower_limit
                result["applicable"] = True
                result["message"] = f"Squeeze theorem applies: limit = {lower_limit}"
            else:
                result["message"] = f"Bounds have different limits: {lower_limit} and {upper_limit}"
            
            self.logger.info("Analyzed squeeze theorem applicability")
            return result
        except Exception as e:
            result["message"] = f"Failed to analyze squeeze theorem: {str(e)}"
            return result
    
    def continuity_check(self, expr: Union[str, sp.Basic],
                        var: Union[str, sp.Symbol],
                        point: Union[float, int, sp.Basic]) -> Dict[str, Any]:
        """
        Check continuity of function at a point
        
        Args:
            expr: Expression to check
            var: Variable
            point: Point to check continuity at
            
        Returns:
            Dictionary with continuity analysis
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        result = {
            "continuous": False,
            "function_value": None,
            "limit_value": None,
            "left_limit": None,
            "right_limit": None,
            "type": "unknown"
        }
        
        try:
            # Evaluate function at the point
            try:
                function_value = expr.subs(var, point)
                result["function_value"] = function_value
            except:
                result["function_value"] = "undefined"
            
            # Compute two-sided limit
            try:
                limit_value = sp.limit(expr, var, point)
                result["limit_value"] = limit_value
            except:
                result["limit_value"] = "undefined"
            
            # Compute one-sided limits
            try:
                left_limit = sp.limit(expr, var, point, '-')
                result["left_limit"] = left_limit
            except:
                result["left_limit"] = "undefined"
            
            try:
                right_limit = sp.limit(expr, var, point, '+')
                result["right_limit"] = right_limit
            except:
                result["right_limit"] = "undefined"
            
            # Determine continuity type
            if (result["function_value"] != "undefined" and 
                result["limit_value"] != "undefined" and
                result["function_value"] == result["limit_value"]):
                result["continuous"] = True
                result["type"] = "continuous"
            elif (result["left_limit"] != "undefined" and 
                  result["right_limit"] != "undefined" and
                  result["left_limit"] != result["right_limit"]):
                result["type"] = "jump_discontinuity"
            elif result["limit_value"] == "undefined":
                result["type"] = "infinite_discontinuity"
            elif (result["function_value"] != "undefined" and
                  result["limit_value"] != "undefined" and
                  result["function_value"] != result["limit_value"]):
                result["type"] = "removable_discontinuity"
            
            self.logger.info(f"Analyzed continuity at point {point}")
            return result
        except Exception as e:
            result["type"] = f"error: {str(e)}"
            return result
    
    def asymptote_analysis(self, expr: Union[str, sp.Basic],
                          var: Union[str, sp.Symbol]) -> Dict[str, List]:
        """
        Find horizontal, vertical, and oblique asymptotes
        
        Args:
            expr: Expression to analyze
            var: Variable
            
        Returns:
            Dictionary with asymptote information
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        result = {
            "horizontal": [],
            "vertical": [],
            "oblique": []
        }
        
        try:
            # Horizontal asymptotes (limits at ±∞)
            try:
                limit_pos_inf = sp.limit(expr, var, sp.oo)
                if limit_pos_inf.is_finite:
                    result["horizontal"].append(f"y = {limit_pos_inf}")
            except:
                pass
            
            try:
                limit_neg_inf = sp.limit(expr, var, -sp.oo)
                if limit_neg_inf.is_finite:
                    result["horizontal"].append(f"y = {limit_neg_inf}")
            except:
                pass
            
            # Vertical asymptotes (find points where denominator = 0)
            if expr.is_rational_function():
                # Get denominator
                numer, denom = sp.fraction(expr)
                
                # Find zeros of denominator
                denom_zeros = sp.solve(denom, var)
                
                for zero in denom_zeros:
                    try:
                        # Check if numerator is also zero (removable discontinuity)
                        numer_at_zero = numer.subs(var, zero)
                        if numer_at_zero != 0:
                            # Check if limits go to infinity
                            left_lim = sp.limit(expr, var, zero, '-')
                            right_lim = sp.limit(expr, var, zero, '+')
                            
                            if (left_lim == sp.oo or left_lim == -sp.oo or
                                right_lim == sp.oo or right_lim == -sp.oo):
                                result["vertical"].append(f"x = {zero}")
                    except:
                        continue
            
            # Oblique asymptotes (for rational functions where degree(num) = degree(den) + 1)
            if expr.is_rational_function():
                numer, denom = sp.fraction(expr)
                
                # Check degree condition
                if (sp.degree(numer, var) == sp.degree(denom, var) + 1):
                    # Perform polynomial long division
                    quotient, remainder = sp.div(numer, denom)
                    
                    # The quotient gives the oblique asymptote
                    if quotient.is_polynomial(var) and sp.degree(quotient, var) == 1:
                        result["oblique"].append(f"y = {quotient}")
            
            self.logger.info("Analyzed asymptotes")
            return result
        except Exception as e:
            self.logger.warning(f"Failed to analyze asymptotes: {str(e)}")
            return result
