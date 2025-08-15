"""
Symbolic integrals module for Calpyt1 framework
"""

import sympy as sp
from typing import Union, Optional, List, Dict, Any, Tuple
from ..core.base import BaseModule


class SymbolicIntegrals(BaseModule):
    """
    Handles symbolic integration operations
    """
    
    def __init__(self, engine):
        """Initialize symbolic integrals module"""
        super().__init__(engine)
        self.logger.info("SymbolicIntegrals module initialized")
    
    def indefinite_integral(self, expr: Union[str, sp.Basic], 
                           var: Union[str, sp.Symbol]) -> sp.Basic:
        """
        Compute indefinite integral (antiderivative)
        
        Args:
            expr: Expression to integrate
            var: Variable of integration
            
        Returns:
            Indefinite integral
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        try:
            result = sp.integrate(expr, var)
            self.logger.info("Computed indefinite integral")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute indefinite integral: {str(e)}")
    
    def definite_integral(self, expr: Union[str, sp.Basic], 
                         var: Union[str, sp.Symbol],
                         lower_limit: Union[float, int, sp.Basic],
                         upper_limit: Union[float, int, sp.Basic]) -> sp.Basic:
        """
        Compute definite integral
        
        Args:
            expr: Expression to integrate
            var: Variable of integration
            lower_limit: Lower limit of integration
            upper_limit: Upper limit of integration
            
        Returns:
            Definite integral value
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        try:
            result = sp.integrate(expr, (var, lower_limit, upper_limit))
            self.logger.info("Computed definite integral")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute definite integral: {str(e)}")
    
    def multiple_integral(self, expr: Union[str, sp.Basic], 
                         integration_vars: List[Tuple]) -> sp.Basic:
        """
        Compute multiple integral (double, triple, etc.)
        
        Args:
            expr: Expression to integrate
            integration_vars: List of (var, lower_limit, upper_limit) tuples
            
        Returns:
            Multiple integral value
        """
        expr = self.validate_expression(expr)
        
        # Validate integration variables and limits
        validated_vars = []
        for var_limits in integration_vars:
            if len(var_limits) == 1:
                # Indefinite case
                var = self.validate_variable(var_limits[0])
                validated_vars.append(var)
            elif len(var_limits) == 3:
                # Definite case
                var, lower, upper = var_limits
                var = self.validate_variable(var)
                validated_vars.append((var, lower, upper))
            else:
                raise ValueError("Each integration variable must be (var,) or (var, lower, upper)")
        
        try:
            result = sp.integrate(expr, *validated_vars)
            self.logger.info(f"Computed {len(integration_vars)}-fold integral")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute multiple integral: {str(e)}")
    
    def line_integral(self, vector_field: List[Union[str, sp.Basic]], 
                     curve_params: Dict[str, Union[str, sp.Basic]],
                     parameter: Union[str, sp.Symbol],
                     param_range: Tuple[Union[float, int, sp.Basic], Union[float, int, sp.Basic]]) -> sp.Basic:
        """
        Compute line integral of vector field along a curve
        
        Args:
            vector_field: List of vector field components [P, Q, R]
            curve_params: Dictionary mapping variables to parametric expressions
            parameter: Parameter variable (e.g., t)
            param_range: (t_min, t_max) range for parameter
            
        Returns:
            Line integral value
        """
        vector_field = [self.validate_expression(component) for component in vector_field]
        parameter = self.validate_variable(parameter)
        
        # Validate curve parameters
        validated_curve = {}
        for var, param_expr in curve_params.items():
            var_symbol = self.validate_variable(var)
            param_expr = self.validate_expression(param_expr)
            validated_curve[var_symbol] = param_expr
        
        try:
            # Substitute parametric equations into vector field
            substituted_field = []
            for component in vector_field:
                sub_component = component
                for var, param_expr in validated_curve.items():
                    sub_component = sub_component.subs(var, param_expr)
                substituted_field.append(sub_component)
            
            # Compute derivatives of parametric equations
            derivatives = []
            for var, param_expr in validated_curve.items():
                derivatives.append(sp.diff(param_expr, parameter))
            
            # Compute dot product F · dr
            integrand = sum(substituted_field[i] * derivatives[i] for i in range(len(derivatives)))
            
            # Integrate over parameter range
            result = sp.integrate(integrand, (parameter, param_range[0], param_range[1]))
            
            self.logger.info("Computed line integral")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute line integral: {str(e)}")
    
    def surface_integral(self, expr: Union[str, sp.Basic],
                        surface_params: Dict[str, Union[str, sp.Basic]],
                        parameters: List[Union[str, sp.Symbol]],
                        param_ranges: List[Tuple]) -> sp.Basic:
        """
        Compute surface integral
        
        Args:
            expr: Expression to integrate over surface
            surface_params: Dictionary mapping (x,y,z) to parametric expressions
            parameters: List of parameter variables [u, v]
            param_ranges: List of parameter ranges [(u_min, u_max), (v_min, v_max)]
            
        Returns:
            Surface integral value
        """
        expr = self.validate_expression(expr)
        parameters = [self.validate_variable(param) for param in parameters]
        
        if len(parameters) != 2:
            raise ValueError("Surface integrals require exactly 2 parameters")
        
        # Validate surface parameters
        validated_surface = {}
        for var, param_expr in surface_params.items():
            var_symbol = self.validate_variable(var)
            param_expr = self.validate_expression(param_expr)
            validated_surface[var_symbol] = param_expr
        
        try:
            u, v = parameters
            
            # Get parametric equations
            x_param = validated_surface.get(self.engine.create_symbol('x'))
            y_param = validated_surface.get(self.engine.create_symbol('y'))
            z_param = validated_surface.get(self.engine.create_symbol('z'))
            
            if not all([x_param, y_param, z_param]):
                raise ValueError("Surface parameters must include x, y, and z")
            
            # Compute partial derivatives
            xu = sp.diff(x_param, u)
            xv = sp.diff(x_param, v)
            yu = sp.diff(y_param, u)
            yv = sp.diff(y_param, v)
            zu = sp.diff(z_param, u)
            zv = sp.diff(z_param, v)
            
            # Compute cross product magnitude (surface element)
            cross_product = [
                yu * zv - zu * yv,
                zu * xv - xu * zv,
                xu * yv - yu * xv
            ]
            
            surface_element = sp.sqrt(sum(comp**2 for comp in cross_product))
            
            # Substitute parametric equations into expression
            substituted_expr = expr
            for var, param_expr in validated_surface.items():
                substituted_expr = substituted_expr.subs(var, param_expr)
            
            # Create integrand
            integrand = substituted_expr * surface_element
            
            # Integrate over parameter ranges
            result = sp.integrate(integrand, 
                                (u, param_ranges[0][0], param_ranges[0][1]),
                                (v, param_ranges[1][0], param_ranges[1][1]))
            
            self.logger.info("Computed surface integral")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute surface integral: {str(e)}")
    
    def integration_by_parts(self, u: Union[str, sp.Basic], 
                           dv: Union[str, sp.Basic],
                           var: Union[str, sp.Symbol]) -> sp.Basic:
        """
        Perform integration by parts: ∫u dv = uv - ∫v du
        
        Args:
            u: First function
            dv: Second function differential
            var: Variable of integration
            
        Returns:
            Result of integration by parts
        """
        u = self.validate_expression(u)
        dv = self.validate_expression(dv)
        var = self.validate_variable(var)
        
        try:
            # Compute du and v
            du = sp.diff(u, var)
            v = sp.integrate(dv, var)
            
            # Apply integration by parts formula
            result = u * v - sp.integrate(v * du, var)
            
            self.logger.info("Applied integration by parts")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to apply integration by parts: {str(e)}")
    
    def substitution_integral(self, expr: Union[str, sp.Basic],
                             substitution: Dict[Union[str, sp.Symbol], Union[str, sp.Basic]],
                             new_var: Union[str, sp.Symbol],
                             var: Union[str, sp.Symbol]) -> sp.Basic:
        """
        Perform integration by substitution
        
        Args:
            expr: Expression to integrate
            substitution: Dictionary of substitutions {old_var: new_expr}
            new_var: New variable after substitution
            var: Original variable of integration
            
        Returns:
            Integral with substitution applied
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        new_var = self.validate_variable(new_var)
        
        # Validate substitutions
        validated_subs = {}
        for old_var, new_expr in substitution.items():
            old_var = self.validate_variable(old_var) if isinstance(old_var, str) else old_var
            new_expr = self.validate_expression(new_expr)
            validated_subs[old_var] = new_expr
        
        try:
            # Apply substitution
            substituted_expr = expr
            for old_var, new_expr in validated_subs.items():
                substituted_expr = substituted_expr.subs(old_var, new_expr)
            
            # Integrate with respect to new variable
            result = sp.integrate(substituted_expr, new_var)
            
            self.logger.info("Applied substitution integration")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to apply substitution integration: {str(e)}")
    
    def improper_integral(self, expr: Union[str, sp.Basic],
                         var: Union[str, sp.Symbol],
                         lower_limit: Union[float, int, sp.Basic],
                         upper_limit: Union[float, int, sp.Basic]) -> sp.Basic:
        """
        Compute improper integral with infinite or discontinuous limits
        
        Args:
            expr: Expression to integrate
            var: Variable of integration
            lower_limit: Lower limit (can be -oo)
            upper_limit: Upper limit (can be oo)
            
        Returns:
            Improper integral value
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        try:
            # Replace infinity symbols
            if lower_limit == float('-inf'):
                lower_limit = -sp.oo
            if upper_limit == float('inf'):
                upper_limit = sp.oo
            
            result = sp.integrate(expr, (var, lower_limit, upper_limit))
            
            self.logger.info("Computed improper integral")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute improper integral: {str(e)}")
    
    def get_integration_techniques(self, expr: Union[str, sp.Basic], 
                                  var: Union[str, sp.Symbol]) -> Dict[str, Any]:
        """
        Suggest appropriate integration techniques for an expression
        
        Args:
            expr: Expression to analyze
            var: Variable of integration
            
        Returns:
            Dictionary with suggested techniques and information
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        techniques = {
            "direct_integration": False,
            "substitution": False,
            "integration_by_parts": False,
            "partial_fractions": False,
            "trigonometric_substitution": False,
            "suggestions": []
        }
        
        try:
            # Check if direct integration works
            try:
                direct_result = sp.integrate(expr, var)
                if direct_result != sp.Integral(expr, var):  # Check if it's evaluated
                    techniques["direct_integration"] = True
                    techniques["suggestions"].append("Direct integration is possible")
            except:
                pass
            
            # Check for polynomial expressions (easy integration)
            if expr.is_polynomial(var):
                techniques["suggestions"].append("Polynomial - use power rule")
            
            # Check for rational functions (partial fractions)
            if expr.is_rational_function(var):
                techniques["partial_fractions"] = True
                techniques["suggestions"].append("Rational function - consider partial fractions")
            
            # Check for products (integration by parts)
            if expr.is_Mul:
                techniques["integration_by_parts"] = True
                techniques["suggestions"].append("Product of functions - consider integration by parts")
            
            # Check for trigonometric functions
            trig_funcs = [sp.sin, sp.cos, sp.tan, sp.sec, sp.csc, sp.cot]
            if any(expr.has(func) for func in trig_funcs):
                techniques["trigonometric_substitution"] = True
                techniques["suggestions"].append("Contains trigonometric functions - consider trigonometric substitution")
            
            # Check for composite functions (substitution)
            if expr.is_Function or expr.has(sp.exp) or expr.has(sp.log):
                techniques["substitution"] = True
                techniques["suggestions"].append("Composite function - consider substitution")
            
            self.logger.info("Analyzed integration techniques")
            return techniques
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze integration techniques: {str(e)}")
            return techniques
