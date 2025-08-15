"""
Symbolic derivatives module for Calpyt1 framework
"""

import sympy as sp
from typing import Union, Optional, List, Dict, Any
from ..core.base import BaseModule


class SymbolicDerivatives(BaseModule):
    """
    Handles symbolic differentiation operations
    """
    
    def __init__(self, engine):
        """Initialize symbolic derivatives module"""
        super().__init__(engine)
        self.logger.info("SymbolicDerivatives module initialized")
    
    def derivative(self, expr: Union[str, sp.Basic], var: Union[str, sp.Symbol], 
                  order: int = 1) -> sp.Basic:
        """
        Compute symbolic derivative
        
        Args:
            expr: Expression to differentiate
            var: Variable to differentiate with respect to
            order: Order of derivative (default: 1)
            
        Returns:
            Symbolic derivative
        """
        expr = self.validate_expression(expr)
        var = self.validate_variable(var)
        
        if order < 1:
            raise ValueError("Order must be positive integer")
        
        try:
            result = sp.diff(expr, var, order)
            self.logger.info(f"Computed derivative of order {order}")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute derivative: {str(e)}")
    
    def partial_derivative(self, expr: Union[str, sp.Basic], 
                          vars_orders: List[tuple]) -> sp.Basic:
        """
        Compute partial derivative with respect to multiple variables
        
        Args:
            expr: Expression to differentiate
            vars_orders: List of (variable, order) tuples
            
        Returns:
            Partial derivative
        """
        expr = self.validate_expression(expr)
        
        # Validate variables and orders
        validated_vars_orders = []
        for var_order in vars_orders:
            if len(var_order) != 2:
                raise ValueError("Each element must be (variable, order) tuple")
            var, order = var_order
            var = self.validate_variable(var)
            if not isinstance(order, int) or order < 1:
                raise ValueError("Order must be positive integer")
            validated_vars_orders.append((var, order))
        
        try:
            result = expr
            for var, order in validated_vars_orders:
                result = sp.diff(result, var, order)
            
            self.logger.info(f"Computed partial derivative with respect to {len(vars_orders)} variables")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute partial derivative: {str(e)}")
    
    def gradient(self, expr: Union[str, sp.Basic], 
                variables: List[Union[str, sp.Symbol]]) -> List[sp.Basic]:
        """
        Compute gradient vector
        
        Args:
            expr: Scalar expression
            variables: List of variables
            
        Returns:
            List of partial derivatives (gradient components)
        """
        expr = self.validate_expression(expr)
        variables = [self.validate_variable(var) for var in variables]
        
        try:
            gradient = [sp.diff(expr, var) for var in variables]
            self.logger.info(f"Computed gradient with {len(variables)} components")
            return gradient
        except Exception as e:
            raise RuntimeError(f"Failed to compute gradient: {str(e)}")
    
    def jacobian(self, expressions: List[Union[str, sp.Basic]], 
                variables: List[Union[str, sp.Symbol]]) -> sp.Matrix:
        """
        Compute Jacobian matrix
        
        Args:
            expressions: List of expressions (vector function)
            variables: List of variables
            
        Returns:
            Jacobian matrix
        """
        expressions = [self.validate_expression(expr) for expr in expressions]
        variables = [self.validate_variable(var) for var in variables]
        
        try:
            # Create matrix of expressions
            expr_matrix = sp.Matrix(expressions)
            var_matrix = sp.Matrix(variables)
            
            # Compute Jacobian
            jacobian = expr_matrix.jacobian(var_matrix)
            
            self.logger.info(f"Computed Jacobian matrix ({len(expressions)}x{len(variables)})")
            return jacobian
        except Exception as e:
            raise RuntimeError(f"Failed to compute Jacobian: {str(e)}")
    
    def hessian(self, expr: Union[str, sp.Basic], 
               variables: List[Union[str, sp.Symbol]]) -> sp.Matrix:
        """
        Compute Hessian matrix (matrix of second-order partial derivatives)
        
        Args:
            expr: Scalar expression
            variables: List of variables
            
        Returns:
            Hessian matrix
        """
        expr = self.validate_expression(expr)
        variables = [self.validate_variable(var) for var in variables]
        
        try:
            hessian = sp.hessian(expr, variables)
            self.logger.info(f"Computed Hessian matrix ({len(variables)}x{len(variables)})")
            return hessian
        except Exception as e:
            raise RuntimeError(f"Failed to compute Hessian: {str(e)}")
    
    def laplacian(self, expr: Union[str, sp.Basic], 
                 variables: List[Union[str, sp.Symbol]]) -> sp.Basic:
        """
        Compute Laplacian (sum of second partial derivatives)
        
        Args:
            expr: Scalar expression
            variables: List of variables
            
        Returns:
            Laplacian
        """
        expr = self.validate_expression(expr)
        variables = [self.validate_variable(var) for var in variables]
        
        try:
            laplacian = sum(sp.diff(expr, var, 2) for var in variables)
            self.logger.info(f"Computed Laplacian with {len(variables)} variables")
            return laplacian
        except Exception as e:
            raise RuntimeError(f"Failed to compute Laplacian: {str(e)}")
    
    def directional_derivative(self, expr: Union[str, sp.Basic], 
                              variables: List[Union[str, sp.Symbol]],
                              direction: List[Union[float, int, sp.Basic]]) -> sp.Basic:
        """
        Compute directional derivative
        
        Args:
            expr: Scalar expression
            variables: List of variables
            direction: Direction vector (same length as variables)
            
        Returns:
            Directional derivative
        """
        if len(variables) != len(direction):
            raise ValueError("Direction vector must have same length as variables list")
        
        expr = self.validate_expression(expr)
        variables = [self.validate_variable(var) for var in variables]
        
        try:
            # Compute gradient
            grad = self.gradient(expr, variables)
            
            # Dot product with direction vector
            directional_deriv = sum(grad[i] * direction[i] for i in range(len(variables)))
            
            self.logger.info("Computed directional derivative")
            return directional_deriv
        except Exception as e:
            raise RuntimeError(f"Failed to compute directional derivative: {str(e)}")
    
    def implicit_derivative(self, equation: Union[str, sp.Basic],
                           dependent_var: Union[str, sp.Symbol],
                           independent_var: Union[str, sp.Symbol]) -> sp.Basic:
        """
        Compute implicit derivative using implicit differentiation
        
        Args:
            equation: Equation in the form F(x,y) = 0
            dependent_var: Dependent variable (e.g., y)
            independent_var: Independent variable (e.g., x)
            
        Returns:
            dy/dx expression
        """
        equation = self.validate_expression(equation)
        dependent_var = self.validate_variable(dependent_var)
        independent_var = self.validate_variable(independent_var)
        
        try:
            # Use SymPy's idiff function for implicit differentiation
            result = sp.idiff(equation, dependent_var, independent_var)
            
            self.logger.info("Computed implicit derivative")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute implicit derivative: {str(e)}")
    
    def chain_rule(self, outer_expr: Union[str, sp.Basic],
                   inner_expr: Union[str, sp.Basic],
                   var: Union[str, sp.Symbol]) -> sp.Basic:
        """
        Apply chain rule for composite functions
        
        Args:
            outer_expr: Outer function f(u)
            inner_expr: Inner function u(x) 
            var: Variable to differentiate with respect to
            
        Returns:
            Chain rule result: f'(u) * u'(x)
        """
        outer_expr = self.validate_expression(outer_expr)
        inner_expr = self.validate_expression(inner_expr)
        var = self.validate_variable(var)
        
        try:
            # Create temporary variable for substitution
            u = self.engine.create_symbol('u_temp')
            
            # Get outer derivative with respect to u
            outer_deriv = sp.diff(outer_expr, u)
            
            # Substitute inner expression for u
            outer_deriv = outer_deriv.subs(u, inner_expr)
            
            # Get inner derivative
            inner_deriv = sp.diff(inner_expr, var)
            
            # Apply chain rule
            result = outer_deriv * inner_deriv
            
            self.logger.info("Applied chain rule")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to apply chain rule: {str(e)}")
    
    def get_critical_points(self, expr: Union[str, sp.Basic], 
                           variables: List[Union[str, sp.Symbol]]) -> List[Dict]:
        """
        Find critical points by solving gradient = 0
        
        Args:
            expr: Scalar expression
            variables: List of variables
            
        Returns:
            List of critical points as dictionaries
        """
        expr = self.validate_expression(expr)
        variables = [self.validate_variable(var) for var in variables]
        
        try:
            # Compute gradient
            grad = self.gradient(expr, variables)
            
            # Solve gradient = 0
            critical_points = sp.solve(grad, variables)
            
            # Convert to list of dictionaries if not already
            if isinstance(critical_points, dict):
                critical_points = [critical_points]
            elif isinstance(critical_points, list) and len(critical_points) > 0:
                if not isinstance(critical_points[0], dict):
                    # Handle case where solutions are tuples
                    if len(variables) == 1:
                        critical_points = [{variables[0]: pt} for pt in critical_points]
                    else:
                        critical_points = [dict(zip(variables, pt)) for pt in critical_points]
            
            self.logger.info(f"Found {len(critical_points)} critical point(s)")
            return critical_points
        except Exception as e:
            raise RuntimeError(f"Failed to find critical points: {str(e)}")
