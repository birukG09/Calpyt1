"""
Plotting and visualization module for Calpyt1 framework
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sympy as sp
from typing import Union, Optional, List, Dict, Any, Callable, Tuple
from ..core.base import BaseModule


class CalcPlotter(BaseModule):
    """
    Handles plotting and visualization for mathematical functions and data
    """
    
    def __init__(self, engine):
        """Initialize plotting module"""
        super().__init__(engine)
        self.logger.info("CalcPlotter module initialized")
        
        # Set default style
        plt.style.use('default')
        self.default_figsize = (10, 6)
        self.default_dpi = 100
    
    def _sympify_to_callable(self, expr: Union[str, sp.Basic], 
                            variables: List[str]) -> Callable:
        """Convert SymPy expression to callable function"""
        if callable(expr):
            return expr
            
        if isinstance(expr, str):
            expr = sp.sympify(expr)
        
        # Create symbols
        sym_vars = [sp.Symbol(var) for var in variables]
        
        # Create lambdified function
        return sp.lambdify(sym_vars, expr, modules=['numpy'])
    
    def plot_2d(self, func: Union[str, sp.Basic, Callable],
               x_range: Tuple[float, float] = (-10, 10),
               n_points: int = 1000,
               title: str = "2D Plot",
               xlabel: str = "x",
               ylabel: str = "y",
               grid: bool = True,
               figsize: Optional[Tuple[int, int]] = None,
               style: str = '-',
               color: str = 'blue',
               save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot 2D function
        
        Args:
            func: Function to plot
            x_range: Range of x values (xmin, xmax)
            n_points: Number of points to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            grid: Whether to show grid
            figsize: Figure size (width, height)
            style: Line style
            color: Line color
            save_path: Path to save plot (optional)
            
        Returns:
            Plot information dictionary
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x = self.engine.create_symbol('x')
                func = self._sympify_to_callable(func, ['x'])
            
            # Generate points
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            y_vals = np.array([func(x) for x in x_vals])
            
            # Create plot
            figsize = figsize or self.default_figsize
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            ax.plot(x_vals, y_vals, style, color=color, linewidth=2)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            
            if grid:
                ax.grid(True, alpha=0.3)
            
            # Add zero lines
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.7)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.7)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
            result = {
                'success': True,
                'x_range': x_range,
                'n_points': n_points,
                'x_data': x_vals.tolist(),
                'y_data': y_vals.tolist(),
                'plot_type': '2D function plot'
            }
            
            self.logger.info(f"Created 2D plot with {n_points} points")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to create 2D plot: {str(e)}")
    
    def plot_3d_surface(self, func: Union[str, sp.Basic, Callable],
                       x_range: Tuple[float, float] = (-5, 5),
                       y_range: Tuple[float, float] = (-5, 5),
                       n_points: int = 50,
                       title: str = "3D Surface Plot",
                       xlabel: str = "x",
                       ylabel: str = "y",
                       zlabel: str = "z",
                       colormap: str = 'viridis',
                       figsize: Optional[Tuple[int, int]] = None,
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot 3D surface
        
        Args:
            func: Function to plot f(x, y)
            x_range: Range of x values
            y_range: Range of y values
            n_points: Number of points per axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            zlabel: Z-axis label
            colormap: Colormap for surface
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Plot information dictionary
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x, y = self.engine.create_symbols('x y')
                func = self._sympify_to_callable(func, ['x', 'y'])
            
            # Generate meshgrid
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            y_vals = np.linspace(y_range[0], y_range[1], n_points)
            X, Y = np.meshgrid(x_vals, y_vals)
            
            # Evaluate function
            Z = np.zeros_like(X)
            for i in range(n_points):
                for j in range(n_points):
                    Z[i, j] = func(X[i, j], Y[i, j])
            
            # Create 3D plot
            figsize = figsize or (12, 8)
            fig = plt.figure(figsize=figsize, dpi=self.default_dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            surface = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=0.9,
                                    linewidth=0, antialiased=True)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_zlabel(zlabel, fontsize=12)
            
            # Add colorbar
            fig.colorbar(surface, shrink=0.5, aspect=5)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
            result = {
                'success': True,
                'x_range': x_range,
                'y_range': y_range,
                'n_points': n_points,
                'plot_type': '3D surface plot'
            }
            
            self.logger.info(f"Created 3D surface plot with {n_points}x{n_points} points")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to create 3D surface plot: {str(e)}")
    
    def plot_gradient_field(self, func: Union[str, sp.Basic, Callable],
                           x_range: Tuple[float, float] = (-5, 5),
                           y_range: Tuple[float, float] = (-5, 5),
                           n_points: int = 20,
                           title: str = "Gradient Field",
                           figsize: Optional[Tuple[int, int]] = None,
                           scale: float = 1.0,
                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot gradient vector field
        
        Args:
            func: Scalar function f(x, y)
            x_range: Range of x values
            y_range: Range of y values
            n_points: Number of grid points per axis
            title: Plot title
            figsize: Figure size
            scale: Scale factor for arrows
            save_path: Path to save plot
            
        Returns:
            Plot information dictionary
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x, y = self.engine.create_symbols('x y')
                func_expr = self.validate_expression(func)
                func = self._sympify_to_callable(func_expr, ['x', 'y'])
                
                # Compute gradient symbolically
                grad_x = sp.diff(func_expr, x)
                grad_y = sp.diff(func_expr, y)
                
                grad_x_func = self._sympify_to_callable(grad_x, ['x', 'y'])
                grad_y_func = self._sympify_to_callable(grad_y, ['x', 'y'])
            else:
                # Compute gradient numerically
                def grad_x_func(x, y):
                    h = 1e-8
                    return (func(x + h, y) - func(x - h, y)) / (2 * h)
                
                def grad_y_func(x, y):
                    h = 1e-8
                    return (func(x, y + h) - func(x, y - h)) / (2 * h)
            
            # Generate grid
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            y_vals = np.linspace(y_range[0], y_range[1], n_points)
            X, Y = np.meshgrid(x_vals, y_vals)
            
            # Compute gradient at each point
            U = np.zeros_like(X)
            V = np.zeros_like(Y)
            
            for i in range(n_points):
                for j in range(n_points):
                    U[i, j] = grad_x_func(X[i, j], Y[i, j])
                    V[i, j] = grad_y_func(X[i, j], Y[i, j])
            
            # Create plot
            figsize = figsize or self.default_figsize
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            # Plot gradient field
            ax.quiver(X, Y, U, V, scale=scale, alpha=0.8, width=0.003)
            
            # Add contour lines of the original function
            Z = np.zeros_like(X)
            for i in range(n_points):
                for j in range(n_points):
                    Z[i, j] = func(X[i, j], Y[i, j])
            
            contour = ax.contour(X, Y, Z, levels=15, alpha=0.6, colors='gray', linewidths=0.5)
            ax.clabel(contour, inline=True, fontsize=8)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
            result = {
                'success': True,
                'x_range': x_range,
                'y_range': y_range,
                'n_points': n_points,
                'plot_type': 'gradient field plot'
            }
            
            self.logger.info(f"Created gradient field plot with {n_points}x{n_points} vectors")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to create gradient field plot: {str(e)}")
    
    def plot_phase_portrait(self, ode_system: List[Union[str, sp.Basic]],
                           x_range: Tuple[float, float] = (-5, 5),
                           y_range: Tuple[float, float] = (-5, 5),
                           n_points: int = 20,
                           trajectories: Optional[List[Tuple[float, float]]] = None,
                           title: str = "Phase Portrait",
                           figsize: Optional[Tuple[int, int]] = None,
                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot phase portrait for 2D ODE system
        
        Args:
            ode_system: System of ODEs [dx/dt, dy/dt]
            x_range: Range of x values
            y_range: Range of y values
            n_points: Number of grid points per axis
            trajectories: List of initial conditions for trajectories
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Plot information dictionary
        """
        try:
            if len(ode_system) != 2:
                raise ValueError("Phase portrait requires exactly 2 ODEs")
            
            # Get phase portrait data from ODE solver
            phase_data = self.engine.ode_solver.phase_portrait_data(
                ode_system, x_range, y_range, n_points
            )
            
            # Create plot
            figsize = figsize or self.default_figsize
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            # Plot direction field
            X, Y = phase_data['Y1'], phase_data['Y2']
            U, V = phase_data['DY1_normalized'], phase_data['DY2_normalized']
            
            ax.quiver(X, Y, U, V, alpha=0.6, width=0.003, scale=30)
            
            # Plot trajectories if provided
            if trajectories:
                from scipy.integrate import odeint
                
                # Convert system to function
                variables = ['t', 'y0', 'y1']
                func1 = self._sympify_to_callable(ode_system[0], variables)
                func2 = self._sympify_to_callable(ode_system[1], variables)
                
                def system_func(state, t):
                    y1, y2 = state
                    return [func1(t, y1, y2), func2(t, y1, y2)]
                
                # Time points
                t = np.linspace(0, 10, 1000)
                
                for x0, y0 in trajectories:
                    try:
                        # Forward trajectory
                        traj_forward = odeint(system_func, [x0, y0], t)
                        ax.plot(traj_forward[:, 0], traj_forward[:, 1], 'r-', linewidth=2, alpha=0.8)
                        
                        # Backward trajectory
                        traj_backward = odeint(system_func, [x0, y0], -t)
                        ax.plot(traj_backward[:, 0], traj_backward[:, 1], 'r-', linewidth=2, alpha=0.8)
                        
                        # Mark initial condition
                        ax.plot(x0, y0, 'ro', markersize=6)
                    except:
                        # Skip if trajectory computation fails
                        continue
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
            result = {
                'success': True,
                'x_range': x_range,
                'y_range': y_range,
                'n_trajectories': len(trajectories) if trajectories else 0,
                'plot_type': 'phase portrait'
            }
            
            self.logger.info("Created phase portrait plot")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to create phase portrait: {str(e)}")
    
    def plot_interactive_3d(self, func: Union[str, sp.Basic, Callable],
                           x_range: Tuple[float, float] = (-5, 5),
                           y_range: Tuple[float, float] = (-5, 5),
                           n_points: int = 50,
                           title: str = "Interactive 3D Plot",
                           colorscale: str = 'viridis') -> go.Figure:
        """
        Create interactive 3D plot using Plotly
        
        Args:
            func: Function to plot f(x, y)
            x_range: Range of x values
            y_range: Range of y values
            n_points: Number of points per axis
            title: Plot title
            colorscale: Plotly colorscale
            
        Returns:
            Plotly figure object
        """
        try:
            # Convert to callable if needed
            if not callable(func):
                x, y = self.engine.create_symbols('x y')
                func = self._sympify_to_callable(func, ['x', 'y'])
            
            # Generate meshgrid
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            y_vals = np.linspace(y_range[0], y_range[1], n_points)
            X, Y = np.meshgrid(x_vals, y_vals)
            
            # Evaluate function
            Z = np.zeros_like(X)
            for i in range(n_points):
                for j in range(n_points):
                    Z[i, j] = func(X[i, j], Y[i, j])
            
            # Create interactive plot
            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale=colorscale,
                showscale=True
            )])
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='x',
                    yaxis_title='y',
                    zaxis_title='z'
                ),
                autosize=True,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            
            self.logger.info(f"Created interactive 3D plot with {n_points}x{n_points} points")
            return fig
        except Exception as e:
            raise RuntimeError(f"Failed to create interactive 3D plot: {str(e)}")
    
    def plot_multiple_functions(self, functions: Dict[str, Union[str, sp.Basic, Callable]],
                               x_range: Tuple[float, float] = (-10, 10),
                               n_points: int = 1000,
                               title: str = "Multiple Functions",
                               figsize: Optional[Tuple[int, int]] = None,
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot multiple functions on the same axes
        
        Args:
            functions: Dictionary of {label: function} pairs
            x_range: Range of x values
            n_points: Number of points to plot
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Plot information dictionary
        """
        try:
            # Generate x values
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            
            # Create plot
            figsize = figsize or self.default_figsize
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, (label, func) in enumerate(functions.items()):
                # Convert to callable if needed
                if not callable(func):
                    x = self.engine.create_symbol('x')
                    func = self._sympify_to_callable(func, ['x'])
                
                # Evaluate function
                y_vals = np.array([func(x) for x in x_vals])
                
                # Plot with different color
                color = colors[i % len(colors)]
                ax.plot(x_vals, y_vals, label=label, color=color, linewidth=2)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add zero lines
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.7)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.7)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
            result = {
                'success': True,
                'n_functions': len(functions),
                'function_labels': list(functions.keys()),
                'x_range': x_range,
                'n_points': n_points,
                'plot_type': 'multiple functions plot'
            }
            
            self.logger.info(f"Created plot with {len(functions)} functions")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to create multiple functions plot: {str(e)}")
    
    def plot_convergence(self, data: Dict[str, List[float]],
                        title: str = "Convergence Plot",
                        xlabel: str = "Iteration",
                        ylabel: str = "Value",
                        log_scale: bool = False,
                        figsize: Optional[Tuple[int, int]] = None,
                        save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot convergence data from iterative algorithms
        
        Args:
            data: Dictionary of {series_name: values_list}
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Whether to use log scale for y-axis
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Plot information dictionary
        """
        try:
            figsize = figsize or self.default_figsize
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            
            for i, (label, values) in enumerate(data.items()):
                iterations = range(len(values))
                color = colors[i % len(colors)]
                ax.plot(iterations, values, label=label, color=color, 
                       marker='o', markersize=4, linewidth=2)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            
            if log_scale:
                ax.set_yscale('log')
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
            result = {
                'success': True,
                'n_series': len(data),
                'series_names': list(data.keys()),
                'log_scale': log_scale,
                'plot_type': 'convergence plot'
            }
            
            self.logger.info(f"Created convergence plot with {len(data)} series")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to create convergence plot: {str(e)}")
    
    def create_dashboard(self, plots_config: List[Dict[str, Any]],
                        title: str = "Mathematical Dashboard") -> go.Figure:
        """
        Create interactive dashboard with multiple plots
        
        Args:
            plots_config: List of plot configuration dictionaries
            title: Dashboard title
            
        Returns:
            Plotly figure with subplots
        """
        try:
            n_plots = len(plots_config)
            
            # Determine subplot layout
            if n_plots <= 2:
                rows, cols = 1, n_plots
            elif n_plots <= 4:
                rows, cols = 2, 2
            else:
                rows = int(np.ceil(np.sqrt(n_plots)))
                cols = int(np.ceil(n_plots / rows))
            
            # Create subplots
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[config.get('title', f'Plot {i+1}') 
                              for i, config in enumerate(plots_config)],
                specs=[[{'type': 'scatter3d' if config.get('type') == '3d' else 'xy'} 
                       for j in range(cols)] for i in range(rows)]
            )
            
            for i, config in enumerate(plots_config):
                row = i // cols + 1
                col = i % cols + 1
                
                plot_type = config.get('type', '2d')
                func = config.get('function')
                x_range = config.get('x_range', (-5, 5))
                
                if plot_type == '2d':
                    # 2D plot
                    if not callable(func):
                        x = self.engine.create_symbol('x')
                        func = self._sympify_to_callable(func, ['x'])
                    
                    x_vals = np.linspace(x_range[0], x_range[1], 200)
                    y_vals = np.array([func(x) for x in x_vals])
                    
                    fig.add_trace(
                        go.Scatter(x=x_vals, y=y_vals, mode='lines', 
                                 name=config.get('name', f'Function {i+1}')),
                        row=row, col=col
                    )
                
                elif plot_type == '3d':
                    # 3D surface plot
                    if not callable(func):
                        x, y = self.engine.create_symbols('x y')
                        func = self._sympify_to_callable(func, ['x', 'y'])
                    
                    y_range = config.get('y_range', (-5, 5))
                    n_points = config.get('n_points', 30)
                    
                    x_vals = np.linspace(x_range[0], x_range[1], n_points)
                    y_vals = np.linspace(y_range[0], y_range[1], n_points)
                    X, Y = np.meshgrid(x_vals, y_vals)
                    
                    Z = np.zeros_like(X)
                    for j in range(n_points):
                        for k in range(n_points):
                            Z[j, k] = func(X[j, k], Y[j, k])
                    
                    fig.add_trace(
                        go.Surface(x=X, y=Y, z=Z, showscale=False,
                                 name=config.get('name', f'Surface {i+1}')),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title=title,
                height=600 * rows,
                showlegend=True
            )
            
            self.logger.info(f"Created dashboard with {n_plots} plots")
            return fig
        except Exception as e:
            raise RuntimeError(f"Failed to create dashboard: {str(e)}")
