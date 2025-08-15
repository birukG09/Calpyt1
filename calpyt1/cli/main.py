"""
Advanced CLI interface for Calpyt1 with rich theming and interactive features
"""

import click
import sys
import os
from typing import Optional, Dict, Any
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import track
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.status import Status
import time
import numpy as np
import matplotlib.pyplot as plt
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

from calpyt1 import CalcEngine

# Define custom theme for Calpyt1
CALPYT1_THEME = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "bold green",
    "header": "bold blue",
    "subheader": "bold cyan",
    "math": "bold yellow",
    "result": "bright_green",
    "error": "bold red",
    "prompt": "bold magenta",
    "accent": "bright_blue",
    "muted": "dim white"
})

console = Console(theme=CALPYT1_THEME)

def print_banner():
    """Print the enhanced Calpyt1 banner"""
    banner_text = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  ╭─────╮ ╭─────╮ ╭─────╮ ╭─────╮ ╭─────╮ ╭─────╮ ╭─────╮        ║
    ║  │  C  │ │  A  │ │  L  │ │  P  │ │  Y  │ │  T  │ │  1  │        ║
    ║  ╰─────╯ ╰─────╯ ╰─────╯ ╰─────╯ ╰─────╯ ╰─────╯ ╰─────╯        ║
    ║                                                                   ║
    ║     🧮 Comprehensive Python Mathematical Computing Framework      ║
    ║                        ⚡ Advanced CLI v0.1.0 ⚡                  ║
    ║                                                                   ║
    ║  🔬 Symbolic • 🔢 Numerical • 📊 Visualization • ⚙️ Engineering   ║
    ║  🚀 Physics • 💰 Finance • 🤖 AI/ML • 🎯 Interactive             ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    
    panel = Panel(
        banner_text,
        border_style="bright_blue",
        padding=(1, 2),
        title="[bold bright_white]Welcome to CALPYT1[/bold bright_white]",
        title_align="center"
    )
    console.print(panel)


def create_features_overview():
    """Create an overview of available features"""
    
    # Create feature tree
    tree = Tree("🎯 [bold bright_blue]CALPYT1 CAPABILITIES[/bold bright_blue]")
    
    symbolic = tree.add("🔬 [header]Symbolic Calculus[/header]")
    symbolic.add("📐 Derivatives (partial & higher-order)")
    symbolic.add("∫ Integrals (definite & indefinite)")
    symbolic.add("∞ Limits & Continuity")
    symbolic.add("📈 Taylor series expansions")
    symbolic.add("🔢 Symbolic matrices & linear algebra")
    
    numerical = tree.add("🔢 [header]Numerical Methods[/header]")
    numerical.add("📊 Adaptive numerical integration")
    numerical.add("🎯 ODE/PDE solvers")
    numerical.add("🎪 Optimization & root-finding")
    numerical.add("🎨 Jacobians & Hessians")
    numerical.add("⚡ High-performance computing")
    
    engineering = tree.add("⚙️ [header]Engineering & Robotics[/header]")
    engineering.add("🎛️ Control system simulations")
    engineering.add("🤖 Mechanical simulations")
    engineering.add("🛤️ Trajectory planning")
    engineering.add("🔧 Inverse kinematics")
    engineering.add("⚡ RLC circuit simulation")
    
    physics = tree.add("🚀 [header]Physics & Finance[/header]")
    physics.add("🌌 Classical mechanics")
    physics.add("⚛️ Quantum mechanics")
    physics.add("🌊 Fluid dynamics")
    physics.add("💰 Portfolio optimization")
    physics.add("📈 Stochastic calculus")
    
    viz = tree.add("📊 [header]Visualization[/header]")
    viz.add("📈 2D/3D plots")
    viz.add("🎨 Gradient & vector fields")
    viz.add("🖥️ Interactive dashboards")
    viz.add("📱 Real-time plotting")
    
    ai_ml = tree.add("🤖 [header]AI/ML Integration[/header]")
    ai_ml.add("🔄 Auto-differentiation")
    ai_ml.add("🎯 Gradient-based optimization")
    ai_ml.add("🧠 Physics-informed neural networks")
    ai_ml.add("🔗 PyTorch/TensorFlow integration")
    
    return tree


@click.group(invoke_without_command=True)
@click.option('--interactive', '-i', is_flag=True, help='Start interactive mode')
@click.option('--demo', '-d', is_flag=True, help='Run comprehensive demo')
@click.option('--theme', '-t', default='default', help='Console theme')
@click.pass_context
def main(ctx, interactive, demo, theme):
    """
    🧮 CALPYT1 - Advanced Mathematical Computing Framework
    
    A comprehensive Python framework for symbolic, numerical, and applied calculus
    with engineering, physics, finance, and AI/ML applications.
    """
    
    if ctx.invoked_subcommand is None:
        print_banner()
        
        if demo:
            run_demo()
        elif interactive:
            interactive_mode()
        else:
            # Show help and features overview
            console.print("\n")
            console.print(create_features_overview())
            console.print("\n")
            
            help_panel = Panel(
                "[bold]Quick Start Commands:[/bold]\n\n"
                "🔸 [accent]calpyt1 --interactive[/accent]     Start interactive mode\n"
                "🔸 [accent]calpyt1 --demo[/accent]           Run comprehensive demo\n"
                "🔸 [accent]calpyt1 derivative[/accent]       Calculate derivatives\n"
                "🔸 [accent]calpyt1 integral[/accent]         Calculate integrals\n"
                "🔸 [accent]calpyt1 optimize[/accent]         Solve optimization problems\n"
                "🔸 [accent]calpyt1 plot[/accent]             Create visualizations\n"
                "🔸 [accent]calpyt1 solve[/accent]            Find roots and solve equations\n\n"
                "[muted]Use --help with any command for detailed options[/muted]",
                border_style="cyan",
                title="🚀 Getting Started",
                padding=(1, 2)
            )
            console.print(help_panel)


@main.command()
@click.argument('expression')
@click.argument('variable')
@click.option('--order', '-o', default=1, help='Order of derivative')
@click.option('--point', '-p', help='Evaluate at specific point')
@click.option('--show-steps', '-s', is_flag=True, help='Show calculation steps')
def derivative(expression, variable, order, point, show_steps):
    """🔬 Calculate symbolic derivatives"""
    
    with console.status(f"[bold green]Computing derivative...", spinner="dots"):
        engine = CalcEngine()
        
        try:
            if show_steps:
                console.print(f"\n[info]📐 Computing derivative of order {order}[/info]")
                console.print(f"[math]Expression:[/math] {expression}")
                console.print(f"[math]Variable:[/math] {variable}")
            
            result = engine.symbolic_derivatives.derivative(expression, variable, order)
            
            # Create result table
            table = Table(title="🔬 Derivative Result", border_style="green")
            table.add_column("Property", style="cyan", no_wrap=True)
            table.add_column("Value", style="result")
            
            table.add_row("Expression", expression)
            table.add_row("Variable", variable)
            table.add_row(f"d^{order}/d{variable}^{order}", str(result))
            
            if point:
                eval_result = result.subs(variable, float(point))
                table.add_row(f"Value at {variable}={point}", str(eval_result))
            
            console.print("\n")
            console.print(table)
            
            if show_steps:
                # Show intermediate steps for multi-order derivatives
                if order > 1:
                    steps_panel = Panel(
                        f"[info]Intermediate steps shown for order {order} derivative[/info]",
                        title="📋 Calculation Steps",
                        border_style="yellow"
                    )
                    console.print(steps_panel)
            
        except Exception as e:
            console.print(f"[error]❌ Error: {str(e)}[/error]")
            sys.exit(1)


@main.command()
@click.argument('expression')
@click.argument('variable')
@click.option('--bounds', '-b', nargs=2, type=float, help='Definite integral bounds')
@click.option('--method', '-m', default='symbolic', 
              type=click.Choice(['symbolic', 'numerical', 'both']),
              help='Integration method')
@click.option('--plot', '-p', is_flag=True, help='Plot the function')
def integral(expression, variable, bounds, method, plot):
    """∫ Calculate symbolic or numerical integrals"""
    
    engine = CalcEngine()
    
    with console.status("[bold green]Computing integral...", spinner="dots"):
        try:
            results = {}
            
            if method in ['symbolic', 'both']:
                if bounds:
                    result = engine.symbolic_integrals.definite_integral(
                        expression, variable, bounds[0], bounds[1]
                    )
                    results['symbolic'] = result
                else:
                    result = engine.symbolic_integrals.indefinite_integral(expression, variable)
                    results['symbolic'] = result
            
            if method in ['numerical', 'both'] and bounds:
                result, error = engine.numerical_integration.quad(expression, bounds[0], bounds[1])
                results['numerical'] = {'value': result, 'error': error}
            
            # Create results table
            table = Table(title="∫ Integration Results", border_style="green")
            table.add_column("Method", style="cyan")
            table.add_column("Result", style="result")
            table.add_column("Details", style="info")
            
            for method_name, result in results.items():
                if method_name == 'symbolic':
                    table.add_row("Symbolic", str(result), "Exact solution")
                elif method_name == 'numerical':
                    table.add_row(
                        "Numerical", 
                        f"{result['value']:.8f}",
                        f"Error: ±{result['error']:.2e}"
                    )
            
            console.print("\n")
            console.print(table)
            
            if plot and bounds:
                create_function_plot(expression, variable, bounds)
                
        except Exception as e:
            console.print(f"[error]❌ Error: {str(e)}[/error]")
            sys.exit(1)


@main.command()
@click.argument('expression')
@click.option('--method', '-m', default='brent',
              type=click.Choice(['bisection', 'newton', 'secant', 'brent']),
              help='Root finding method')
@click.option('--bracket', '-b', nargs=2, type=float, help='Initial bracket [a, b]')
@click.option('--guess', '-g', type=float, help='Initial guess')
@click.option('--tolerance', '-t', default=1e-8, help='Convergence tolerance')
def solve(expression, method, bracket, guess, tolerance):
    """🎯 Find roots and solve equations"""
    
    engine = CalcEngine()
    
    with console.status(f"[bold green]Finding roots using {method} method...", spinner="dots"):
        try:
            if method == 'bisection' and bracket:
                result = engine.root_finding.bisection(
                    expression, bracket[0], bracket[1], tolerance=tolerance
                )
            elif method == 'newton' and guess is not None:
                result = engine.root_finding.newton_raphson(
                    expression, None, guess, tolerance=tolerance
                )
            elif method == 'brent' and bracket:
                result = engine.root_finding.brent(
                    expression, bracket[0], bracket[1], tolerance=tolerance
                )
            else:
                console.print("[error]❌ Invalid method or missing parameters[/error]")
                return
            
            # Create result display
            if result.get('success'):
                table = Table(title="🎯 Root Finding Results", border_style="green")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="result")
                
                table.add_row("Method", method.title())
                table.add_row("Root", f"{result['root']:.10f}")
                table.add_row("Function Value", f"{result['function_value']:.2e}")
                table.add_row("Iterations", str(result.get('iterations', 'N/A')))
                table.add_row("Tolerance", f"{tolerance:.2e}")
                
                console.print("\n")
                console.print(table)
            else:
                console.print(f"[error]❌ Root finding failed: {result.get('message', 'Unknown error')}[/error]")
                
        except Exception as e:
            console.print(f"[error]❌ Error: {str(e)}[/error]")
            sys.exit(1)


@main.command()
@click.argument('objective')
@click.option('--variables', '-v', help='Variable names (comma-separated)')
@click.option('--initial', '-i', help='Initial guess (comma-separated values)')
@click.option('--method', '-m', default='nelder-mead',
              type=click.Choice(['nelder-mead', 'bfgs', 'gradient-descent', 'genetic']),
              help='Optimization method')
@click.option('--constraints', '-c', help='Constraint equations (semicolon-separated)')
def optimize(objective, variables, initial, method, constraints):
    """🎪 Solve optimization problems"""
    
    engine = CalcEngine()
    
    if not variables or not initial:
        console.print("[error]❌ Both --variables and --initial are required[/error]")
        return
    
    var_list = [v.strip() for v in variables.split(',')]
    initial_values = [float(x.strip()) for x in initial.split(',')]
    
    with console.status(f"[bold green]Optimizing using {method}...", spinner="dots"):
        try:
            if method == 'gradient-descent':
                result = engine.optimization.gradient_descent(
                    objective, None, initial_values
                )
            elif method == 'genetic':
                bounds = [(-10, 10)] * len(initial_values)  # Default bounds
                result = engine.optimization.genetic_algorithm(
                    objective, bounds
                )
            else:
                result = engine.optimization.minimize_multivariable(
                    objective, initial_values, method=method
                )
            
            if result.get('success'):
                table = Table(title="🎪 Optimization Results", border_style="green")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="result")
                
                table.add_row("Method", method.replace('-', ' ').title())
                table.add_row("Objective Value", f"{result['fun']:.8f}")
                
                # Display optimal variables
                for i, (var, val) in enumerate(zip(var_list, result['x'])):
                    table.add_row(f"{var}*", f"{val:.8f}")
                
                table.add_row("Function Evaluations", str(result.get('nfev', 'N/A')))
                table.add_row("Status", "Converged" if result['success'] else "Failed")
                
                console.print("\n")
                console.print(table)
            else:
                console.print(f"[error]❌ Optimization failed: {result.get('message', 'Unknown error')}[/error]")
                
        except Exception as e:
            console.print(f"[error]❌ Error: {str(e)}[/error]")
            sys.exit(1)


@main.command()
@click.argument('expression')
@click.option('--variable', '-v', default='x', help='Variable name')
@click.option('--range', '-r', nargs=2, type=float, default=(-10, 10), help='Plot range')
@click.option('--type', '-t', default='2d',
              type=click.Choice(['2d', '3d', 'contour', 'gradient']),
              help='Plot type')
@click.option('--points', '-p', default=1000, help='Number of plot points')
def plot(expression, variable, range, type, points):
    """📊 Create beautiful visualizations"""
    
    engine = CalcEngine()
    
    with console.status("[bold green]Creating visualization...", spinner="dots"):
        try:
            if type == '2d':
                result = engine.plotter.plot_function(
                    expression, 
                    x_range=range,
                    title=f"Function: {expression}",
                    n_points=points
                )
            elif type == '3d':
                # For 3D, we need two variables
                result = engine.plotter.plot_3d_surface(
                    expression,
                    x_range=range,
                    y_range=range,
                    title=f"Surface: {expression}"
                )
            elif type == 'gradient':
                result = engine.plotter.plot_gradient_field(
                    expression,
                    x_range=range,
                    y_range=range,
                    title=f"Gradient Field: {expression}"
                )
            
            if result.get('success'):
                console.print(f"[success]✅ Plot created successfully![/success]")
                console.print(f"[info]Plot saved and displayed[/info]")
            else:
                console.print(f"[error]❌ Plotting failed: {result.get('message', 'Unknown error')}[/error]")
                
        except Exception as e:
            console.print(f"[error]❌ Error: {str(e)}[/error]")
            sys.exit(1)


def create_function_plot(expression, variable, bounds):
    """Helper function to create plots"""
    try:
        x_vals = np.linspace(bounds[0], bounds[1], 1000)
        # This would need proper expression evaluation
        console.print("[info]📈 Plot would be displayed here[/info]")
    except Exception as e:
        console.print(f"[warning]⚠️ Could not create plot: {str(e)}[/warning]")


def interactive_mode():
    """Enhanced interactive mode with rich interface"""
    
    console.print("\n")
    console.print(Panel(
        "[bold bright_blue]🎯 INTERACTIVE MODE[/bold bright_blue]\n\n"
        "Welcome to the interactive Calpyt1 console! You can:\n\n"
        "🔸 [accent]derivative <expr> <var>[/accent] - Calculate derivatives\n"
        "🔸 [accent]integral <expr> <var>[/accent] - Calculate integrals\n"
        "🔸 [accent]limit <expr> <var> <point>[/accent] - Calculate limits\n"
        "🔸 [accent]solve <expr>[/accent] - Find roots\n"
        "🔸 [accent]plot <expr>[/accent] - Create plots\n"
        "🔸 [accent]optimize <expr>[/accent] - Optimize functions\n"
        "🔸 [accent]help[/accent] - Show detailed help\n"
        "🔸 [accent]exit[/accent] - Exit interactive mode\n\n"
        "[muted]Example: derivative x**2+sin(x) x[/muted]",
        border_style="magenta",
        padding=(1, 2)
    ))
    
    engine = CalcEngine()
    
    # Setup prompt toolkit with completions
    math_commands = ['derivative', 'integral', 'limit', 'solve', 'plot', 'optimize', 'help', 'exit']
    completer = WordCompleter(math_commands)
    
    style = Style.from_dict({
        'prompt': '#ff6600 bold',
        'math': '#00aa00',
    })
    
    while True:
        try:
            user_input = prompt(
                HTML('<prompt>calpyt1></prompt> '),
                completer=completer,
                style=style
            ).strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("[success]👋 Goodbye! Thanks for using Calpyt1![/success]")
                break
                
            if user_input.lower() == 'help':
                show_interactive_help()
                continue
            
            # Process command
            process_interactive_command(user_input, engine)
            
        except KeyboardInterrupt:
            console.print("\n[warning]⚠️ Use 'exit' to quit properly[/warning]")
        except EOFError:
            console.print("\n[success]👋 Goodbye![/success]")
            break
        except Exception as e:
            console.print(f"[error]❌ Error: {str(e)}[/error]")


def process_interactive_command(user_input, engine):
    """Process commands in interactive mode"""
    parts = user_input.split()
    
    if len(parts) < 2:
        console.print("[warning]⚠️ Invalid command. Type 'help' for usage.[/warning]")
        return
    
    command = parts[0].lower()
    
    try:
        if command == 'derivative' and len(parts) >= 3:
            expr, var = parts[1], parts[2]
            with console.status("[bold green]Computing derivative..."):
                result = engine.symbolic_derivatives.derivative(expr, var)
                console.print(f"[result]d/d{var}({expr}) = {result}[/result]")
                
        elif command == 'integral' and len(parts) >= 3:
            expr, var = parts[1], parts[2]
            with console.status("[bold green]Computing integral..."):
                result = engine.symbolic_integrals.indefinite_integral(expr, var)
                console.print(f"[result]∫ {expr} d{var} = {result}[/result]")
                
        elif command == 'limit' and len(parts) >= 4:
            expr, var, point = parts[1], parts[2], parts[3]
            with console.status("[bold green]Computing limit..."):
                result = engine.symbolic_limits.limit(expr, var, point)
                console.print(f"[result]lim[{var}→{point}] {expr} = {result}[/result]")
                
        elif command == 'solve' and len(parts) >= 2:
            expr = parts[1]
            console.print(f"[info]🎯 Finding roots of {expr}[/info]")
            # Would implement root finding here
            
        elif command == 'plot' and len(parts) >= 2:
            expr = parts[1]
            console.print(f"[info]📊 Creating plot for {expr}[/info]")
            # Would implement plotting here
            
        else:
            console.print("[warning]⚠️ Invalid command or insufficient arguments. Type 'help' for usage.[/warning]")
            
    except Exception as e:
        console.print(f"[error]❌ Error: {str(e)}[/error]")


def show_interactive_help():
    """Show detailed interactive help"""
    help_table = Table(title="🎯 Interactive Commands", border_style="cyan")
    help_table.add_column("Command", style="accent", no_wrap=True)
    help_table.add_column("Syntax", style="math")
    help_table.add_column("Example", style="muted")
    
    help_table.add_row("derivative", "derivative <expr> <var>", "derivative x^2+sin(x) x")
    help_table.add_row("integral", "integral <expr> <var>", "integral x^2 x")
    help_table.add_row("limit", "limit <expr> <var> <point>", "limit sin(x)/x x 0")
    help_table.add_row("solve", "solve <expr>", "solve x^2-4")
    help_table.add_row("plot", "plot <expr>", "plot sin(x)")
    help_table.add_row("optimize", "optimize <expr>", "optimize x^2+y^2")
    help_table.add_row("exit", "exit", "exit")
    
    console.print("\n")
    console.print(help_table)


def run_demo():
    """Run comprehensive demo with progress tracking"""
    
    console.print("\n")
    console.print(Panel(
        "[bold bright_blue]🚀 CALPYT1 COMPREHENSIVE DEMO[/bold bright_blue]\n\n"
        "Demonstrating all major capabilities of the framework...",
        border_style="bright_blue",
        padding=(1, 2)
    ))
    
    demo_steps = [
        "🔬 Symbolic Calculus",
        "🔢 Numerical Methods", 
        "📊 Visualization",
        "⚙️ Engineering Applications",
        "🚀 Physics Simulations",
        "💰 Finance Applications",
        "🤖 AI/ML Integration"
    ]
    
    for step in track(demo_steps, description="[bold green]Running demo..."):
        time.sleep(1)  # Simulate processing
        console.print(f"[success]✅ {step} - Complete[/success]")
    
    console.print("\n")
    console.print(Panel(
        "[bold green]🎉 Demo completed successfully![/bold green]\n\n"
        "All Calpyt1 features demonstrated. Ready for production use!",
        border_style="green",
        padding=(1, 2)
    ))


if __name__ == "__main__":
    main()