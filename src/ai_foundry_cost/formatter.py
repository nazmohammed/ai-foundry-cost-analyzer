"""Rich formatting utilities for terminal output."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


def format_currency(value: float) -> str:
    """Format a float as a USD currency string."""
    if value < 0.01:
        return f"${value:.6f}"
    return f"${value:.4f}"


def print_model_table(models: list) -> None:
    """Print a table of all available models with pricing."""
    table = Table(title="Azure AI Foundry Model Pricing", show_lines=True)
    table.add_column("Model", style="cyan bold")
    table.add_column("Input / 1M tokens", justify="right", style="green")
    table.add_column("Output / 1M tokens", justify="right", style="red")
    table.add_column("Cached Input / 1M", justify="right", style="yellow")
    table.add_column("Context Window", justify="right")
    table.add_column("Max Output", justify="right")

    for m in models:
        table.add_row(
            m.name,
            f"${m.input_per_1m:.2f}",
            f"${m.output_per_1m:.2f}",
            f"${m.cached_input_per_1m:.3f}",
            f"{m.context_window:,}",
            f"{m.max_output:,}",
        )

    console.print(table)


def print_cost_result(result: dict) -> None:
    """Print a token cost calculation result."""
    table = Table(title=f"Cost Estimate — {result['model']}", show_lines=True)
    table.add_column("Component", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right", style="green")

    table.add_row("Input tokens", f"{result['input_tokens']:,}", format_currency(result['input_cost']))
    if result.get('cached_input_tokens', 0) > 0:
        table.add_row("Cached input", f"{result['cached_input_tokens']:,}", format_currency(result['cached_input_cost']))
    table.add_row("Output tokens", f"{result['output_tokens']:,}", format_currency(result['output_cost']))
    table.add_row("Total", "", format_currency(result['total_cost']))

    console.print(table)


def print_context_analysis(result: dict) -> None:
    """Print context window cost analysis."""
    table = Table(
        title=f"Context Cost Escalation — {result['model']} ({result['num_turns']} turns)",
        show_lines=True,
    )
    table.add_column("Turn", justify="right", style="cyan")
    table.add_column("Context Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Turn Cost", justify="right", style="yellow")
    table.add_column("Cumulative", justify="right", style="green")

    for tc in result["turns"]:
        table.add_row(
            str(tc.turn_number),
            f"{tc.context_tokens:,}",
            f"{tc.output_tokens:,}",
            format_currency(tc.total_cost),
            format_currency(tc.cumulative_cost),
        )

    console.print(table)

    summary = Text()
    summary.append(f"Total input tokens: {result['total_input_tokens']:,}\n")
    summary.append(f"Total output tokens: {result['total_output_tokens']:,}\n")
    summary.append(f"Total cost: {format_currency(result['total_cost'])}", style="bold green")
    console.print(Panel(summary, title="Summary"))


def print_tool_overhead(result: dict) -> None:
    """Print tool definition overhead analysis."""
    table = Table(title=f"Tool Definition Overhead — {result['model']}", show_lines=True)
    table.add_column("Tool", style="cyan")
    table.add_column("Tokens", justify="right")

    for t in result["per_tool"]:
        table.add_row(t["name"], f"{t['tokens']:,}")

    table.add_row("[bold]Total[/bold]", f"[bold]{result['total_tokens']:,}[/bold]")
    console.print(table)
    console.print(f"Cost per API call: [green]{format_currency(result['cost_per_call'])}[/green]")


def print_multi_agent_result(result: dict) -> None:
    """Print multi-agent simulation results."""
    table = Table(title="Multi-Agent Cost Breakdown", show_lines=True)
    table.add_column("Component", style="cyan")
    table.add_column("Cost", justify="right", style="green")

    breakdown = result["cost_breakdown"]
    table.add_row("Orchestrator routing", format_currency(breakdown["orchestrator_routing"]))
    table.add_row("Sub-agent processing", format_currency(breakdown["sub_agent_processing"]))
    table.add_row("Orchestrator synthesis", format_currency(breakdown["orchestrator_synthesis"]))
    table.add_row("[bold]Total[/bold]", f"[bold]{format_currency(result['total_cost'])}[/bold]")

    console.print(table)

    # Sub-agent details
    if result["sub_agents"]:
        detail = Table(title="Sub-Agent Details", show_lines=True)
        detail.add_column("Agent", style="cyan")
        detail.add_column("Model", style="dim")
        detail.add_column("Input Tokens", justify="right")
        detail.add_column("Output Tokens", justify="right")
        detail.add_column("Cost", justify="right", style="green")

        for sa in result["sub_agents"]:
            detail.add_row(
                sa.agent_name,
                sa.model,
                f"{sa.input_tokens:,}",
                f"{sa.output_tokens:,}",
                format_currency(sa.total_cost),
            )

        console.print(detail)

    console.print(f"\nTotal API calls: [cyan]{result['num_api_calls']}[/cyan]")


def print_comparison(result: dict) -> None:
    """Print single vs multi-agent cost comparison."""
    table = Table(title="Single Agent vs Multi-Agent Cost", show_lines=True)
    table.add_column("Approach", style="cyan")
    table.add_column("Cost", justify="right", style="green")

    table.add_row("Single agent", format_currency(result["single_agent_cost"]))
    table.add_row("Multi-agent", format_currency(result["multi_agent_cost"]))
    table.add_row("Additional cost", format_currency(result["additional_cost"]))
    table.add_row("Cost multiplier", f"{result['cost_multiplier']}x")

    console.print(table)
