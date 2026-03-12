"""CLI entry point for ai-foundry-cost-analyzer."""

import click

from .pricing import list_models, get_model
from .tokens import calculate_cost, estimate_from_text, count_tokens
from .context import project_cost, analyze_conversation
from .agents import (
    AgentConfig,
    simulate_single_agent,
    simulate_multi_agent_request,
    compare_single_vs_multi,
)
from .tools_cost import ToolDefinition
from .formatter import (
    console,
    print_model_table,
    print_cost_result,
    print_context_analysis,
    print_multi_agent_result,
    print_comparison,
)


@click.group()
@click.version_option()
def cli():
    """AI Foundry Cost Analyzer — Understand LLM token costs."""
    pass


@cli.command()
def models():
    """List all supported models with pricing."""
    print_model_table(list_models())


@cli.command()
@click.option("-m", "--model", default="gpt-4.1", help="Model name")
@click.option("-i", "--input-tokens", type=int, required=True, help="Number of input tokens")
@click.option("-o", "--output-tokens", type=int, required=True, help="Number of output tokens")
@click.option("-c", "--cached-tokens", type=int, default=0, help="Cached input tokens")
def calculate(model, input_tokens, output_tokens, cached_tokens):
    """Calculate cost for a given number of tokens."""
    result = calculate_cost(input_tokens, output_tokens, model, cached_tokens)
    print_cost_result(result)


@cli.command()
@click.option("-m", "--model", default="gpt-4.1", help="Model name")
@click.option("-p", "--prompt", required=True, help="Prompt text (or @file to read from file)")
@click.option("-r", "--response", required=True, help="Response text (or @file to read from file)")
def estimate(model, prompt, response):
    """Estimate cost from prompt and response text."""
    prompt_text = _read_text(prompt)
    response_text = _read_text(response)
    result = estimate_from_text(prompt_text, response_text, model)
    print_cost_result(result)
    console.print(f"\nPrompt: {result['prompt_chars']:,} chars → {result['input_tokens']:,} tokens")
    console.print(f"Response: {result['response_chars']:,} chars → {result['output_tokens']:,} tokens")


@cli.command("context")
@click.option("-m", "--model", default="gpt-4.1", help="Model name")
@click.option("-t", "--turns", type=int, required=True, help="Number of conversation turns")
@click.option("--user-tokens", type=int, default=150, help="Average user message tokens")
@click.option("--assistant-tokens", type=int, default=300, help="Average assistant response tokens")
@click.option("--system-tokens", type=int, default=500, help="System prompt tokens")
def context_cmd(model, turns, user_tokens, assistant_tokens, system_tokens):
    """Project context window cost escalation over N turns."""
    result = project_cost(
        avg_user_tokens=user_tokens,
        avg_assistant_tokens=assistant_tokens,
        num_turns=turns,
        model=model,
        system_prompt_tokens=system_tokens,
    )
    print_context_analysis(result)


@cli.command("multi-agent")
@click.option("-m", "--model", default="gpt-4.1", help="Model for all agents")
@click.option("--agents", type=int, default=2, help="Number of sub-agents invoked")
@click.option("--tools-per-agent", type=int, default=3, help="Tools per sub-agent")
@click.option("--tool-calls", type=int, default=1, help="Avg tool calls per agent per turn")
@click.option("--user-tokens", type=int, default=150, help="User message tokens")
@click.option("--context-tokens", type=int, default=500, help="Shared context tokens")
def multi_agent_cmd(model, agents, tools_per_agent, tool_calls, user_tokens, context_tokens):
    """Simulate multi-agent request cost."""
    # Build orchestrator
    orchestrator = AgentConfig(
        name="Orchestrator",
        model=model,
        system_prompt_tokens=800,
        tools=[
            ToolDefinition(
                name=f"route_to_agent_{i}",
                description=f"Route request to sub-agent {i}",
                parameters_json='{"type": "object", "properties": {"query": {"type": "string"}}}',
            )
            for i in range(agents)
        ],
        avg_tool_calls_per_turn=1,
        avg_response_tokens=200,
    )

    # Build sub-agents
    sub_agents = []
    delegated = []
    for i in range(agents):
        name = f"Agent-{i+1}"
        sub_agents.append(AgentConfig(
            name=name,
            model=model,
            system_prompt_tokens=500,
            tools=[
                ToolDefinition(
                    name=f"tool_{j}",
                    description=f"Tool {j} for agent {i+1}",
                    parameters_json='{"type": "object", "properties": {"input": {"type": "string"}}}',
                )
                for j in range(tools_per_agent)
            ],
            avg_tool_calls_per_turn=tool_calls,
            avg_response_tokens=300,
        ))
        delegated.append(name)

    result = simulate_multi_agent_request(
        orchestrator, sub_agents, delegated,
        shared_context_tokens=context_tokens,
        user_message_tokens=user_tokens,
    )
    print_multi_agent_result(result)


@cli.command("compare")
@click.option("-m", "--model", default="gpt-4.1", help="Model name")
@click.option("-i", "--input-tokens", type=int, default=500, help="Task input tokens")
@click.option("-o", "--output-tokens", type=int, default=300, help="Response tokens")
@click.option("--agents", type=int, default=2, help="Sub-agents in multi-agent setup")
def compare_cmd(model, input_tokens, output_tokens, agents):
    """Compare single agent vs multi-agent cost."""
    orchestrator = AgentConfig(name="Orchestrator", model=model, avg_response_tokens=100)
    sub_agents = [
        AgentConfig(name=f"Agent-{i+1}", model=model, avg_response_tokens=output_tokens // agents)
        for i in range(agents)
    ]
    delegated = [a.name for a in sub_agents]

    result = compare_single_vs_multi(
        task_tokens=input_tokens,
        response_tokens=output_tokens,
        single_model=model,
        orchestrator=orchestrator,
        sub_agents=sub_agents,
        delegated=delegated,
    )
    print_comparison(result)


@cli.command("count")
@click.option("-m", "--model", default="gpt-4.1", help="Model name")
@click.argument("text")
def count_cmd(model, text):
    """Count tokens in a text string or file (@filename)."""
    content = _read_text(text)
    tokens = count_tokens(content, model)
    console.print(f"[cyan]{tokens:,}[/cyan] tokens ({len(content):,} chars) using [dim]{model}[/dim] encoding")


def _read_text(value: str) -> str:
    """Read text from a value; if starts with @, read from file."""
    if value.startswith("@"):
        path = value[1:]
        with open(path, encoding="utf-8") as f:
            return f.read()
    return value
