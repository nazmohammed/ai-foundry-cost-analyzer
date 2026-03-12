"""Multi-agent cost simulation.

In a multi-agent architecture (like the Bruno Travel Companion), an
orchestrator delegates to specialized sub-agents.  Each delegation is
a separate API call chain with its own context, tools, and responses.

This module simulates the cost of an end-to-end multi-agent request:

  User → Orchestrator → [Sub-Agent 1, Sub-Agent 2, ...] → Final Response

Cost multipliers in multi-agent systems:
- Orchestrator call: classifies intent + routes
- Each sub-agent call: full context + agent-specific tools + response
- Orchestrator synthesis: combines sub-agent results into final answer
- Shared context (user profile, session) is duplicated across agents
"""

from dataclasses import dataclass, field

from .pricing import ModelPricing, get_model
from .tools_cost import ToolDefinition, estimate_tool_definitions_overhead


@dataclass
class AgentConfig:
    """Configuration for a single agent in the system."""
    name: str
    model: str = "gpt-4.1"
    system_prompt_tokens: int = 500
    tools: list[ToolDefinition] = field(default_factory=list)
    avg_tool_calls_per_turn: int = 0
    avg_tool_result_tokens: int = 200
    avg_response_tokens: int = 300


@dataclass
class AgentCostResult:
    """Cost result for a single agent invocation."""
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    tool_def_tokens: int
    tool_result_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


def simulate_single_agent(
    agent: AgentConfig,
    shared_context_tokens: int = 0,
    user_message_tokens: int = 100,
) -> AgentCostResult:
    """Simulate cost for a single agent invocation.

    Input tokens = system_prompt + shared_context + user_message + tool_defs
                 + tool_results (from tool calls within the turn)
    Output tokens = response + tool_call outputs
    """
    pricing = get_model(agent.model)

    # Tool definition overhead
    tool_def_tokens = 0
    if agent.tools:
        overhead = estimate_tool_definitions_overhead(agent.tools, pricing)
        tool_def_tokens = overhead["total_tokens"]

    # Approximate tool call tokens (model output for calling tools)
    tool_call_output_tokens = agent.avg_tool_calls_per_turn * 30  # ~30 tokens per call

    # Tool results come back as input
    tool_result_tokens = agent.avg_tool_calls_per_turn * agent.avg_tool_result_tokens

    # Number of API calls: 1 initial + 1 per tool call batch + 1 final if tools used
    if agent.avg_tool_calls_per_turn > 0:
        num_calls = 2  # initial call → tool calls → final response
    else:
        num_calls = 1

    # Total input per call (tool defs + system + context + user msg are in every call)
    base_input = agent.system_prompt_tokens + shared_context_tokens + user_message_tokens + tool_def_tokens
    total_input = base_input * num_calls + tool_result_tokens

    # Total output
    total_output = agent.avg_response_tokens + tool_call_output_tokens

    input_cost = total_input * pricing.input_per_token
    output_cost = total_output * pricing.output_per_token
    total = input_cost + output_cost

    return AgentCostResult(
        agent_name=agent.name,
        model=pricing.name,
        input_tokens=total_input,
        output_tokens=total_output,
        tool_def_tokens=tool_def_tokens * num_calls,
        tool_result_tokens=tool_result_tokens,
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        total_cost=round(total, 6),
    )


def simulate_multi_agent_request(
    orchestrator: AgentConfig,
    sub_agents: list[AgentConfig],
    delegated_agents: list[str],
    shared_context_tokens: int = 0,
    user_message_tokens: int = 100,
) -> dict:
    """Simulate the full cost of a multi-agent request.

    Flow:
    1. Orchestrator receives user message, classifies intent
    2. Orchestrator delegates to one or more sub-agents
    3. Each sub-agent processes independently
    4. Orchestrator synthesizes final response

    Args:
        orchestrator: The orchestrator agent config
        sub_agents: All available sub-agent configs
        delegated_agents: Names of sub-agents actually invoked for this request
        shared_context_tokens: Shared context (user profile, session state)
        user_message_tokens: Tokens in the user's message
    """
    # Step 1: Orchestrator classifies + routes
    orch_result = simulate_single_agent(
        orchestrator, shared_context_tokens, user_message_tokens
    )

    # Step 2-3: Each delegated sub-agent processes
    agent_map = {a.name: a for a in sub_agents}
    sub_results = []
    for name in delegated_agents:
        if name not in agent_map:
            continue
        result = simulate_single_agent(
            agent_map[name],
            shared_context_tokens,
            user_message_tokens + orch_result.output_tokens,  # orchestrator adds routing context
        )
        sub_results.append(result)

    # Step 4: Orchestrator synthesis (optional second call)
    # The orchestrator gets back sub-agent results as input
    synthesis_input = sum(r.output_tokens for r in sub_results) * 2  # rough token estimate for results
    synthesis_pricing = get_model(orchestrator.model)
    synthesis_cost = synthesis_input * synthesis_pricing.input_per_token + \
                     orchestrator.avg_response_tokens * synthesis_pricing.output_per_token

    total_cost = orch_result.total_cost + sum(r.total_cost for r in sub_results) + synthesis_cost

    return {
        "orchestrator": orch_result,
        "sub_agents": sub_results,
        "synthesis_input_tokens": synthesis_input,
        "synthesis_cost": round(synthesis_cost, 6),
        "total_cost": round(total_cost, 6),
        "num_api_calls": 1 + len(sub_results) + 1,  # orch + subs + synthesis
        "cost_breakdown": {
            "orchestrator_routing": round(orch_result.total_cost, 6),
            "sub_agent_processing": round(sum(r.total_cost for r in sub_results), 6),
            "orchestrator_synthesis": round(synthesis_cost, 6),
        },
    }


def compare_single_vs_multi(
    task_tokens: int,
    response_tokens: int,
    single_model: str = "gpt-4.1",
    orchestrator: AgentConfig | None = None,
    sub_agents: list[AgentConfig] | None = None,
    delegated: list[str] | None = None,
) -> dict:
    """Compare cost of handling a request with one model vs multi-agent."""
    single_pricing = get_model(single_model)
    single_cost = (
        task_tokens * single_pricing.input_per_token +
        response_tokens * single_pricing.output_per_token
    )

    if orchestrator and sub_agents and delegated:
        multi = simulate_multi_agent_request(
            orchestrator, sub_agents, delegated,
            user_message_tokens=task_tokens,
        )
        multi_cost = multi["total_cost"]
    else:
        multi_cost = 0.0

    return {
        "single_agent_cost": round(single_cost, 6),
        "multi_agent_cost": round(multi_cost, 6),
        "cost_multiplier": round(multi_cost / single_cost, 2) if single_cost > 0 else 0,
        "additional_cost": round(multi_cost - single_cost, 6),
    }
