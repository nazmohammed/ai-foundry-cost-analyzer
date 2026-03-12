"""Tool usage cost analysis.

When an agent uses tools (function calling), several hidden token costs arise:

1. **Tool definitions** — The JSON schema for every registered tool is injected
   into the system prompt on every API call.  More tools = more input tokens.

2. **Tool call output** — The model generates a structured tool_call (function
   name + JSON arguments) which counts as output tokens.

3. **Tool results** — The execution result is sent back as input tokens in the
   next API call, alongside the full conversation context.

4. **RAG / grounding context** — Retrieved documents injected into the prompt
   add input tokens proportional to chunk size × number of chunks.

This module quantifies each of these cost components.
"""

from dataclasses import dataclass

from .pricing import ModelPricing, get_model
from .tokens import count_tokens


@dataclass(frozen=True)
class ToolDefinition:
    """A tool/function registered with the model."""
    name: str
    description: str
    parameters_json: str  # JSON schema string for the tool's parameters


@dataclass
class ToolCallCost:
    """Cost breakdown for a single tool invocation round-trip."""
    tool_name: str
    call_output_tokens: int     # model generating the tool_call
    result_input_tokens: int    # tool result sent back as input
    call_output_cost: float
    result_input_cost: float
    total_cost: float


def estimate_tool_definitions_overhead(
    tools: list[ToolDefinition],
    model: str | ModelPricing = "gpt-4.1",
) -> dict:
    """Estimate the token overhead of tool definitions in the system prompt.

    Tool definitions are sent on EVERY API call, so this overhead
    multiplies by the number of turns in a conversation.
    """
    pricing = model if isinstance(model, ModelPricing) else get_model(model)

    per_tool = []
    total_tokens = 0
    for tool in tools:
        # Approximate how the API serializes tool defs
        serialized = f"function: {tool.name}\n{tool.description}\nparameters: {tool.parameters_json}"
        tokens = count_tokens(serialized, pricing.name)
        total_tokens += tokens
        per_tool.append({"name": tool.name, "tokens": tokens})

    cost_per_call = total_tokens * pricing.input_per_token

    return {
        "model": pricing.name,
        "num_tools": len(tools),
        "total_tokens": total_tokens,
        "cost_per_call": round(cost_per_call, 6),
        "per_tool": per_tool,
    }


def estimate_tool_call_cost(
    tool_name: str,
    call_arguments_json: str,
    tool_result: str,
    model: str | ModelPricing = "gpt-4.1",
) -> ToolCallCost:
    """Estimate the cost of a single tool call round-trip.

    A tool call has two token costs:
    1. Output tokens: the model generates function_name + arguments JSON
    2. Input tokens: the tool result is sent back in the next message
    """
    pricing = model if isinstance(model, ModelPricing) else get_model(model)

    # The model outputs the function call (name + args)
    call_text = f"{tool_name}({call_arguments_json})"
    call_tokens = count_tokens(call_text, pricing.name)
    call_cost = call_tokens * pricing.output_per_token

    # The tool result comes back as input
    result_tokens = count_tokens(tool_result, pricing.name)
    result_cost = result_tokens * pricing.input_per_token

    return ToolCallCost(
        tool_name=tool_name,
        call_output_tokens=call_tokens,
        result_input_tokens=result_tokens,
        call_output_cost=round(call_cost, 6),
        result_input_cost=round(result_cost, 6),
        total_cost=round(call_cost + result_cost, 6),
    )


def estimate_rag_cost(
    chunks: list[str],
    model: str | ModelPricing = "gpt-4.1",
) -> dict:
    """Estimate the input token cost of RAG/grounding context.

    Each retrieved chunk is injected into the prompt as additional
    input tokens.  This cost recurs on every API call if the chunks
    are included in the conversation context.
    """
    pricing = model if isinstance(model, ModelPricing) else get_model(model)

    per_chunk = []
    total_tokens = 0
    for i, chunk in enumerate(chunks):
        tokens = count_tokens(chunk, pricing.name)
        total_tokens += tokens
        per_chunk.append({"chunk_index": i, "tokens": tokens, "chars": len(chunk)})

    cost_per_call = total_tokens * pricing.input_per_token

    return {
        "model": pricing.name,
        "num_chunks": len(chunks),
        "total_tokens": total_tokens,
        "cost_per_call": round(cost_per_call, 6),
        "per_chunk": per_chunk,
    }


def estimate_agent_turn_cost(
    context_tokens: int,
    tool_definitions: list[ToolDefinition],
    tool_calls: list[dict],
    response_tokens: int,
    model: str | ModelPricing = "gpt-4.1",
) -> dict:
    """Estimate total cost for one agent turn with tool usage.

    An agent turn may involve:
    1. Initial call (context + tool defs → response or tool_call)
    2. One or more tool call round-trips
    3. Final response after tool results

    Args:
        context_tokens: Input tokens from conversation history
        tool_definitions: Registered tools (added to every call)
        tool_calls: List of {"name", "arguments_json", "result"} dicts
        response_tokens: Final assistant response tokens
        model: Model name or pricing
    """
    pricing = model if isinstance(model, ModelPricing) else get_model(model)

    # Tool definition overhead (present in every API call)
    tool_def_overhead = estimate_tool_definitions_overhead(tool_definitions, pricing)
    tool_def_tokens = tool_def_overhead["total_tokens"]
    num_api_calls = 1 + len(tool_calls)  # initial + one per tool result

    tool_def_total_cost = tool_def_tokens * pricing.input_per_token * num_api_calls

    # Initial context cost (present in every API call)
    context_total_cost = context_tokens * pricing.input_per_token * num_api_calls

    # Tool call round-trip costs
    tool_costs = []
    tool_call_total = 0.0
    accumulated_result_tokens = 0
    for tc in tool_calls:
        tc_cost = estimate_tool_call_cost(
            tc["name"], tc["arguments_json"], tc["result"], pricing
        )
        tool_call_total += tc_cost.total_cost
        accumulated_result_tokens += tc_cost.result_input_tokens
        tool_costs.append(tc_cost)

    # Final response
    response_cost = response_tokens * pricing.output_per_token

    total = tool_def_total_cost + context_total_cost + tool_call_total + response_cost

    return {
        "model": pricing.name,
        "num_api_calls": num_api_calls,
        "context_tokens_per_call": context_tokens,
        "tool_def_tokens_per_call": tool_def_tokens,
        "tool_def_total_cost": round(tool_def_total_cost, 6),
        "context_total_cost": round(context_total_cost, 6),
        "tool_calls": tool_costs,
        "tool_call_total_cost": round(tool_call_total, 6),
        "response_tokens": response_tokens,
        "response_cost": round(response_cost, 6),
        "total_cost": round(total, 6),
    }
