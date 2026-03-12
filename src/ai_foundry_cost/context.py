"""Context window cost analysis.

In multi-turn conversations, the entire conversation history is resent
as input tokens on every turn.  This module models how costs escalate
as the context window fills up.
"""

from dataclasses import dataclass

from .pricing import ModelPricing, get_model
from .tokens import count_tokens


@dataclass
class Turn:
    """One turn in a conversation."""
    role: str            # "user" or "assistant"
    content: str
    token_count: int = 0  # populated during analysis


@dataclass
class TurnCost:
    """Cost breakdown for a single API call (one assistant turn)."""
    turn_number: int
    context_tokens: int   # all previous turns re-sent as input
    output_tokens: int    # tokens generated this turn
    input_cost: float
    output_cost: float
    total_cost: float
    cumulative_cost: float


def analyze_conversation(
    turns: list[dict],
    model: str | ModelPricing = "gpt-4.1",
    system_prompt: str = "",
) -> dict:
    """Analyze cost escalation across a multi-turn conversation.

    Args:
        turns: List of {"role": "user"|"assistant", "content": "..."}
        model: Model name or ModelPricing instance
        system_prompt: System message prepended to every API call

    Returns dict with:
        - turns: list of TurnCost for each assistant response
        - total_cost: total cost of the entire conversation
        - total_input_tokens: sum of all input tokens across all calls
        - total_output_tokens: sum of all output tokens
        - context_growth: how input tokens grew per turn
    """
    pricing = model if isinstance(model, ModelPricing) else get_model(model)
    model_name = pricing.name

    # Tokenize everything
    system_tokens = count_tokens(system_prompt, model_name) if system_prompt else 0
    parsed: list[Turn] = []
    for t in turns:
        content = t["content"]
        parsed.append(Turn(
            role=t["role"],
            content=content,
            token_count=count_tokens(content, model_name),
        ))

    results: list[TurnCost] = []
    cumulative = 0.0
    turn_num = 0

    for i, turn in enumerate(parsed):
        if turn.role != "assistant":
            continue

        turn_num += 1

        # Context = system prompt + all turns up to (but not including) this one
        context_tokens = system_tokens + sum(
            t.token_count for t in parsed[:i]
        )
        output_tokens = turn.token_count

        input_cost = context_tokens * pricing.input_per_token
        output_cost = output_tokens * pricing.output_per_token
        total = input_cost + output_cost
        cumulative += total

        results.append(TurnCost(
            turn_number=turn_num,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(total, 6),
            cumulative_cost=round(cumulative, 6),
        ))

    total_input = sum(tc.context_tokens for tc in results)
    total_output = sum(tc.output_tokens for tc in results)

    return {
        "model": pricing.name,
        "system_prompt_tokens": system_tokens,
        "num_turns": turn_num,
        "turns": results,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost": round(cumulative, 6),
    }


def project_cost(
    avg_user_tokens: int,
    avg_assistant_tokens: int,
    num_turns: int,
    model: str | ModelPricing = "gpt-4.1",
    system_prompt_tokens: int = 0,
) -> dict:
    """Project cost for a conversation of N turns with average token counts.

    Each API call sends system_prompt + all previous turns as context,
    so input tokens grow linearly with turn count.

    Returns dict with per-turn breakdown and totals.
    """
    pricing = model if isinstance(model, ModelPricing) else get_model(model)
    avg_turn_tokens = avg_user_tokens + avg_assistant_tokens

    results: list[TurnCost] = []
    cumulative = 0.0

    for n in range(1, num_turns + 1):
        # Context for turn n = system + (n-1) full roundtrips + current user msg
        context_tokens = system_prompt_tokens + (n - 1) * avg_turn_tokens + avg_user_tokens
        output_tokens = avg_assistant_tokens

        input_cost = context_tokens * pricing.input_per_token
        output_cost = output_tokens * pricing.output_per_token
        total = input_cost + output_cost
        cumulative += total

        results.append(TurnCost(
            turn_number=n,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(total, 6),
            cumulative_cost=round(cumulative, 6),
        ))

    return {
        "model": pricing.name,
        "num_turns": num_turns,
        "avg_user_tokens": avg_user_tokens,
        "avg_assistant_tokens": avg_assistant_tokens,
        "system_prompt_tokens": system_prompt_tokens,
        "turns": results,
        "total_input_tokens": sum(tc.context_tokens for tc in results),
        "total_output_tokens": sum(tc.output_tokens for tc in results),
        "total_cost": round(cumulative, 6),
    }
