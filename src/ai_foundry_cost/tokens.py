"""Token counting and cost calculation."""

import tiktoken

from .pricing import ModelPricing, get_model


def count_tokens(text: str, model: str = "gpt-4.1") -> int:
    """Count tokens in a text string using tiktoken.

    Falls back to cl100k_base encoding for unknown models.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str | ModelPricing,
    cached_input_tokens: int = 0,
) -> dict:
    """Calculate the cost for a given number of input/output tokens.

    Returns a dict with:
        - input_cost: cost of non-cached input tokens
        - cached_input_cost: cost of cached input tokens
        - output_cost: cost of output tokens
        - total_cost: sum of all costs
        - breakdown: human-readable summary
    """
    pricing = model if isinstance(model, ModelPricing) else get_model(model)

    non_cached = max(0, input_tokens - cached_input_tokens)
    input_cost = non_cached * pricing.input_per_token
    cached_cost = cached_input_tokens * pricing.cached_input_per_token
    output_cost = output_tokens * pricing.output_per_token
    total = input_cost + cached_cost + output_cost

    return {
        "model": pricing.name,
        "input_tokens": non_cached,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "input_cost": round(input_cost, 6),
        "cached_input_cost": round(cached_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total, 6),
    }


def estimate_from_text(
    prompt: str,
    response: str,
    model: str = "gpt-4.1",
) -> dict:
    """Estimate cost from raw prompt and response text."""
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    result = calculate_cost(input_tokens, output_tokens, model)
    result["prompt_chars"] = len(prompt)
    result["response_chars"] = len(response)
    return result
