"""Azure AI Foundry model pricing data.

Prices are per 1 million tokens (USD) as of March 2026.
Source: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/

Pricing tiers:
- Input tokens:  What you send (prompt, system message, history, tool defs, RAG context)
- Output tokens: What the model generates (response, tool calls)
- Cached input:  Prompt prefix cache hits (available on some models)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a single model."""
    name: str
    input_per_1m: float       # USD per 1M input tokens
    output_per_1m: float      # USD per 1M output tokens
    cached_input_per_1m: float  # USD per 1M cached input tokens (0 if not supported)
    context_window: int       # Max tokens (input + output)
    max_output: int           # Max output tokens

    @property
    def input_per_token(self) -> float:
        return self.input_per_1m / 1_000_000

    @property
    def output_per_token(self) -> float:
        return self.output_per_1m / 1_000_000

    @property
    def cached_input_per_token(self) -> float:
        return self.cached_input_per_1m / 1_000_000


# Azure OpenAI model pricing registry
MODELS: dict[str, ModelPricing] = {
    "gpt-4.1": ModelPricing(
        name="gpt-4.1",
        input_per_1m=2.00,
        output_per_1m=8.00,
        cached_input_per_1m=0.50,
        context_window=1_047_576,
        max_output=32_768,
    ),
    "gpt-4.1-mini": ModelPricing(
        name="gpt-4.1-mini",
        input_per_1m=0.40,
        output_per_1m=1.60,
        cached_input_per_1m=0.10,
        context_window=1_047_576,
        max_output=32_768,
    ),
    "gpt-4.1-nano": ModelPricing(
        name="gpt-4.1-nano",
        input_per_1m=0.10,
        output_per_1m=0.40,
        cached_input_per_1m=0.025,
        context_window=1_047_576,
        max_output=32_768,
    ),
    "gpt-4o": ModelPricing(
        name="gpt-4o",
        input_per_1m=2.50,
        output_per_1m=10.00,
        cached_input_per_1m=1.25,
        context_window=128_000,
        max_output=16_384,
    ),
    "gpt-4o-mini": ModelPricing(
        name="gpt-4o-mini",
        input_per_1m=0.15,
        output_per_1m=0.60,
        cached_input_per_1m=0.075,
        context_window=128_000,
        max_output=16_384,
    ),
    "o3": ModelPricing(
        name="o3",
        input_per_1m=10.00,
        output_per_1m=40.00,
        cached_input_per_1m=2.50,
        context_window=200_000,
        max_output=100_000,
    ),
    "o3-mini": ModelPricing(
        name="o3-mini",
        input_per_1m=1.10,
        output_per_1m=4.40,
        cached_input_per_1m=0.275,
        context_window=200_000,
        max_output=100_000,
    ),
    "o4-mini": ModelPricing(
        name="o4-mini",
        input_per_1m=1.10,
        output_per_1m=4.40,
        cached_input_per_1m=0.275,
        context_window=200_000,
        max_output=100_000,
    ),
}


def get_model(name: str) -> ModelPricing:
    """Get pricing for a model by name. Raises KeyError if not found."""
    key = name.lower().strip()
    if key not in MODELS:
        available = ", ".join(sorted(MODELS.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODELS[key]


def list_models() -> list[ModelPricing]:
    """Return all available models sorted by input price."""
    return sorted(MODELS.values(), key=lambda m: m.input_per_1m)
