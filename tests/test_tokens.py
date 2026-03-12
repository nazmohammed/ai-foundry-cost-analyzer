"""Tests for tokens module."""

import pytest
from ai_foundry_cost.tokens import count_tokens, calculate_cost, estimate_from_text


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = count_tokens(text)
        assert tokens > 50

    def test_unknown_model_falls_back(self):
        # Should not raise, falls back to cl100k_base
        tokens = count_tokens("test", model="unknown-model-xyz")
        assert tokens > 0


class TestCalculateCost:
    def test_basic_cost(self):
        result = calculate_cost(1000, 500, "gpt-4.1")
        assert result["model"] == "gpt-4.1"
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["input_cost"] > 0
        assert result["output_cost"] > 0
        assert result["total_cost"] == pytest.approx(
            result["input_cost"] + result["output_cost"] + result["cached_input_cost"]
        )

    def test_with_cached_tokens(self):
        result = calculate_cost(1000, 500, "gpt-4.1", cached_input_tokens=400)
        assert result["input_tokens"] == 600  # 1000 - 400
        assert result["cached_input_tokens"] == 400
        assert result["cached_input_cost"] > 0
        # Cached should be cheaper than full price
        full_result = calculate_cost(1000, 500, "gpt-4.1")
        assert result["total_cost"] < full_result["total_cost"]

    def test_zero_tokens(self):
        result = calculate_cost(0, 0, "gpt-4.1")
        assert result["total_cost"] == 0.0

    def test_accepts_model_pricing(self):
        from ai_foundry_cost.pricing import get_model
        pricing = get_model("gpt-4o")
        result = calculate_cost(1000, 500, pricing)
        assert result["model"] == "gpt-4o"

    def test_negative_cached_clamped(self):
        # cached > input should not produce negative non-cached
        result = calculate_cost(100, 50, "gpt-4.1", cached_input_tokens=200)
        assert result["input_tokens"] == 0  # max(0, 100-200)


class TestEstimateFromText:
    def test_basic_estimate(self):
        result = estimate_from_text("What is 2+2?", "4", "gpt-4.1")
        assert result["prompt_chars"] == len("What is 2+2?")
        assert result["response_chars"] == 1
        assert result["input_tokens"] > 0
        assert result["output_tokens"] > 0
        assert result["total_cost"] > 0
