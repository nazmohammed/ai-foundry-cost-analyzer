"""Tests for pricing module."""

import pytest
from ai_foundry_cost.pricing import get_model, list_models, MODELS, ModelPricing


class TestModelPricing:
    def test_input_per_token(self):
        m = ModelPricing("test", 2.0, 8.0, 0.5, 128000, 4096)
        assert m.input_per_token == pytest.approx(2.0 / 1_000_000)

    def test_output_per_token(self):
        m = ModelPricing("test", 2.0, 8.0, 0.5, 128000, 4096)
        assert m.output_per_token == pytest.approx(8.0 / 1_000_000)

    def test_cached_input_per_token(self):
        m = ModelPricing("test", 2.0, 8.0, 0.5, 128000, 4096)
        assert m.cached_input_per_token == pytest.approx(0.5 / 1_000_000)

    def test_frozen(self):
        m = ModelPricing("test", 2.0, 8.0, 0.5, 128000, 4096)
        with pytest.raises(AttributeError):
            m.name = "changed"


class TestGetModel:
    def test_known_model(self):
        m = get_model("gpt-4.1")
        assert m.name == "gpt-4.1"
        assert m.input_per_1m == 2.00

    def test_case_insensitive(self):
        m = get_model("GPT-4.1")
        assert m.name == "gpt-4.1"

    def test_strips_whitespace(self):
        m = get_model("  gpt-4.1  ")
        assert m.name == "gpt-4.1"

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("gpt-99")


class TestListModels:
    def test_returns_all(self):
        models = list_models()
        assert len(models) == len(MODELS)

    def test_sorted_by_input_price(self):
        models = list_models()
        prices = [m.input_per_1m for m in models]
        assert prices == sorted(prices)

    def test_contains_gpt4_1(self):
        names = [m.name for m in list_models()]
        assert "gpt-4.1" in names
