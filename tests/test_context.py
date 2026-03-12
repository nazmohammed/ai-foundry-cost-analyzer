"""Tests for context module."""

import pytest
from ai_foundry_cost.context import analyze_conversation, project_cost


class TestAnalyzeConversation:
    def test_simple_conversation(self):
        turns = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great, thanks for asking!"},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I don't have access to weather data."},
        ]
        result = analyze_conversation(turns, model="gpt-4.1")
        assert result["num_turns"] == 2
        assert len(result["turns"]) == 2
        assert result["total_cost"] > 0

    def test_context_grows(self):
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Tell me a story"},
            {"role": "assistant", "content": "Once upon a time there was a robot."},
        ]
        result = analyze_conversation(turns, model="gpt-4.1")
        # Second turn should have more context tokens than first
        assert result["turns"][1].context_tokens > result["turns"][0].context_tokens

    def test_with_system_prompt(self):
        turns = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = analyze_conversation(turns, model="gpt-4.1", system_prompt="You are a helpful assistant.")
        assert result["system_prompt_tokens"] > 0

    def test_cumulative_cost_increases(self):
        turns = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        result = analyze_conversation(turns, model="gpt-4.1")
        assert result["turns"][1].cumulative_cost > result["turns"][0].cumulative_cost

    def test_empty_conversation(self):
        result = analyze_conversation([], model="gpt-4.1")
        assert result["num_turns"] == 0
        assert result["total_cost"] == 0.0


class TestProjectCost:
    def test_basic_projection(self):
        result = project_cost(
            avg_user_tokens=100,
            avg_assistant_tokens=200,
            num_turns=5,
            model="gpt-4.1",
        )
        assert result["num_turns"] == 5
        assert len(result["turns"]) == 5
        assert result["total_cost"] > 0

    def test_cost_increases_per_turn(self):
        result = project_cost(
            avg_user_tokens=100,
            avg_assistant_tokens=200,
            num_turns=5,
        )
        costs = [t.total_cost for t in result["turns"]]
        # Each turn should cost more than the previous (growing context)
        for i in range(1, len(costs)):
            assert costs[i] > costs[i - 1]

    def test_with_system_prompt(self):
        without = project_cost(100, 200, 3, system_prompt_tokens=0)
        with_sys = project_cost(100, 200, 3, system_prompt_tokens=1000)
        assert with_sys["total_cost"] > without["total_cost"]

    def test_single_turn(self):
        result = project_cost(100, 200, 1)
        assert len(result["turns"]) == 1
        assert result["turns"][0].turn_number == 1

    def test_cheaper_model_is_cheaper(self):
        expensive = project_cost(100, 200, 5, model="gpt-4.1")
        cheap = project_cost(100, 200, 5, model="gpt-4.1-nano")
        assert cheap["total_cost"] < expensive["total_cost"]
