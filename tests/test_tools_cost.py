"""Tests for tools_cost module."""

import pytest
from ai_foundry_cost.tools_cost import (
    ToolDefinition,
    estimate_tool_definitions_overhead,
    estimate_tool_call_cost,
    estimate_rag_cost,
    estimate_agent_turn_cost,
)


SAMPLE_TOOLS = [
    ToolDefinition(
        name="get_weather",
        description="Get the current weather for a location",
        parameters_json='{"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}',
    ),
    ToolDefinition(
        name="search_flights",
        description="Search for available flights between two airports",
        parameters_json='{"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination"]}',
    ),
]


class TestToolDefinitionsOverhead:
    def test_basic_overhead(self):
        result = estimate_tool_definitions_overhead(SAMPLE_TOOLS)
        assert result["num_tools"] == 2
        assert result["total_tokens"] > 0
        assert result["cost_per_call"] > 0

    def test_empty_tools(self):
        result = estimate_tool_definitions_overhead([])
        assert result["total_tokens"] == 0
        assert result["cost_per_call"] == 0.0

    def test_per_tool_breakdown(self):
        result = estimate_tool_definitions_overhead(SAMPLE_TOOLS)
        assert len(result["per_tool"]) == 2
        total = sum(t["tokens"] for t in result["per_tool"])
        assert total == result["total_tokens"]

    def test_more_tools_more_tokens(self):
        one = estimate_tool_definitions_overhead(SAMPLE_TOOLS[:1])
        two = estimate_tool_definitions_overhead(SAMPLE_TOOLS)
        assert two["total_tokens"] > one["total_tokens"]


class TestToolCallCost:
    def test_basic_call(self):
        result = estimate_tool_call_cost(
            tool_name="get_weather",
            call_arguments_json='{"location": "London"}',
            tool_result='{"temp": 15, "condition": "cloudy"}',
        )
        assert result.tool_name == "get_weather"
        assert result.call_output_tokens > 0
        assert result.result_input_tokens > 0
        assert result.total_cost > 0

    def test_larger_result_costs_more(self):
        small = estimate_tool_call_cost("fn", '{"a": 1}', "ok")
        large = estimate_tool_call_cost("fn", '{"a": 1}', "x" * 1000)
        assert large.result_input_tokens > small.result_input_tokens
        assert large.total_cost > small.total_cost


class TestRagCost:
    def test_basic_rag(self):
        chunks = [
            "The airline operates daily flights between London and Paris.",
            "Premium economy passengers receive priority boarding and extra legroom.",
        ]
        result = estimate_rag_cost(chunks)
        assert result["num_chunks"] == 2
        assert result["total_tokens"] > 0
        assert result["cost_per_call"] > 0

    def test_empty_chunks(self):
        result = estimate_rag_cost([])
        assert result["total_tokens"] == 0

    def test_per_chunk_breakdown(self):
        chunks = ["chunk one", "chunk two that is longer"]
        result = estimate_rag_cost(chunks)
        assert len(result["per_chunk"]) == 2
        assert result["per_chunk"][1]["tokens"] >= result["per_chunk"][0]["tokens"]


class TestAgentTurnCost:
    def test_basic_turn(self):
        result = estimate_agent_turn_cost(
            context_tokens=500,
            tool_definitions=SAMPLE_TOOLS,
            tool_calls=[
                {
                    "name": "get_weather",
                    "arguments_json": '{"location": "London"}',
                    "result": '{"temp": 15}',
                }
            ],
            response_tokens=200,
        )
        assert result["num_api_calls"] == 2  # initial + after tool result
        assert result["total_cost"] > 0

    def test_no_tools(self):
        result = estimate_agent_turn_cost(
            context_tokens=500,
            tool_definitions=[],
            tool_calls=[],
            response_tokens=200,
        )
        assert result["num_api_calls"] == 1
        assert result["tool_call_total_cost"] == 0.0
