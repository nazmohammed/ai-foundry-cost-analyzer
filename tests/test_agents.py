"""Tests for multi-agent simulation."""

import pytest
from ai_foundry_cost.agents import (
    AgentConfig,
    simulate_single_agent,
    simulate_multi_agent_request,
    compare_single_vs_multi,
)
from ai_foundry_cost.tools_cost import ToolDefinition


SAMPLE_TOOLS = [
    ToolDefinition("lookup_booking", "Look up a booking by reference", '{"type": "object", "properties": {"ref": {"type": "string"}}}'),
    ToolDefinition("check_loyalty", "Check loyalty tier and points", '{"type": "object", "properties": {"passenger_id": {"type": "string"}}}'),
]


class TestSimulateSingleAgent:
    def test_basic_agent(self):
        agent = AgentConfig(name="TestAgent", model="gpt-4.1")
        result = simulate_single_agent(agent)
        assert result.agent_name == "TestAgent"
        assert result.total_cost > 0
        assert result.input_tokens > 0
        assert result.output_tokens > 0

    def test_agent_with_tools(self):
        without_tools = AgentConfig(name="NoTools", model="gpt-4.1")
        with_tools = AgentConfig(
            name="WithTools",
            model="gpt-4.1",
            tools=SAMPLE_TOOLS,
            avg_tool_calls_per_turn=2,
        )
        r_without = simulate_single_agent(without_tools)
        r_with = simulate_single_agent(with_tools)
        assert r_with.total_cost > r_without.total_cost

    def test_shared_context_increases_cost(self):
        agent = AgentConfig(name="Test", model="gpt-4.1")
        r_no_ctx = simulate_single_agent(agent, shared_context_tokens=0)
        r_with_ctx = simulate_single_agent(agent, shared_context_tokens=2000)
        assert r_with_ctx.total_cost > r_no_ctx.total_cost

    def test_cheaper_model(self):
        expensive = AgentConfig(name="A", model="gpt-4.1")
        cheap = AgentConfig(name="B", model="gpt-4.1-nano")
        r_exp = simulate_single_agent(expensive)
        r_cheap = simulate_single_agent(cheap)
        assert r_cheap.total_cost < r_exp.total_cost


class TestMultiAgentRequest:
    def setup_method(self):
        self.orchestrator = AgentConfig(
            name="Orchestrator",
            model="gpt-4.1",
            system_prompt_tokens=800,
            avg_response_tokens=100,
        )
        self.sub_agents = [
            AgentConfig(name="CustomerService", model="gpt-4.1", tools=SAMPLE_TOOLS, avg_tool_calls_per_turn=1),
            AgentConfig(name="Loyalty", model="gpt-4.1-mini"),
            AgentConfig(name="Operations", model="gpt-4.1"),
        ]

    def test_single_delegation(self):
        result = simulate_multi_agent_request(
            self.orchestrator, self.sub_agents, ["CustomerService"]
        )
        assert result["total_cost"] > 0
        assert len(result["sub_agents"]) == 1
        assert result["sub_agents"][0].agent_name == "CustomerService"

    def test_multi_delegation(self):
        one = simulate_multi_agent_request(
            self.orchestrator, self.sub_agents, ["CustomerService"]
        )
        two = simulate_multi_agent_request(
            self.orchestrator, self.sub_agents, ["CustomerService", "Loyalty"]
        )
        assert two["total_cost"] > one["total_cost"]
        assert len(two["sub_agents"]) == 2

    def test_unknown_agent_skipped(self):
        result = simulate_multi_agent_request(
            self.orchestrator, self.sub_agents, ["NonExistent"]
        )
        assert len(result["sub_agents"]) == 0

    def test_cost_breakdown_sums(self):
        result = simulate_multi_agent_request(
            self.orchestrator, self.sub_agents, ["CustomerService", "Loyalty"]
        )
        breakdown = result["cost_breakdown"]
        component_sum = (
            breakdown["orchestrator_routing"] +
            breakdown["sub_agent_processing"] +
            breakdown["orchestrator_synthesis"]
        )
        assert result["total_cost"] == pytest.approx(component_sum, abs=0.000001)

    def test_api_call_count(self):
        result = simulate_multi_agent_request(
            self.orchestrator, self.sub_agents, ["CustomerService", "Operations"]
        )
        # 1 orch + 2 sub-agents + 1 synthesis = 4
        assert result["num_api_calls"] == 4


class TestCompareSingleVsMulti:
    def test_multi_is_more_expensive(self):
        orch = AgentConfig(name="Orch", model="gpt-4.1", avg_response_tokens=100)
        subs = [AgentConfig(name="A1", model="gpt-4.1", avg_response_tokens=200)]
        result = compare_single_vs_multi(
            task_tokens=500,
            response_tokens=300,
            single_model="gpt-4.1",
            orchestrator=orch,
            sub_agents=subs,
            delegated=["A1"],
        )
        assert result["cost_multiplier"] > 1.0
        assert result["additional_cost"] > 0

    def test_without_multi_agent(self):
        result = compare_single_vs_multi(500, 300, "gpt-4.1")
        assert result["single_agent_cost"] > 0
        assert result["multi_agent_cost"] == 0.0
