"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner
from ai_foundry_cost.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestModelsCommand:
    def test_models_runs(self, runner):
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        assert "gpt-4.1" in result.output

    def test_models_shows_pricing(self, runner):
        result = runner.invoke(cli, ["models"])
        assert "$" in result.output


class TestCalculateCommand:
    def test_basic_calculate(self, runner):
        result = runner.invoke(cli, ["calculate", "-i", "1000", "-o", "500"])
        assert result.exit_code == 0
        assert "$" in result.output

    def test_with_model(self, runner):
        result = runner.invoke(cli, ["calculate", "-m", "gpt-4o", "-i", "1000", "-o", "500"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_with_cached(self, runner):
        result = runner.invoke(cli, ["calculate", "-i", "1000", "-o", "500", "-c", "400"])
        assert result.exit_code == 0


class TestEstimateCommand:
    def test_basic_estimate(self, runner):
        result = runner.invoke(cli, ["estimate", "-p", "Hello world", "-r", "Hi there!"])
        assert result.exit_code == 0
        assert "tokens" in result.output

    def test_from_file(self, runner, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("What is machine learning?")
        response_file = tmp_path / "response.txt"
        response_file.write_text("Machine learning is a subset of AI.")
        result = runner.invoke(cli, [
            "estimate", "-p", f"@{prompt_file}", "-r", f"@{response_file}"
        ])
        assert result.exit_code == 0


class TestContextCommand:
    def test_basic_context(self, runner):
        result = runner.invoke(cli, ["context", "-t", "5"])
        assert result.exit_code == 0
        assert "Cumulative" in result.output or "cumulative" in result.output.lower()

    def test_with_options(self, runner):
        result = runner.invoke(cli, [
            "context", "-t", "3", "--user-tokens", "200",
            "--assistant-tokens", "400", "--system-tokens", "1000"
        ])
        assert result.exit_code == 0


class TestMultiAgentCommand:
    def test_basic_multi_agent(self, runner):
        result = runner.invoke(cli, ["multi-agent"])
        assert result.exit_code == 0
        assert "Total" in result.output or "total" in result.output.lower()

    def test_with_options(self, runner):
        result = runner.invoke(cli, [
            "multi-agent", "--agents", "3", "--tools-per-agent", "5",
            "--tool-calls", "2"
        ])
        assert result.exit_code == 0


class TestCompareCommand:
    def test_basic_compare(self, runner):
        result = runner.invoke(cli, ["compare"])
        assert result.exit_code == 0
        assert "multiplier" in result.output.lower() or "x" in result.output

    def test_with_options(self, runner):
        result = runner.invoke(cli, [
            "compare", "-i", "1000", "-o", "500", "--agents", "3"
        ])
        assert result.exit_code == 0


class TestCountCommand:
    def test_basic_count(self, runner):
        result = runner.invoke(cli, ["count", "Hello world this is a test"])
        assert result.exit_code == 0
        assert "tokens" in result.output

    def test_from_file(self, runner, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Sample text for token counting.")
        result = runner.invoke(cli, ["count", f"@{f}"])
        assert result.exit_code == 0
