# AI Foundry Cost Analyzer

**Understand exactly how Azure AI Foundry bills you** — token by token, turn by turn, agent by agent.

A Python CLI tool that breaks down LLM costs across the dimensions that matter: input/output tokens, context window growth, tool definitions, RAG grounding, and multi-agent orchestration overhead.

## Why This Exists

Azure AI Foundry charges per token, but the real cost drivers are hidden:

- **Context window inflation** — Every conversation turn resends the entire history as input tokens. A 10-turn chat doesn't cost 10× a single turn; it costs ~55× due to the growing context.
- **Tool definition tax** — Every registered function's JSON schema is injected into every API call. 10 tools × 50 tokens each = 500 extra input tokens per call.
- **Tool call round-trips** — Each tool invocation generates output tokens (the function call) AND input tokens (the result sent back).
- **Multi-agent multipliers** — An orchestrator + 3 sub-agents means 5+ API calls for a single user message, each carrying its own context.

This tool makes all of that visible and quantifiable.

## Installation

```bash
pip install ai-foundry-cost-analyzer
```

Or from source:

```bash
git clone https://github.com/nazmohammed/ai-foundry-cost-analyzer.git
cd ai-foundry-cost-analyzer
pip install -e .
```

## Quick Start

### List model pricing

```bash
ai-cost models
```

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                        Azure AI Foundry Model Pricing                             │
├──────────────┬──────────────────┬───────────────────┬──────────────┬──────────────┤
│ Model        │ Input / 1M tokens│ Output / 1M tokens│ Context      │ Max Output   │
├──────────────┼──────────────────┼───────────────────┼──────────────┼──────────────┤
│ gpt-4.1-nano │ $0.10            │ $0.40             │ 1,047,576    │ 32,768       │
│ gpt-4.1-mini │ $0.40            │ $1.60             │ 1,047,576    │ 32,768       │
│ gpt-4.1      │ $2.00            │ $8.00             │ 1,047,576    │ 32,768       │
│ gpt-4o       │ $2.50            │ $10.00            │ 128,000      │ 16,384       │
│ ...          │                  │                   │              │              │
└──────────────┴──────────────────┴───────────────────┴──────────────┴──────────────┘
```

### Calculate token costs

```bash
# Direct token counts
ai-cost calculate -i 5000 -o 1000 -m gpt-4.1

# From text (supports @file)
ai-cost estimate -p "What flights are available to London?" -r "I found 3 flights..."

# Count tokens in a file
ai-cost count @my_prompt.txt
```

### Analyze context window cost escalation

```bash
ai-cost context -t 10 --user-tokens 150 --assistant-tokens 300 --system-tokens 500
```

Shows how cost grows per turn as the context window fills:

| Turn | Context Tokens | Output Tokens | Turn Cost | Cumulative |
|------|---------------|---------------|-----------|------------|
| 1    | 650           | 300           | $0.0037   | $0.0037    |
| 2    | 1,100         | 300           | $0.0046   | $0.0083    |
| 5    | 2,450         | 300           | $0.0073   | $0.0247    |
| 10   | 4,700         | 300           | $0.0118   | $0.0652    |

### Simulate multi-agent costs

```bash
# 3 sub-agents, 5 tools each, 2 tool calls per turn
ai-cost multi-agent --agents 3 --tools-per-agent 5 --tool-calls 2
```

### Compare single vs multi-agent

```bash
ai-cost compare -i 500 -o 300 --agents 3
```

## Supported Models

| Model | Input/1M | Output/1M | Context Window |
|-------|----------|-----------|----------------|
| gpt-4.1-nano | $0.10 | $0.40 | 1,047,576 |
| gpt-4.1-mini | $0.40 | $1.60 | 1,047,576 |
| gpt-4.1 | $2.00 | $8.00 | 1,047,576 |
| gpt-4o-mini | $0.15 | $0.60 | 128,000 |
| gpt-4o | $2.50 | $10.00 | 128,000 |
| o3-mini | $1.10 | $4.40 | 200,000 |
| o3 | $10.00 | $40.00 | 200,000 |
| o4-mini | $1.10 | $4.40 | 200,000 |

## How LLM Billing Works

### Token basics
- **Input tokens**: Everything you send — system prompt, conversation history, tool definitions, RAG context, user message
- **Output tokens**: Everything the model generates — response text, tool calls (function name + JSON arguments)
- **Cached input tokens**: Prompt prefix cache hits (discounted rate, available on some models)

### The context window trap
In a multi-turn conversation, the full history is resent on every API call:

```
Turn 1: [system + user_1]                    → 650 input tokens
Turn 2: [system + user_1 + asst_1 + user_2]  → 1,100 input tokens
Turn 5: [system + ... + user_5]              → 2,450 input tokens
Turn 10: [system + ... + user_10]            → 4,700 input tokens
```

Total input tokens for a 10-turn chat: **~27,500** (not 6,500).

### Tool usage costs
```
Per API call:
  + tool_definition_tokens × num_tools   (input, every call)
  + tool_call_output_tokens              (output, when model calls a tool)
  + tool_result_tokens                   (input, on the follow-up call)
```

### Multi-agent overhead
```
User message → Orchestrator (classify + route)
            → Sub-Agent 1 (process + tools)
            → Sub-Agent 2 (process + tools)
            → Orchestrator (synthesize response)

= 4+ API calls × (context + tool defs + response)
```

## Python API

```python
from ai_foundry_cost.tokens import calculate_cost, estimate_from_text
from ai_foundry_cost.context import project_cost
from ai_foundry_cost.agents import AgentConfig, simulate_multi_agent_request

# Quick cost calculation
result = calculate_cost(input_tokens=5000, output_tokens=1000, model="gpt-4.1")
print(f"Cost: ${result['total_cost']:.4f}")

# Project 10-turn conversation cost
conv = project_cost(avg_user_tokens=150, avg_assistant_tokens=300, num_turns=10)
print(f"10-turn cost: ${conv['total_cost']:.4f}")
```

## Development

```bash
git clone https://github.com/nazmohammed/ai-foundry-cost-analyzer.git
cd ai-foundry-cost-analyzer
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
