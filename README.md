# auto-aneki

Autonomous LLM pretraining research for consumer GPUs. Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for RTX 3070 (8GB VRAM).

Give an AI agent a small but real LLM training setup and let it experiment autonomously. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats.

## What changed from upstream

| Setting | autoresearch (H100) | auto-aneki (RTX 3070) |
|---|---|---|
| Attention | Flash Attention 3 | PyTorch SDPA |
| Context length | 2048 | 1024 |
| Model depth | 8 layers | 4 layers |
| Model dim | 512 (8*64) | 192 (4*48) |
| Params | ~50M | ~8M |
| Device batch | 128 | 16 |
| Total batch | 524K tokens | 65K tokens |
| Grad checkpointing | No | Yes |
| Eval tokens | 20M | 5M |

## Quick start

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data and train tokenizer (~2 min)
uv run prepare.py

# Run a single training experiment (~5 min)
uv run train.py
```

## Running the agent

Point Claude Code (or any agent) at this repo and prompt:

```
Read program.md and let's kick off a new experiment! Do the setup first.
```

The agent will autonomously iterate on `train.py`, running 5-minute experiments and keeping improvements.

## Project structure

```
prepare.py      - constants, data prep + runtime utilities (do not modify)
train.py        - model, optimizer, training loop (agent modifies this)
program.md      - agent instructions
pyproject.toml  - dependencies
```

## License

MIT
