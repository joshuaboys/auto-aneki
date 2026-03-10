# auto-aneki

This is an experiment to have the LLM do its own research, adapted for RTX 3070 (8GB VRAM).

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Hardware Profile

- GPU: NVIDIA RTX 3070 (8GB VRAM, Ampere sm_86)
- RAM: 30GB system memory
- VRAM is the binding constraint — every change must respect the 8GB ceiling

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM constraint (HARD):** Unlike the original H100 setup, VRAM is a **hard** constraint here. You have 8GB total and the system/driver uses ~500MB. Stay under ~7GB peak. If you OOM, reduce batch size or model size. The defaults are conservative — there's room to push the model larger with careful tuning.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**The first run**: Your very first run should always be to establish the baseline.

## RTX 3070-Specific Tuning Notes

Key differences from the H100 version that inform your experimentation:

- **No Flash Attention 3** — uses PyTorch SDPA instead. Already handled, no action needed.
- **Gradient checkpointing** is ON by default. Trades compute for memory. You can try turning it off if you shrink the model enough.
- **DEVICE_BATCH_SIZE=16** — this is conservative. You might be able to push to 24-32 depending on model size.
- **TOTAL_BATCH_SIZE=65K** — you can experiment with this. Larger = more gradient accumulation steps = slower iteration but potentially better per-step learning.
- **DEPTH=4, ASPECT_RATIO=48** — the model is ~8M params. There's likely headroom to go to DEPTH=6 or increase ASPECT_RATIO if you reduce batch size.
- **MAX_SEQ_LEN=1024** (in prepare.py, fixed) — half the original 2048.
- **bf16 autocast** is used throughout. RTX 3070 supports bf16 natively.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          1.234567
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     6500.2
mfu_percent:      15.80
total_tokens_M:   120.6
num_steps:        1800
num_params_M:     8.3
depth:            4
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 6.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix, fix it and re-run. If the idea itself is fundamentally broken, just skip it.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder. The loop runs until the human interrupts you, period.
