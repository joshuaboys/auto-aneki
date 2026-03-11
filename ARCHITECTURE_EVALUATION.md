# Architecture Evaluation: auto-aneki as a base for domain-specific spikes

## Verdict: Strong base with a few gaps to address

The architecture is clean, minimal, and well-thought-out. The separation between
immutable infrastructure (`prepare.py`) and the experimentation surface (`train.py`)
is the right call.

## What works well as a base

### 1. Clean two-file architecture
`prepare.py` owns data, tokenization, and evaluation. `train.py` owns model and
training. This separation is the core abstraction — domain variants swap the "what"
(data/tokenizer/eval) while the agent iterates on the "how" (architecture/hyperparams).
The boundary is right.

### 2. The autonomous loop design is domain-agnostic
The git-branch-per-experiment, commit-run-measure-keep/discard cycle (`program.md`)
doesn't depend on what's being trained. It's a general-purpose autonomous research
protocol. Domain spikes inherit this for free.

### 3. Hardware-aware defaults
Gradient checkpointing, bf16 autocast, VRAM-conscious batch sizing, `torch.compile` —
these are consumer-GPU realities that transfer across domains. The
`PYTORCH_ALLOC_CONF=expandable_segments:True` trick is a nice touch.

### 4. BPB as the metric
Bits-per-byte is vocabulary-size-independent, which means domain spikes with different
tokenizers still produce comparable numbers.

### 5. Muon+AdamW optimizer split
Using Muon for 2D matrices and AdamW for embeddings/scalars is a meaningful optimization
that transfers across model architectures. The `@torch.compile` fused kernels are
production-quality.

## What needs work before spiking

### 1. `prepare.py` is monolithic — it's both library and script

It serves three roles: (a) download data, (b) train tokenizer, (c) provide runtime
utilities (`Tokenizer`, `make_dataloader`, `evaluate_bpb`). For domain spikes, you'd
need to change (a) and (b) but keep (c). Right now that means forking the entire file.

**Recommendation:** Split into:
- `prepare.py` → domain-specific setup script (data download + tokenizer training)
- `runtime.py` → shared runtime utilities (Tokenizer class, make_dataloader, evaluate_bpb, constants)

Domain spikes only replace `prepare.py` and inherit the runtime.

### 2. Dataset source is hardcoded

`BASE_URL`, `MAX_SHARD`, `VAL_SHARD`, `VAL_FILENAME` are all constants pointing to
`karpathy/climbmix-400b-shuffle`. A domain spike for code or medical text needs
different data. Making the data source configurable (even just a different constants
block per domain) reduces fork divergence.

### 3. No abstraction for "what to measure"

`evaluate_bpb` is the single metric. Some domains might want additional metrics
(e.g., code completion accuracy, perplexity on a domain-specific test set) alongside
BPB. The agent playbook in `program.md` is hardcoded to parse `val_bpb:` from logs.

### 4. Tokenizer is tightly coupled to general text

`SPLIT_PATTERN` uses GPT-4-style regex. Code, structured data, or non-Latin-script
domains benefit from different split patterns. `VOCAB_SIZE = 8192` is very small —
fine for general text experiments but potentially limiting for code. These are easy
to change per-spike but are the primary configuration surface.

### 5. No configuration file — everything is inline constants

Intentional (simplicity) and works for a single variant, but domain spikes create
unnecessary diffs. A minimal `domain.py` or config that the agent can't touch would
let domain variants be expressed as config changes rather than code forks.

### 6. No test harness

The framework relies on "run it and see if it crashes." A quick smoke test
(e.g., 10-step training on synthetic data, assert loss decreases) would catch breakage
from domain-specific changes before burning 5 minutes on a real run.

## Recommended spike approach

For each domain variant:

```
auto-aneki/                    # base (this repo)
auto-aneki-code/               # domain spike
├── prepare.py                 # new: code data source, code-aware tokenizer
├── train.py                   # inherited, maybe different defaults
├── program.md                 # inherited, maybe domain-specific hints
└── pyproject.toml             # inherited
```

Minimal changes per spike:
1. **Data source** — point to a domain dataset (e.g., The Stack for code)
2. **Tokenizer config** — adjust `SPLIT_PATTERN`, `VOCAB_SIZE` for the domain
3. **Model defaults** — tune `DEPTH`, `ASPECT_RATIO`, `DEVICE_BATCH_SIZE`
4. **Agent hints** — add domain-specific notes to `program.md`

## Summary scores

| Dimension | Score | Notes |
|---|---|---|
| Modularity | 7/10 | Clean 2-file split, but prepare.py mixes concerns |
| Extensibility | 6/10 | No plugin points — you fork, not extend |
| Domain portability | 7/10 | BPB metric transfers; data/tokenizer need replacement |
| Agent protocol | 9/10 | Git-based experiment loop is domain-agnostic |
| Hardware abstraction | 8/10 | Good VRAM management, portable across consumer GPUs |
| Simplicity | 9/10 | Intentionally minimal — this is a feature |
| Test infrastructure | 3/10 | No automated tests, relies on runtime success |

## Bottom line

This is a solid 5-minute-experiment chassis. The main refactor before spiking is
splitting `prepare.py` into "domain setup" vs "shared runtime" so domain variants
don't have to fork evaluation infrastructure. Everything else is config-level changes
per spike.
