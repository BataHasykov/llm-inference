# llm-inference

Guidance for Claude Code when working in this repository.

## What this is

A benchmark harness that measures LLM inference optimizations on a single RTX 3090 24GB with plain HuggingFace `generate()`: KV-cache, weight quantization (bitsandbytes, AWQ/GPTQ on Triton and Marlin), speculative decoding (draft model, prompt-lookup), and static batching. It also measures quality for the lossy formats (MMLU, GSM8K, perplexity).

The experiments are complete. Results, methodology and limitations are in `README.md`. The repo is the experimental part of a master's thesis; the thesis text lives outside the repo (`thesis/` is gitignored).

## Environment

- Runs on Linux / WSL2 with CUDA. torch 2.11.0+cu128, transformers 5.8.1, Python 3.11–3.12.
- Package manager: `uv` (`uv sync`, then `source .venv/bin/activate`). Do not assume conda.
- Dependency versions in `pyproject.toml` are pinned to the ones the published results were measured with. Don't bump them without re-measuring.
- Marlin kernels need the CUDA toolkit: `export CUDA_HOME=/usr/local/cuda-12.8; export PATH=$CUDA_HOME/bin:$PATH`.

## Code layout

- `scripts/` is flat on purpose. There is one script per method (`benchmark_*.py`, `manual_kv_loop.py`) plus small shared helpers: `cli.py`, `data.py`, `modeling.py`, `timing.py`, `runner.py`, `summary.py`, `quality.py`, `env.py`.
- `run_matrix.py` runs each config in its own process. `aggregate.py` builds `summary_table.{csv,json}`, and `plots.py` / `plots_methods.py` draw the figures.
- `scripts/launchers/*.sh` are the long-run entry points. They `cd` to the repo root themselves.
- Results go to `results/<model>/`. The per-run summary `*.json` and the plots are committed; the per-prompt `*.jsonl` is gitignored.

## Conventions

- Speed runs: `--limit 40 --repeats 3 --max-new-tokens 256 --fixed-length`, greedy.
- Time with `torch.cuda.Event`, never wall-clock. Measure peak memory with `torch.cuda.max_memory_allocated()`.
- New configs must not overwrite existing results. Use a new `kind` suffix, as `*_marlin` does.
- Keep the style: simple, explicit functions and small files, with no frameworks or abstraction layers.

## Ask first

- Before installing heavy dependencies.
- Before downloading models larger than 5 GB.
- Before re-running long experiments, which can take hours of GPU time.
