# Diffusion Text Generation

This directory contains implementations for Diffusion LLMs (DLLMs)

More Info:
- https://github.com/ggml-org/llama.cpp/pull/14644
- https://github.com/ggml-org/llama.cpp/pull/14771

Targets (all entered by `examples/CMakeLists.txt` -> `add_subdirectory(diffusion)`):
- `llama-diffusion` (static library): `diffusion_generate` (masked path: Dream / LLaDA / RND1, plus
  DiffusionGemma's masked-path extensions `self_conditioning` and `suppress_mask_token`) and
  `diffusion_generate_entropy_bound` (the DiffusionGemma entropy-bound canvas denoiser). Linked by
  `llama-diffusion-cli`, `llama-diffusion-gemma-visual-server` (`examples/diffusion-gemma-server/`) and the self-test.
- `llama-diffusion-cli`
- `llama-diffusion-eb-selftest` (CPU self-test against a random-weight model; see `tests/diffusion-gemma-check/`)

## Parameters
The diffusion CLI supports various parameters to control the generation process:

### Core Diffusion Parameters
- `--diffusion-steps`: Number of diffusion steps (default: 128)
- `--diffusion-algorithm`: Algorithm for token selection
  - `0`: DIFFUSION_ALGORITHM_ORIGIN - Token will be generated in a purely random order from https://arxiv.org/abs/2107.03006.
  - `1`: DIFFUSION_ALGORITHM_ENTROPY_BASED - Entropy-based selection
  - `2`: DIFFUSION_ALGORITHM_MARGIN_BASED - Margin-based selection
  - `3`: DIFFUSION_ALGORITHM_RANDOM - Random selection
  - `4`: DIFFUSION_ALGORITHM_CONFIDENCE_BASED - Confidence-based selection (default)
  - More documentation here https://github.com/DreamLM/Dream
- `--diffusion-visual`: Enable live visualization during generation
- `--diffusion-visual-progress`, `--diffusion-visual-interval N`: progress bar / redraw every Nth step

### Scheduling Parameters
Choose one of the following scheduling methods (both default to 0; the CLI asserts exactly one is set for
non-canvas models):

**Timestep-based scheduling:**
- `--diffusion-eps`: Epsilon value for timestep scheduling (e.g., 0.001)

**Block-based scheduling:**
- `--diffusion-block-length`: Block size for block-based scheduling (e.g., 32)

### Sampling Parameters
- `--temp`: Temperature for sampling (0.0 = greedy/deterministic, higher = more random)
- `--top-k`: Top-k filtering for sampling
- `--top-p`: Top-p (nucleus) filtering for sampling
- `--seed`: Random seed for reproducibility

### Model Parameters
- `-m`: Path to the GGUF model file
- `-p`: Input prompt text
- `-ub`: Maximum sequence length (ubatch size)
- `-c`: Context size
- `-b`: Batch size

### DiffusionGemma (canvas / block diffusion)
A model with `diffusion.canvas_length` in its GGUF is denoised block by block: the prompt is followed by a canvas
of that many positions, and `-n N` derives the block count and grows `-ub/-b/-c` to fit. The entropy-bound decoder
is used automatically (`--diffusion-eb off` forces the masked path). Its parameters come from the GGUF
(`diffusion.eb_max_steps`, `eb_t_min`, `eb_t_max`, `eb_entropy_bound`, `eb_stability_threshold`,
`eb_confidence_threshold`), then the reference defaults, then these command-line overrides:
`--diffusion-eb-max-steps`, `--diffusion-eb-t-min`, `--diffusion-eb-t-max`, `--diffusion-eb-entropy-bound`,
`--diffusion-eb-stability`, `--diffusion-eb-confidence`. Performance switches: `--diffusion-kv-cache`,
`--diffusion-gpu-sampling`, `--diffusion-gpu-sample-reduce` (each `auto|on|off`; auto is on for a single device).
`--diffusion-blocks N` caps block-autoregressive blocks; `-cnv` gives a multi-turn prompt.

`llama-diffusion-gemma-visual-server <model.gguf>` (env `NGL`, `MAXTOK`, `FA`) keeps the model loaded and streams the
per-step canvas over a line protocol on stdin/stdout (`READY`, `F`, `C`, `STATS`, `DONE`, `ERR`); the protocol is
documented at the top of `examples/diffusion-gemma-server/diffusion-gemma-visual-server.cpp`.

### Examples
#### Dream architecture:
```
llama-diffusion-cli -m dream7b.gguf -p "write code to train MNIST in pytorch" -ub 512 --diffusion-eps 0.001 --diffusion-algorithm 3 --diffusion-steps 256 --diffusion-visual
```

#### LLaDA architecture:
```
llama-diffusion-cli -m llada-8b.gguf -p "write code to train MNIST in pytorch" -ub 512 --diffusion-block-length 32 --diffusion-steps 256 --diffusion-visual
```

#### RND1 architecture:
```
llama-diffusion-cli -m RND1-Base-0910.gguf -p "write code to train MNIST in pytorch" -ub 512 --diffusion-algorithm 1 --diffusion-steps 256 --diffusion-visual --temp 0.5 --diffusion-eps 0.001
```

#### DiffusionGemma:
```
llama-diffusion-cli -m diffusion-gemma.gguf -p "write a haiku about rain" -n 256 --diffusion-visual
```
