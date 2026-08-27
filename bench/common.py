"""Model shapes, variants and the shared plumbing of the dense memory/time sweep.

The variant names are deliberately the ones the MoE half of this benchmark uses
(`stage3_moe.ARMS` in H-MoE-Part-cloud), so the two tables can be joined on the
variant column without a translation layer.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from pathlib import Path

# Every shape is the repo's own Llama: MHA, SwiGLU whose hidden width is rounded up
# to `multiple_of`, RMSNorm, untied lm_head, no biases. `parameters_expected` is that
# arithmetic evaluated ahead of time, and the worker refuses to measure a model that
# does not hit it -- a geometry typo is otherwise invisible in a table of timings.
MODEL_SPECS = (
    {
        "name": "257m",
        "label": "257M",
        "n_layer": 12,
        "n_embd": 1024,
        "n_head": 8,
        "parameters_expected": 257_188_864,
    },
    {
        "name": "500m",
        "label": "0.5B",
        "n_layer": 18,
        "n_embd": 1280,
        "n_head": 20,
        "parameters_expected": 494_516_480,
    },
    {
        "name": "1p4b",
        "label": "1.44B",
        "n_layer": 24,
        "n_embd": 2048,
        "n_head": 16,
        "parameters_expected": 1_439_270_912,
    },
)

# optimizer x precision. `bf16_state_fp32` is the baseline every ratio is taken
# against: BF16 autocast over FP32 master weights, FP32 optimizer state -- what the
# 0.5B training runs actually do.
OPTIMIZERS = ("adamw", "muon", "soap")
PRECISIONS = (
    {"suffix": "bf16_state_fp32", "gemm": "bf16", "state": "fp32"},
    {"suffix": "fp8gemm_state_fp32", "gemm": "fp8", "state": "fp32"},
    {"suffix": "bf16_state_fp8", "gemm": "bf16", "state": "fp8"},
)

VARIANTS = tuple(
    {
        "name": f"{optimizer}_{precision['suffix']}",
        "label": f"{optimizer.upper() if optimizer == 'soap' else optimizer.capitalize()}"
                 f" {'FP8 GEMM' if precision['gemm'] == 'fp8' else 'BF16'}"
                 f"{' / FP8 state' if precision['state'] == 'fp8' else ''}",
        "optimizer": optimizer,
        "gemm": precision["gemm"],
        "state": precision["state"],
    }
    for optimizer in OPTIMIZERS
    for precision in PRECISIONS
)

MICROBATCHES = (1, 2, 4, 8, 16)

DEFAULT_SEQUENCE_LENGTH = 1024
DEFAULT_TOKENS_PER_STEP = 16384
DEFAULT_VOCAB_SIZE = 50304

HARNESS_REVISION = 1

# Identical at every point of a sweep; a report refuses a mix.
COMMON_CONTROL_FIELDS = (
    "harness_revision",
    "storage_dtype",
    "autocast_dtype",
    "sequence_length",
    "tokens_per_step",
    "vocab_size",
    "warmup_steps",
    "measured_steps",
    "grad_clip",
    "fp8_group_size",
    "fp8_qgroup_size",
    "lr",
    "seed",
)

# Varied by the sweep, so they identify a point rather than the sweep.
POINT_CONTROL_FIELDS = ("microbatch", "accumulation_steps")


def model_spec(name: str) -> dict:
    for spec in MODEL_SPECS:
        if spec["name"] == name:
            return dict(spec)
    raise KeyError(f"unknown model size {name!r}")


def variant_spec(name: str) -> dict:
    for spec in VARIANTS:
        if spec["name"] == name:
            return dict(spec)
    raise KeyError(f"unknown variant {name!r}")


def mlp_hidden_dim(n_embd: int, multiple_of: int = 256) -> int:
    """src/models/llama.py::_mlp_hidden_dim, restated so the count can be checked."""
    hidden = int(2 * (n_embd * 4) / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


def expected_parameters(spec: dict, vocab_size: int = DEFAULT_VOCAB_SIZE) -> int:
    """Embeddings and lm_head are separate matrices; attention is MHA; no biases."""
    d = spec["n_embd"]
    hidden = mlp_hidden_dim(d)
    per_layer = 2 * d + 4 * d * d + 3 * d * hidden  # two RMSNorms, q/k/v/o, SwiGLU
    return 2 * vocab_size * d + spec["n_layer"] * per_layer + d


def accumulation_steps(tokens_per_step: int, microbatch: int, sequence_length: int) -> int:
    tokens_per_microstep = microbatch * sequence_length
    if tokens_per_step % tokens_per_microstep:
        raise ValueError(
            f"{tokens_per_step} tokens per step is not divisible by "
            f"microbatch {microbatch} x sequence length {sequence_length}"
        )
    return tokens_per_step // tokens_per_microstep


def requested_controls(args, microbatch: int, accumulation: int) -> dict:
    """What a stored result must reproduce to be reused instead of rerun."""
    return {
        "harness_revision": HARNESS_REVISION,
        "storage_dtype": "float32",
        "autocast_dtype": "bfloat16",
        "sequence_length": args.sequence_length,
        "microbatch": microbatch,
        "accumulation_steps": accumulation,
        "tokens_per_step": args.sequence_length * microbatch * accumulation,
        "vocab_size": args.vocab_size,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "grad_clip": args.grad_clip,
        "fp8_group_size": args.fp8_group_size,
        "fp8_qgroup_size": args.fp8_qgroup_size,
        "lr": args.lr,
        "seed": args.seed,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "std": statistics.pstdev(numeric),
        "min": min(numeric),
        "p10": percentile(numeric, 0.1),
        "p90": percentile(numeric, 0.9),
        "max": max(numeric),
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def result_is_recorded(payload: dict) -> bool:
    """A point is finished once it either measured cleanly or ran out of memory."""
    if payload.get("status") == "oom":
        return True
    if payload.get("status") != "complete":
        return False
    samples = payload.get("samples", [])
    return bool(samples) and payload.get("benchmark", {}).get("measured_steps") == len(samples)


def result_matches_request(payload: dict, *, model_name: str, variant_name: str,
                           controls: dict) -> bool:
    if not result_is_recorded(payload):
        return False
    if payload.get("status") == "oom":
        return (
            payload.get("model_size") == model_name
            and payload.get("variant") == variant_name
            and payload.get("requested_controls") == controls
        )
    if payload.get("model", {}).get("name") != model_name:
        return False
    if payload.get("variant", {}).get("name") != variant_name:
        return False
    benchmark = payload.get("benchmark", {})
    return all(benchmark.get(key) == value for key, value in controls.items())
