#!/usr/bin/env python3
"""Benchmark one dense Llama training-step configuration on one GPU.

One point of the sweep: one model shape, one optimizer/precision variant, one
micro-batch. The step is the one `src/optim/base.py::train` runs -- BF16 autocast
over FP32 weights, gradient accumulation, grad-norm clipping, optimizer step -- with
data loading removed, because preallocated inputs are the only way a step time means
the step and not the dataloader.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from bench.common import (  # noqa: E402
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_VOCAB_SIZE,
    MODEL_SPECS,
    VARIANTS,
    atomic_write_json,
    expected_parameters,
    model_spec,
    requested_controls,
    summarize,
    variant_spec,
)


def build_qargs(args, variant: dict):
    """The QuantizationConfig src/main.py builds from the --fp8* flags.

    One object carries both axes: `quantize_model` switches the COAT activation path
    on, and the qgroup/order fields are read by the FP8 optimizer states. A variant
    that uses neither still needs no config at all.
    """
    if variant["gemm"] == "bf16" and variant["state"] == "fp32":
        return None
    from third_party.coat.utils._fp8_quantization_config import QuantizationConfig

    return QuantizationConfig(
        quantize_model="coat_real" if variant["gemm"] == "fp8" else "none",
        fabit="E4M3",
        fwbit="E4M3",
        fobit="E4M3",
        babit="E5M2",
        bwbit="E5M2",
        bobit="E5M2",
        group_size=args.fp8_group_size,
        weight_memory_efficient=True,
        first_order_expansion="expand",
        second_order_expansion="expand",
        first_order_bit="E4M3",
        second_order_bit="E4M3",
        qgroup_size=args.fp8_qgroup_size,
    )


def make_config(spec: dict, args, variant: dict, qargs) -> SimpleNamespace:
    return SimpleNamespace(
        model="llama",
        vocab_size=args.vocab_size,
        sequence_length=args.sequence_length,
        dropout=0.0,
        n_layer=spec["n_layer"],
        n_embd=spec["n_embd"],
        n_head=spec["n_head"],
        multiple_of=256,
        rmsnorm_eps=1e-5,
        init_std=0.02,
        bias=False,
        fp8=variant["gemm"] == "fp8",
        qargs=qargs,
    )


def parameter_group_specs(model) -> list[dict]:
    """model.get_parameter_group_specs() with names resolved to parameters."""
    by_name = dict(model.named_parameters())
    groups = []
    for specification in model.get_parameter_group_specs():
        group = {key: value for key, value in specification.items() if key != "params"}
        group["params"] = [by_name[name] for name in specification["params"]]
        groups.append(group)
    return groups


def muon_parameter_split(model):
    """src/main.py's split: 2-D weights that are not embeddings go to Muon."""
    muon, adamw = [], []
    for name, parameter in model.named_parameters():
        target = muon if (
            parameter.ndim == 2
            and not any(key in name for key in ("wte", "lm_head", "embed"))
        ) else adamw
        target.append((name, parameter))
    return muon, adamw


def build_optimizer(args, variant: dict, model, qargs):
    groups = parameter_group_specs(model)
    fp8_state = variant["state"] == "fp8"

    if variant["optimizer"] == "adamw":
        if fp8_state:
            from third_party.coat.optimizer.triton_fp8_adamw import TritonCoatAdamW

            return TritonCoatAdamW(
                groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
                weight_decay=args.weight_decay, qargs=qargs,
            ), "third_party.coat.optimizer.triton_fp8_adamw.TritonCoatAdamW"
        return torch.optim.AdamW(
            groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
            weight_decay=args.weight_decay,
        ), "torch.optim.AdamW"

    if variant["optimizer"] == "muon":
        from third_party.lite.muonlite import MuonLite

        muon_params, adamw_params = muon_parameter_split(model)
        # `--opt muon` is MuonLite with LITE disabled: no subspace, no amplification.
        return MuonLite(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            ns_steps=args.muon_ns_steps,
            muon_theta=args.muon_theta,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            total_steps=args.warmup_steps + args.measured_steps,
            warmup_steps=1,
            beta1=0.0, beta2=0.0, chi=1.0, chi_adamw=1.0, subspace_ratio=0.0,
            qargs=qargs if fp8_state else None,
        ), "third_party.lite.muonlite.MuonLite"

    if variant["optimizer"] == "soap":
        # Imported by module path, not through optim.sota_opt, whose __init__ drags in
        # the whole SOTA optimizer zoo (MARS, Adan, Shampoo) for no reason here.
        soap_kwargs = dict(
            lr=args.lr, betas=(args.beta1, args.beta2), shampoo_beta=args.shampoo_beta,
            eps=args.eps, weight_decay=args.weight_decay,
            precondition_frequency=args.precondition_frequency,
            precondition_embed_debed=True,
        )
        if fp8_state:
            from optim.sota_opt.fp8_soap import FP8SOAP

            return FP8SOAP(groups, qargs=qargs, **soap_kwargs), "optim.sota_opt.fp8_soap.FP8SOAP"
        from optim.sota_opt.soap.soap_harvard import SOAP

        return SOAP(groups, **soap_kwargs), "optim.sota_opt.soap.soap_harvard.SOAP"

    raise ValueError(variant["optimizer"])


def tensor_bytes(value, seen: set[int] | None = None) -> int:
    """Optimizer state is not always tensors in a dict: FP8 states carry scale and
    expansion metadata, and a projector-style state keeps matrices in attributes."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        return sum(tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item, seen) for item in value)
    attributes = getattr(value, "__dict__", None)
    if attributes is None or id(value) in seen:
        return 0
    seen.add(id(value))
    return sum(tensor_bytes(item, seen) for item in attributes.values())


def tensor_dtypes(value, *, seen: set[int] | None = None) -> set[str]:
    if isinstance(value, torch.Tensor):
        return {str(value.dtype).removeprefix("torch.")}
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        items = value.values()
    elif isinstance(value, (tuple, list)):
        items = value
    else:
        attributes = getattr(value, "__dict__", None)
        if attributes is None or id(value) in seen:
            return set()
        seen.add(id(value))
        items = attributes.values()
    collected = [tensor_dtypes(item, seen=seen) for item in items]
    return set().union(*collected) if collected else set()


def timed_step(model, optimizer, batches, args, variant) -> dict:
    if variant["gemm"] == "fp8" or variant["state"] == "fp8":
        # The FP8 weight cache recomputes scales on the first microbatch of a step and
        # reuses them for the rest; src/optim/base.py sets this at the same point.
        from third_party.coat.utils._fp8manager import FP8Manager

        FP8Manager.is_first_microbatch = True

    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    forward_ns = backward_ns = 0
    losses = []
    for inputs, targets in batches:
        microstep = time.perf_counter_ns()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(inputs, targets=targets)["loss"] / len(batches)
        torch.cuda.synchronize()
        middle = time.perf_counter_ns()
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        forward_ns += middle - microstep
        backward_ns += end - middle
        losses.append(loss.detach())

    clip_started = time.perf_counter_ns()
    if args.grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    torch.cuda.synchronize()
    optimizer_started = time.perf_counter_ns()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    finished = time.perf_counter_ns()

    return {
        "host_total_ms": (finished - started) / 1e6,
        "forward_ms": forward_ns / 1e6,
        "backward_ms": backward_ns / 1e6,
        "clip_ms": (optimizer_started - clip_started) / 1e6,
        "optimizer_ms": (finished - optimizer_started) / 1e6,
        "loss": sum(float(value.float().item()) for value in losses),
    }


METRICS = ("host_total_ms", "forward_ms", "backward_ms", "clip_ms", "optimizer_ms",
           "tokens_per_second")


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    spec = model_spec(args.model_size)
    variant = variant_spec(args.variant)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    predicted = expected_parameters(spec, args.vocab_size)
    if predicted != spec["parameters_expected"]:
        raise RuntimeError(
            f"the recorded parameter count for {spec['name']} does not follow from its "
            f"geometry: {spec['parameters_expected']} vs {predicted}"
        )

    qargs = build_qargs(args, variant)
    from models.utils import get_model

    config = make_config(spec, args, variant, qargs)
    model = get_model(config).to(device)
    model.train()
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != spec["parameters_expected"]:
        raise RuntimeError(
            f"parameter-count mismatch: actual={actual_parameters}, "
            f"expected={spec['parameters_expected']}"
        )

    optimizer, backend = build_optimizer(args, variant, model, qargs)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 17)
    batches = [
        (
            torch.randint(0, args.vocab_size, (args.microbatch, args.sequence_length),
                          generator=generator, device=device),
            torch.randint(0, args.vocab_size, (args.microbatch, args.sequence_length),
                          generator=generator, device=device),
        )
        for _ in range(args.accumulation_steps)
    ]
    tokens_per_step = args.microbatch * args.sequence_length * args.accumulation_steps

    for _ in range(args.warmup_steps):
        timed_step(model, optimizer, batches, args, variant)

    # The optimizer state exists only after a step, so it is sized here and not
    # before; the peak is reset now so warmup allocation does not leak into it.
    model_bytes = sum(tensor_bytes(parameter) for parameter in model.parameters())
    state_bytes = tensor_bytes(optimizer.state)
    state_dtypes = sorted(tensor_dtypes(optimizer.state))

    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for iteration in range(args.measured_steps):
        sample = timed_step(model, optimizer, batches, args, variant)
        sample["iteration"] = iteration
        sample["tokens_per_second"] = tokens_per_step / (sample["host_total_ms"] / 1000.0)
        samples.append(sample)

    memory = {
        "model_bytes": model_bytes,
        "optimizer_state_bytes": state_bytes,
        "optimizer_state_dtypes": state_dtypes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    return {
        "status": "complete",
        "model": {**spec, "actual_parameters": actual_parameters},
        "variant": variant,
        "benchmark": {
            **requested_controls(args, args.microbatch, args.accumulation_steps),
            "optimizer_backend": backend,
            "timing": "host clock with a device sync at every phase boundary",
        },
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "uuid": f"GPU-{torch.cuda.get_device_properties(device).uuid}",
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "memory": memory,
        "summary": {metric: summarize([sample[metric] for sample in samples])
                    for metric in METRICS},
        "samples": samples,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pid": os.getpid(),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True,
                        choices=[variant["name"] for variant in VARIANTS])
    parser.add_argument("--model-size", required=True,
                        choices=[spec["name"] for spec in MODEL_SPECS])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=12)
    parser.add_argument("--fp8-group-size", type=int, default=16)
    parser.add_argument("--fp8-qgroup-size", type=int, default=128)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-theta", type=float, default=0.95)
    parser.add_argument("--shampoo-beta", type=float, default=-1.0)
    parser.add_argument("--precondition-frequency", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.microbatch, args.sequence_length, args.accumulation_steps) <= 0:
        raise ValueError("microbatch, sequence length and accumulation steps must be positive")
    if args.warmup_steps < 1 or args.measured_steps < 1:
        raise ValueError("warmup and measured steps must both be positive")
    started = time.time()
    try:
        payload = run(args)
        payload["wall_started_unix"] = started
        payload["wall_finished_unix"] = time.time()
        atomic_write_json(args.output, payload)
        print(json.dumps({
            "status": payload["status"],
            "model": payload["model"]["name"],
            "variant": payload["variant"]["name"],
            "median_ms": payload["summary"]["host_total_ms"]["median"],
            "peak_gb": payload["memory"]["peak_allocated_bytes"] / 1e9,
            "state_gb": payload["memory"]["optimizer_state_bytes"] / 1e9,
        }, sort_keys=True))
    except BaseException as error:
        status = "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed"
        atomic_write_json(args.output, {
            "status": status,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "model_size": args.model_size,
            "variant": args.variant,
            "requested_controls": requested_controls(
                args, args.microbatch, args.accumulation_steps
            ),
            "wall_started_unix": started,
            "wall_finished_unix": time.time(),
        })
        print(f"{status}: {type(error).__name__}: {error}", file=sys.stderr)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
