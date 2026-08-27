#!/usr/bin/env python3
"""Run the dense memory/time sweep on one GPU, over model shapes, optimizers and batch sizes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.common import (  # noqa: E402
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TOKENS_PER_STEP,
    DEFAULT_VOCAB_SIZE,
    MICROBATCHES,
    MODEL_SPECS,
    VARIANTS,
    accumulation_steps,
    atomic_write_json,
    requested_controls,
    result_matches_request,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def result_path(output_dir: Path, model: str, variant: str, microbatch: int) -> Path:
    return output_dir / "runs" / f"{model}-{variant}-bs{microbatch}.json"


def worker_command(args, model: dict, variant: dict, microbatch: int, accumulation: int,
                   output: Path) -> list[str]:
    return [
        args.python, "-m", "bench.benchmark_train_step",
        "--variant", variant["name"],
        "--model-size", model["name"],
        "--sequence-length", str(args.sequence_length),
        "--vocab-size", str(args.vocab_size),
        "--microbatch", str(microbatch),
        "--accumulation-steps", str(accumulation),
        "--warmup-steps", str(args.warmup_steps),
        "--measured-steps", str(args.measured_steps),
        "--fp8-group-size", str(args.fp8_group_size),
        "--fp8-qgroup-size", str(args.fp8_qgroup_size),
        "--lr", str(args.lr),
        "--beta1", str(args.beta1),
        "--beta2", str(args.beta2),
        "--weight-decay", str(args.weight_decay),
        "--eps", str(args.eps),
        "--grad-clip", str(args.grad_clip),
        "--seed", str(args.seed),
        "--output", str(output),
    ]


def run_worker(args, model: dict, variant: dict, microbatch: int) -> dict:
    accumulation = accumulation_steps(args.tokens_per_step, microbatch, args.sequence_length)
    controls = requested_controls(args, microbatch, accumulation)
    output = result_path(args.output_dir, model["name"], variant["name"], microbatch)

    if output.is_file() and not args.rerun:
        payload = json.loads(output.read_text())
        if result_matches_request(payload, model_name=model["name"],
                                  variant_name=variant["name"], controls=controls):
            log(f"reuse {output.name} ({payload['status']})")
            return payload

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    log(f"run {model['label']} / {variant['label']} / microbatch {microbatch} x {accumulation}")
    completed = subprocess.run(
        worker_command(args, model, variant, microbatch, accumulation, output),
        cwd=ROOT, env=environment, check=False,
    )
    if not output.is_file():
        # The worker writes its own payload even when it fails, so an empty path means
        # it died before it could -- an import error, a missing dependency, a signal.
        raise RuntimeError(f"worker exited {completed.returncode} without producing {output}")
    payload = json.loads(output.read_text())
    if payload.get("status") == "oom":
        log(f"out of memory: {model['label']} / {variant['label']} / microbatch {microbatch}")
    elif payload.get("status") != "complete":
        # Not fatal to the sweep: one variant that cannot be built -- a missing FP8
        # optimizer, say -- must not cost every other point in the job.
        log(f"FAILED {model['label']} / {variant['label']} / microbatch {microbatch}: "
            f"{payload.get('error', payload.get('status'))}")
    return payload


def selected(specs, names: str | None, label: str) -> list[dict]:
    if not names:
        return list(specs)
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    by_name = {spec["name"]: spec for spec in specs}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}")
    return [by_name[name] for name in wanted]


COLUMNS = ("model", "variant", "microbatch", "status", "median_ms", "forward_ms",
           "backward_ms", "optimizer_ms", "tokens_per_second", "peak_gb", "reserved_gb",
           "state_gb", "params_gb")


def export_row(payload: dict) -> str:
    if payload.get("status") == "complete":
        summary, memory = payload["summary"], payload["memory"]
        fields = [
            payload["model"]["name"], payload["variant"]["name"],
            str(payload["benchmark"]["microbatch"]), "complete",
            f"{summary['host_total_ms']['median']:.3f}",
            f"{summary['forward_ms']['median']:.3f}",
            f"{summary['backward_ms']['median']:.3f}",
            f"{summary['optimizer_ms']['median']:.3f}",
            f"{summary['tokens_per_second']['median']:.1f}",
            f"{memory['peak_allocated_bytes'] / 1e9:.3f}",
            f"{memory['peak_reserved_bytes'] / 1e9:.3f}",
            f"{memory['optimizer_state_bytes'] / 1e9:.3f}",
            f"{memory['model_bytes'] / 1e9:.3f}",
        ]
    else:
        fields = [
            payload.get("model_size", "?"), payload.get("variant", "?"),
            str(payload.get("requested_controls", {}).get("microbatch", "?")),
            payload.get("status", "?"),
        ] + [""] * (len(COLUMNS) - 4)
    return "PT\t" + "\t".join(fields)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--models", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--microbatches", default=None)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--tokens-per-step", type=int, default=DEFAULT_TOKENS_PER_STEP)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=12)
    parser.add_argument("--fp8-group-size", type=int, default=16)
    parser.add_argument("--fp8-qgroup-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.export_only:
        print("PT\t" + "\t".join(COLUMNS))
        for path in sorted((args.output_dir / "runs").glob("*.json")):
            print(export_row(json.loads(path.read_text())))
        return 0

    models = selected(MODEL_SPECS, args.models, "model size")
    variants = selected(VARIANTS, args.variants, "variant")
    microbatches = (
        sorted(int(value) for value in args.microbatches.split(","))
        if args.microbatches else list(MICROBATCHES)
    )
    for microbatch in microbatches:
        accumulation_steps(args.tokens_per_step, microbatch, args.sequence_length)

    # Once a point runs out of memory, every larger micro-batch of it will too.
    exhausted: set[tuple[str, str]] = set()
    # A variant that fails to build fails identically at every micro-batch, and each
    # attempt costs a process start; a resubmission retries, since a failed point is
    # never reused.
    broken: set[tuple[str, str]] = set()
    results = []
    for model in models:
        for microbatch in microbatches:
            for variant in variants:
                key = (model["name"], variant["name"])
                if key in exhausted or key in broken:
                    reason = "ran out of memory" if key in exhausted else "failed"
                    log(f"skip {model['label']} / {variant['label']} / microbatch "
                        f"{microbatch}: a smaller micro-batch {reason}")
                    continue
                payload = run_worker(args, model, variant, microbatch)
                if payload.get("status") == "oom":
                    exhausted.add(key)
                elif payload.get("status") != "complete":
                    broken.add(key)
                results.append(payload)
                atomic_write_json(args.output_dir / "results.json", {
                    "status": "complete",
                    "sweep": {
                        "models": [item["name"] for item in models],
                        "variants": [item["name"] for item in variants],
                        "microbatches": microbatches,
                        "sequence_length": args.sequence_length,
                        "tokens_per_step": args.tokens_per_step,
                        "warmup_steps": args.warmup_steps,
                        "measured_steps": args.measured_steps,
                    },
                    "results": results,
                })
    log(f"sweep complete: {args.output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
