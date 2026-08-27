# Dense memory and time benchmark

Peak memory and step time for a dense Llama across **model size**, **optimizer** and
**batch size**, on one H100. It is the dense half of a two-part benchmark; the MoE
half lives in
[`H-MoE-Part-cloud`](https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud)
on the `bench/fp8-membench` branch and uses the same protocol and the same variant
names, so the two tables join on the variant column.

## Why this repository exists

Cloud.ru's `mlsub` clones over public https from github.com or gitlab.com only, and
`modernTalker/Efficient-Training` -- where this code is developed -- is private. So
the code a GPU job runs has to be reachable publicly. This mirror carries the model,
the three optimizers and the FP8 machinery the benchmark measures, and nothing else:
no training entry point, no datasets, no evaluation stack, no run scripts, no
history. `./sync_from_upstream.sh` re-copies that subset and rewrites `UPSTREAM.txt`
with the commit it came from.

The upstream branch is `codex/efficient-image-muon`, which is where FP8 SOAP
(`src/optim/sota_opt/fp8_soap.py`), FP8 Muon (`MuonLite` with `qargs`) and the FP8
AdamW all exist together. `debug-fp8` has none of the first two, and
`codex/cloud-fp8-optimizer-states` has no FP8 SOAP.

The commit `sync_from_upstream.sh` recorded in `UPSTREAM.txt` is part of every
point's controls, so re-syncing invalidates measurements taken against the old
revision instead of silently reusing them.

One deliberate omission is worth knowing about: `src/optim/sota_opt/__init__.py` is
not copied. Importing it pulls in MARS, Adan, SWAN and Shampoo, none of which this
benchmark builds. Without it `optim.sota_opt` resolves as a namespace package and the
SOAP modules import exactly as they do upstream.

## Axes

| | |
|---|---|
| Model | `257m` (12L/1024), `500m` (18L/1280), `1p4b` (24L/2048), `3p5b` (28L/3072), `6p9b` (32L/4096) -- 257,188,864 / 494,516,480 / 1,439,270,912 / 3,480,136,704 / 6,888,361,984 parameters |
| Optimizer | AdamW, Muon (`MuonLite` with LITE disabled), SOAP |
| Precision | `bf16_state_fp32` baseline, `fp8gemm_state_fp32` (COAT FP8 activations), `bf16_state_fp8` (FP8 optimizer state) |
| Micro-batch | 1, 2, 4, 8, 16 |
| Tokens per optimizer step | fixed at 16,384 (16 sequences x 1,024), accumulation traded against the micro-batch |
| Window | 3 warmup steps, then 12 measured steps |

Holding tokens per optimizer step fixed is what makes the time columns comparable:
every cell does the same work, so a step time is a step time and not a batch size in
disguise. Peak memory still moves with the micro-batch, which is that axis's point.

The measured step is the one `src/optim/base.py::train` runs -- BF16 autocast over
FP32 weights, gradient accumulation, grad-norm clipping at 1.0, optimizer step --
with data loading removed. Inputs are random tokens, preallocated once, because a
step time that includes the dataloader is not a step time. That is sound for a dense
model, whose per-step work does not depend on the content of the batch. (It would not
be sound for the MoE half, where routing does; that half reads the real corpus.)

Running out of memory is recorded as a result, not an error, and every larger
micro-batch of that model and variant is then skipped. It is the expected result at
the top of the size axis: the measured recipe keeps FP32 weights, FP32 gradients and
FP32 optimizer moments, which is 110.2 GB at 6.89B before a single activation. More
cards do not move that number -- this harness replicates the way the training run
does, it does not shard the optimizer -- so 6.89B is swept to record where the wall
is and which variants, if any, sit under it.

## Running it

```bash
ssh brain_lab mlsub run \
  --repo https://github.com/AverageMetaheuristicsEnjoyer/efficient-training-membench-cloud \
  --branch main --entry scripts/cloud_run.sh --image torch28 --no-pip --gpus 1 \
  --note membench-dense \
  --args "--models 500m --variants adamw_bf16_state_fp32 --microbatches 1,2"
```

`--image torch28` is required: the default image ships torch 2.1, which has no FP8
dtypes. `--args probe` reports what the image provides and installs what it does not;
`--args export` prints one line per recorded point, which is the only way to read the
table back out of a finished job; `--args peek` shows the newest log; `--args disk`
reports free space.

The sweep is resumable -- a point is reused when the controls it was recorded at match
the ones requested -- so a job that runs out of time is continued by resubmitting it.

### Where it writes

`/workspace-SR006.nfs3/dense-membench`, including its pip prefix. **Not
`/home/jovyan`:** that volume reached 0 bytes free on 2026-08-27, and `mlsub` points
`PYTHONUSERBASE` there by default, so an install would fail on ENOSPC.

## Local checks

```bash
python -m pytest tests -q      # no GPU, no torch
```
