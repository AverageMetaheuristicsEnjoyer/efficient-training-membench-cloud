import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def triton_fp8_adamw_kernel(
    params_ptr,
    grads_ptr,
    exp_avg_ptr,
    scale_exp_avg_ptr,
    exp_avg_sq_ptr,
    scale_exp_avg_sq_ptr,
    beta1,
    beta2,
    step_size,
    bias_correction2_sqrt,
    eps,
    wd_lr,
    fp8_max_m,
    fp8_max_v,
    numel,
    GROUP_SIZE: tl.constexpr,
):
    quant_eps = 1e-20

    pid = tl.program_id(0)
    offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = offs < numel

    p = tl.load(params_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(grads_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    s_m = tl.load(scale_exp_avg_ptr + pid).to(tl.float32)
    raw_m = tl.load(exp_avg_ptr + offs, mask=mask, other=0.0)
    m = raw_m.to(tl.float32) * s_m

    s_v = tl.load(scale_exp_avg_sq_ptr + pid).to(tl.float32)
    raw_v = tl.load(exp_avg_sq_ptr + offs, mask=mask, other=0.0)
    v = raw_v.to(tl.float32) * s_v

    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g

    denom = tl.sqrt(tl.maximum(v, 0.0)) / bias_correction2_sqrt + eps
    normalized = m / denom
    p = p - step_size * normalized - wd_lr * p

    tl.store(params_ptr + offs, p, mask=mask)

    absmax_m = tl.max(tl.where(mask, tl.abs(m), 0.0), axis=0)
    absmax_v = tl.max(tl.where(mask, tl.abs(v), 0.0), axis=0)

    new_s_m = (absmax_m + quant_eps) / fp8_max_m
    new_s_v = (absmax_v + quant_eps) / fp8_max_v

    q_m = m / new_s_m
    q_v = v / new_s_v

    tl.store(exp_avg_ptr + offs, q_m, mask=mask)
    tl.store(scale_exp_avg_ptr + pid, new_s_m)
    tl.store(exp_avg_sq_ptr + offs, q_v, mask=mask)
    tl.store(scale_exp_avg_sq_ptr + pid, new_s_v)


@triton.jit
def triton_fp8_adamw_expand_kernel(
    params_ptr,
    grads_ptr,
    exp_avg_ptr,
    scale_exp_avg_ptr,
    expand_exp_avg_ptr,
    sqrt_minmax_exp_avg_ptr,
    exp_avg_sq_ptr,
    scale_exp_avg_sq_ptr,
    expand_exp_avg_sq_ptr,
    sqrt_minmax_exp_avg_sq_ptr,
    beta1,
    beta2,
    step_size,
    bias_correction2_sqrt,
    eps,
    wd_lr,
    fp8_max_m,
    fp8_max_v,
    expand_min,
    numel,
    GROUP_SIZE: tl.constexpr,
):
    quant_eps = 1e-30
    large_val = tl.full((GROUP_SIZE,), 3.4028235e38, tl.float32)

    pid = tl.program_id(0)
    offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = offs < numel

    p = tl.load(params_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(grads_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    s_m = tl.load(scale_exp_avg_ptr + pid).to(tl.float32)
    raw_m = tl.load(exp_avg_ptr + offs, mask=mask, other=0.0)
    expand_m = tl.maximum(tl.load(expand_exp_avg_ptr + pid).to(tl.float32), quant_eps)
    sqrtmm_m = tl.maximum(tl.load(sqrt_minmax_exp_avg_ptr + pid).to(tl.float32), quant_eps)

    m_scaled = raw_m.to(tl.float32) * s_m
    m_abs = tl.abs(m_scaled)
    m_sign = tl.where(m_scaled >= 0.0, 1.0, -1.0)
    m_pow = libdevice.pow(tl.maximum(m_abs, quant_eps), 1.0 / expand_m)
    m = tl.where(m_abs > 0.0, m_sign * m_pow * sqrtmm_m, 0.0)

    s_v = tl.load(scale_exp_avg_sq_ptr + pid).to(tl.float32)
    raw_v = tl.load(exp_avg_sq_ptr + offs, mask=mask, other=0.0)
    expand_v = tl.maximum(tl.load(expand_exp_avg_sq_ptr + pid).to(tl.float32), quant_eps)
    sqrtmm_v = tl.maximum(tl.load(sqrt_minmax_exp_avg_sq_ptr + pid).to(tl.float32), quant_eps)

    v_scaled = tl.maximum(raw_v.to(tl.float32) * s_v, 0.0)
    v_pow = libdevice.pow(tl.maximum(v_scaled, quant_eps), 1.0 / expand_v)
    v = tl.where(v_scaled > 0.0, v_pow * sqrtmm_v, 0.0)

    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g

    denom = tl.sqrt(tl.maximum(v, 0.0)) / bias_correction2_sqrt + eps
    normalized = m / denom
    p = p - step_size * normalized - wd_lr * p
    tl.store(params_ptr + offs, p, mask=mask)

    abs_m = tl.abs(m)
    valid_m = mask & (abs_m > 0.0)
    absmax_m = tl.max(tl.where(mask, abs_m, 0.0), axis=0)
    absmin_m = tl.min(tl.where(valid_m, abs_m, large_val), axis=0)
    nonzero_m = tl.sum(valid_m.to(tl.int32), axis=0)
    absmax_m = tl.maximum(absmax_m, quant_eps)
    absmin_m = tl.where(nonzero_m > 0, tl.maximum(absmin_m, quant_eps), quant_eps)

    ratio_m = absmax_m / absmin_m
    ratio_m = tl.maximum(ratio_m, 1.0 + quant_eps)
    ratio_upper_m = fp8_max_m * fp8_max_m / 2.0
    log_ratio_m = tl.log2(ratio_m)
    safe_log_ratio_m = tl.where(log_ratio_m > quant_eps, log_ratio_m, 1.0)
    raw_expand_m = tl.floor((tl.log2(ratio_upper_m) / safe_log_ratio_m) * expand_min) / expand_min
    min_expand = 1.0 / expand_min
    new_expand_m = tl.where(ratio_m <= 1.0 + quant_eps, 1.0, tl.maximum(raw_expand_m, min_expand))

    sqrt_minmax_m = tl.sqrt(absmax_m) * tl.sqrt(absmin_m)
    sqrt_minmax_m = tl.maximum(sqrt_minmax_m, quant_eps)
    norm_base_m = tl.maximum(abs_m / sqrt_minmax_m, quant_eps)
    norm_pow_m = libdevice.pow(norm_base_m, new_expand_m)
    norm_sign_m = tl.where(m >= 0.0, 1.0, -1.0)
    normalized_m = tl.where(mask & (abs_m > 0.0), norm_sign_m * norm_pow_m, 0.0)

    scale_base_m = tl.maximum(absmax_m / sqrt_minmax_m, quant_eps)
    new_scale_m = libdevice.pow(scale_base_m, new_expand_m) / fp8_max_m
    new_scale_m = tl.maximum(new_scale_m, quant_eps)
    q_m = (normalized_m / new_scale_m).to(tl.float32)

    abs_v = tl.abs(v)
    valid_v = mask & (abs_v > 0.0)
    absmax_v = tl.max(tl.where(mask, abs_v, 0.0), axis=0)
    absmin_v = tl.min(tl.where(valid_v, abs_v, large_val), axis=0)
    nonzero_v = tl.sum(valid_v.to(tl.int32), axis=0)
    absmax_v = tl.maximum(absmax_v, quant_eps)
    absmin_v = tl.where(nonzero_v > 0, tl.maximum(absmin_v, quant_eps), quant_eps)

    ratio_v = absmax_v / absmin_v
    ratio_v = tl.maximum(ratio_v, 1.0 + quant_eps)
    ratio_upper_v = fp8_max_v * fp8_max_v / 2.0
    log_ratio_v = tl.log2(ratio_v)
    safe_log_ratio_v = tl.where(log_ratio_v > quant_eps, log_ratio_v, 1.0)
    raw_expand_v = tl.floor((tl.log2(ratio_upper_v) / safe_log_ratio_v) * expand_min) / expand_min
    new_expand_v = tl.where(ratio_v <= 1.0 + quant_eps, 1.0, tl.maximum(raw_expand_v, min_expand))

    sqrt_minmax_v = tl.sqrt(absmax_v) * tl.sqrt(absmin_v)
    sqrt_minmax_v = tl.maximum(sqrt_minmax_v, quant_eps)
    norm_base_v = tl.maximum(abs_v / sqrt_minmax_v, quant_eps)
    normalized_v = tl.where(mask & (abs_v > 0.0), libdevice.pow(norm_base_v, new_expand_v), 0.0)

    scale_base_v = tl.maximum(absmax_v / sqrt_minmax_v, quant_eps)
    new_scale_v = libdevice.pow(scale_base_v, new_expand_v) / fp8_max_v
    new_scale_v = tl.maximum(new_scale_v, quant_eps)
    q_v = (normalized_v / new_scale_v).to(tl.float32)

    tl.store(exp_avg_ptr + offs, q_m, mask=mask)
    tl.store(scale_exp_avg_ptr + pid, new_scale_m.to(tl.float32))
    tl.store(expand_exp_avg_ptr + pid, new_expand_m.to(tl.float32))
    tl.store(sqrt_minmax_exp_avg_ptr + pid, sqrt_minmax_m.to(tl.float32))

    tl.store(exp_avg_sq_ptr + offs, q_v, mask=mask)
    tl.store(scale_exp_avg_sq_ptr + pid, new_scale_v.to(tl.float32))
    tl.store(expand_exp_avg_sq_ptr + pid, new_expand_v.to(tl.float32))
    tl.store(sqrt_minmax_exp_avg_sq_ptr + pid, sqrt_minmax_v.to(tl.float32))


def _check_common(param: torch.Tensor, grad: torch.Tensor, qgroup_size: int) -> None:
    if not param.is_cuda or not grad.is_cuda:
        raise ValueError("Triton FP8 AdamW kernels require CUDA tensors.")
    if param.numel() != grad.numel():
        raise ValueError("`param` and `grad` must have the same number of elements.")
    if qgroup_size != 128:
        raise ValueError(f"Only qgroup_size=128 is supported, got {qgroup_size}.")


def _launch_grid(numel: int, group_size: int):
    return (triton.cdiv(numel, group_size),)


def triton_fp8_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    scale_exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    scale_exp_avg_sq: torch.Tensor,
    *,
    beta1: float,
    beta2: float,
    step_size: float,
    bias_correction2_sqrt: float,
    eps: float,
    wd_lr: float,
    qgroup_size: int = 128,
) -> None:
    _check_common(param, grad, qgroup_size)
    numel = param.numel()
    if numel == 0:
        return

    expected_scales = triton.cdiv(numel, qgroup_size)
    if scale_exp_avg.numel() != expected_scales or scale_exp_avg_sq.numel() != expected_scales:
        raise ValueError("Scale tensors do not match parameter size and qgroup size.")

    fp8_max_m = float(torch.finfo(exp_avg.dtype).max)
    fp8_max_v = float(torch.finfo(exp_avg_sq.dtype).max)

    triton_fp8_adamw_kernel[_launch_grid(numel, qgroup_size)](
        param.view(-1),
        grad.view(-1),
        exp_avg.view(-1),
        scale_exp_avg.view(-1),
        exp_avg_sq.view(-1),
        scale_exp_avg_sq.view(-1),
        beta1,
        beta2,
        step_size,
        bias_correction2_sqrt,
        eps,
        wd_lr,
        fp8_max_m,
        fp8_max_v,
        numel,
        GROUP_SIZE=qgroup_size,
    )


def triton_fp8_adamw_expand_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    scale_exp_avg: torch.Tensor,
    expand_exp_avg: torch.Tensor,
    sqrt_minmax_exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    scale_exp_avg_sq: torch.Tensor,
    expand_exp_avg_sq: torch.Tensor,
    sqrt_minmax_exp_avg_sq: torch.Tensor,
    *,
    beta1: float,
    beta2: float,
    step_size: float,
    bias_correction2_sqrt: float,
    eps: float,
    wd_lr: float,
    expand_min: int,
    qgroup_size: int = 128,
) -> None:
    _check_common(param, grad, qgroup_size)
    numel = param.numel()
    if numel == 0:
        return
    if expand_min <= 0:
        raise ValueError(f"`expand_min` must be > 0, got {expand_min}.")

    expected_scales = triton.cdiv(numel, qgroup_size)
    tensors = (
        scale_exp_avg,
        expand_exp_avg,
        sqrt_minmax_exp_avg,
        scale_exp_avg_sq,
        expand_exp_avg_sq,
        sqrt_minmax_exp_avg_sq,
    )
    if any(t.numel() != expected_scales for t in tensors):
        raise ValueError("Expansion metadata tensors do not match parameter size and qgroup size.")

    fp8_max_m = float(torch.finfo(exp_avg.dtype).max)
    fp8_max_v = float(torch.finfo(exp_avg_sq.dtype).max)

    triton_fp8_adamw_expand_kernel[_launch_grid(numel, qgroup_size)](
        param.view(-1),
        grad.view(-1),
        exp_avg.view(-1),
        scale_exp_avg.view(-1),
        expand_exp_avg.view(-1),
        sqrt_minmax_exp_avg.view(-1),
        exp_avg_sq.view(-1),
        scale_exp_avg_sq.view(-1),
        expand_exp_avg_sq.view(-1),
        sqrt_minmax_exp_avg_sq.view(-1),
        beta1,
        beta2,
        step_size,
        bias_correction2_sqrt,
        eps,
        wd_lr,
        fp8_max_m,
        fp8_max_v,
        float(expand_min),
        numel,
        GROUP_SIZE=qgroup_size,
    )
