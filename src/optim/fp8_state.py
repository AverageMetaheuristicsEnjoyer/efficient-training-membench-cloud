from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Any, DefaultDict, Dict, Hashable, Iterable, Optional

import torch


_FP8_TYPES = {
    "E4M3": torch.float8_e4m3fn,
    "E5M2": torch.float8_e5m2,
}
_EXPANSION_MODES = {"true", "expand", "expansion"}
_QUANT_EPS = 1e-30


def get_fp8_dtype(bit: str) -> torch.dtype:
    try:
        return _FP8_TYPES[bit]
    except KeyError as exc:
        raise ValueError(f"Unsupported FP8 type: {bit}") from exc


def use_expansion_mode(expansion: str) -> bool:
    return str(expansion).lower() in _EXPANSION_MODES


def _meta_dtype(reference: torch.Tensor) -> torch.dtype:
    if reference.dtype in {torch.float16, torch.bfloat16, torch.float32}:
        return reference.dtype
    return torch.float32


def _num_groups(numel: int, qgroup_size: int) -> int:
    if qgroup_size <= 0:
        raise ValueError(f"qgroup_size must be > 0, got {qgroup_size}")
    return (numel + qgroup_size - 1) // qgroup_size


def _reshape_groups(tensor: torch.Tensor, qgroup_size: int) -> tuple[torch.Tensor, int]:
    flat = tensor.reshape(-1).to(torch.float32)
    num_groups = _num_groups(flat.numel(), qgroup_size)
    padded = num_groups * qgroup_size - flat.numel()
    if padded:
        flat = torch.cat([flat, flat.new_zeros(padded)], dim=0)
    return flat.view(num_groups, qgroup_size), tensor.numel()


def init_fp8_state(
    state: Dict[str, Any],
    prefix: str,
    reference: torch.Tensor,
    qargs,
    *,
    order: str,
) -> None:
    if order == "first":
        fp8_dtype = get_fp8_dtype(qargs.first_order_bit)
        use_expansion = use_expansion_mode(qargs.first_order_expansion)
    elif order == "second":
        fp8_dtype = get_fp8_dtype(qargs.second_order_bit)
        use_expansion = use_expansion_mode(qargs.second_order_expansion)
    else:
        raise ValueError(f"Unknown order: {order}")

    scale_shape = _num_groups(reference.numel(), qargs.qgroup_size)
    meta_dtype = _meta_dtype(reference)
    state[prefix] = torch.zeros_like(
        reference,
        dtype=fp8_dtype,
        memory_format=torch.preserve_format,
    )
    state[f"scale_{prefix}"] = torch.zeros(
        scale_shape,
        device=reference.device,
        dtype=meta_dtype,
    )
    if use_expansion:
        state[f"expand_{prefix}"] = torch.ones(
            scale_shape,
            device=reference.device,
            dtype=meta_dtype,
        )
        state[f"sqrt_minmax_{prefix}"] = torch.ones(
            scale_shape,
            device=reference.device,
            dtype=meta_dtype,
        )


def dequantize_fp8_state(
    state: Dict[str, Any],
    prefix: str,
    qargs,
    *,
    signed: bool,
) -> torch.Tensor:
    data = state[prefix]
    groups, orig_numel = _reshape_groups(data, qargs.qgroup_size)
    scale = state[f"scale_{prefix}"].to(torch.float32).view(-1, 1)
    raw = groups * scale

    if f"expand_{prefix}" in state:
        expand = state[f"expand_{prefix}"].to(torch.float32).view(-1, 1).clamp_min(_QUANT_EPS)
        sqrt_minmax = (
            state[f"sqrt_minmax_{prefix}"]
            .to(torch.float32)
            .view(-1, 1)
            .clamp_min(_QUANT_EPS)
        )
        abs_raw = raw.abs() if signed else raw.clamp_min(0)
        restored = torch.pow(abs_raw.clamp_min(_QUANT_EPS), 1.0 / expand) * sqrt_minmax
        restored = torch.where(abs_raw > 0, restored, torch.zeros_like(restored))
        if signed:
            restored = torch.sign(raw) * restored
    else:
        restored = raw if signed else raw.clamp_min(0)

    return restored.reshape(-1)[:orig_numel].reshape_as(data)


def quantize_fp8_state_(
    state: Dict[str, Any],
    prefix: str,
    value: torch.Tensor,
    qargs,
    *,
    signed: bool,
) -> None:
    groups, orig_numel = _reshape_groups(value, qargs.qgroup_size)
    stored = state[prefix]
    fp8_max = float(torch.finfo(stored.dtype).max)
    source = groups if signed else groups.clamp_min(0)
    abs_source = source.abs() if signed else source

    if f"expand_{prefix}" in state:
        nonzero = abs_source > 0
        absmax = abs_source.max(dim=1).values.clamp_min(_QUANT_EPS)
        inf = torch.full_like(abs_source, torch.finfo(torch.float32).max)
        absmin = torch.where(nonzero, abs_source, inf).min(dim=1).values
        has_nonzero = nonzero.any(dim=1)
        absmin = torch.where(
            has_nonzero,
            absmin.clamp_min(_QUANT_EPS),
            torch.full_like(absmin, _QUANT_EPS),
        )

        ratio = (absmax / absmin).clamp_min(1.0 + _QUANT_EPS)
        ratio_upper = torch.full_like(ratio, fp8_max * fp8_max / 2.0)
        log_ratio = torch.log2(ratio)
        raw_expand = torch.floor(
            (torch.log2(ratio_upper) / log_ratio.clamp_min(_QUANT_EPS)) * qargs.expand_min
        )
        raw_expand = raw_expand / qargs.expand_min
        min_expand = torch.full_like(raw_expand, 1.0 / qargs.expand_min)
        expand = torch.where(
            ratio <= 1.0 + _QUANT_EPS,
            torch.ones_like(raw_expand),
            torch.maximum(raw_expand, min_expand),
        )

        sqrt_minmax = (absmax.sqrt() * absmin.sqrt()).clamp_min(_QUANT_EPS)
        base = (abs_source / sqrt_minmax.view(-1, 1)).clamp_min(_QUANT_EPS)
        normalized = torch.pow(base, expand.view(-1, 1))
        normalized = torch.where(nonzero, normalized, torch.zeros_like(normalized))
        if signed:
            normalized = torch.sign(source) * normalized
        scale_base = (absmax / sqrt_minmax).clamp_min(_QUANT_EPS)
        scale = torch.pow(scale_base, expand) / fp8_max
    else:
        absmax = abs_source.max(dim=1).values
        scale = (absmax + _QUANT_EPS) / fp8_max
        normalized = source

    scale = scale.clamp_min(_QUANT_EPS)
    quantized = (normalized / scale.view(-1, 1)).reshape(-1)[:orig_numel].reshape_as(stored)
    state[prefix] = quantized.to(dtype=stored.dtype)
    state[f"scale_{prefix}"] = scale.to(
        dtype=state[f"scale_{prefix}"].dtype,
        device=stored.device,
    )
    if f"expand_{prefix}" in state:
        state[f"expand_{prefix}"] = expand.to(
            dtype=state[f"expand_{prefix}"].dtype,
            device=stored.device,
        )
        state[f"sqrt_minmax_{prefix}"] = sqrt_minmax.to(
            dtype=state[f"sqrt_minmax_{prefix}"].dtype,
            device=stored.device,
        )


class FP8StateDictMixin:
    """Load optimizer states without casting FP8 tensors to parameter dtype."""

    @torch._disable_dynamo
    def load_state_dict(self, state_dict):  # type: ignore[override]
        state_dict = state_dict.copy()
        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        groups = self.param_groups
        saved_groups = deepcopy(state_dict["param_groups"])
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        if any(
            len(group["params"]) != len(saved["params"])
            for group, saved in zip(groups, saved_groups)
        ):
            raise ValueError("loaded state dict has incompatible parameter groups")

        id_map = dict(
            zip(
                chain.from_iterable(group["params"] for group in saved_groups),
                chain.from_iterable(group["params"] for group in groups),
            )
        )

        def _cast(param, value, key: Optional[Hashable] = None):
            del key
            if isinstance(value, torch.Tensor):
                return value.to(device=param.device)
            if isinstance(value, dict):
                return {k: _cast(param, v, key=k) for k, v in value.items()}
            if isinstance(value, (str, bytes)):
                return value
            if isinstance(value, Iterable):
                return type(value)(_cast(param, item) for item in value)
            return value

        state: DefaultDict[torch.Tensor, Dict[Any, Any]] = defaultdict(dict)
        for key, value in state_dict["state"].items():
            if key in id_map:
                state[id_map[key]] = _cast(id_map[key], value)
            else:
                state[key] = value

        for group, saved in zip(groups, saved_groups):
            saved["params"] = group["params"]
        self.__setstate__({"state": state, "param_groups": saved_groups})

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)
