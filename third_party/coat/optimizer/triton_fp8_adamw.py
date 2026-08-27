from collections import defaultdict
from copy import deepcopy
from itertools import chain
import math
from typing import Any, DefaultDict, Dict, Hashable, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing_extensions import TypeAlias

from ..utils._fp8_quantization_config import QuantizationConfig
from .triton_kernels import triton_fp8_adamw_expand_step, triton_fp8_adamw_step

StateDict: TypeAlias = Dict[str, Any]

convert_str_to_fp8 = {"E4M3": torch.float8_e4m3fn, "E5M2": torch.float8_e5m2}


def _use_expansion_mode(expansion: str) -> bool:
    return str(expansion).lower() in {"true", "expand", "expansion"}


class TritonCoatAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        qargs: QuantizationConfig = None,
        *,
        fused: Optional[bool] = None,
    ):
        if qargs is None:
            qargs = QuantizationConfig()
        self.qargs = qargs

        if self.qargs.first_order_expansion != self.qargs.second_order_expansion:
            raise ValueError(
                "Expected first/second-order momentum to use the same expansion mode."
            )
        if self.qargs.qgroup_size != 128:
            raise ValueError(f"Only qgroup_size=128 is supported, got {self.qargs.qgroup_size}.")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            fused=fused,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    p_state["step"] = torch.tensor(float(p_state["step"]), dtype=torch.float32)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        use_expansion,
        exp_avgs,
        scale_exp_avgs,
        expand_exp_avgs,
        sqrt_minmax_exp_avgs,
        exp_avg_sqs,
        scale_exp_avg_sqs,
        expand_exp_avg_sqs,
        sqrt_minmax_exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)

                first_order_dtype = convert_str_to_fp8[self.qargs.first_order_bit]
                second_order_dtype = convert_str_to_fp8[self.qargs.second_order_bit]
                scale_shape = (p.numel() + self.qargs.qgroup_size - 1) // self.qargs.qgroup_size

                state["exp_avg"] = torch.zeros_like(
                    p, dtype=first_order_dtype, memory_format=torch.preserve_format
                )
                state["scale_exp_avg"] = torch.zeros(scale_shape, device=p.device, dtype=p.dtype)
                if use_expansion:
                    state["expand_exp_avg"] = torch.ones(scale_shape, device=p.device, dtype=p.dtype)
                    state["sqrt_minmax_exp_avg"] = torch.ones(scale_shape, device=p.device, dtype=p.dtype)

                state["exp_avg_sq"] = torch.zeros_like(
                    p, dtype=second_order_dtype, memory_format=torch.preserve_format
                )
                state["scale_exp_avg_sq"] = torch.zeros(scale_shape, device=p.device, dtype=p.dtype)
                if use_expansion:
                    state["expand_exp_avg_sq"] = torch.ones(scale_shape, device=p.device, dtype=p.dtype)
                    state["sqrt_minmax_exp_avg_sq"] = torch.ones(scale_shape, device=p.device, dtype=p.dtype)

                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avgs.append(state["exp_avg"])
            scale_exp_avgs.append(state["scale_exp_avg"])
            if use_expansion:
                expand_exp_avgs.append(state["expand_exp_avg"])
                sqrt_minmax_exp_avgs.append(state["sqrt_minmax_exp_avg"])

            exp_avg_sqs.append(state["exp_avg_sq"])
            scale_exp_avg_sqs.append(state["scale_exp_avg_sq"])
            if use_expansion:
                expand_exp_avg_sqs.append(state["expand_exp_avg_sq"])
                sqrt_minmax_exp_avg_sqs.append(state["sqrt_minmax_exp_avg_sq"])

            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            state_steps.append(state["step"])

    @torch._disable_dynamo
    def load_state_dict(self, state_dict: StateDict) -> None:
        state_dict = state_dict.copy()

        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        groups = self.param_groups
        saved_groups = deepcopy(state_dict["param_groups"])

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )

        def _cast(param, value, param_id=None, param_groups=None, key=None):
            if isinstance(value, torch.Tensor):
                return TritonCoatAdamW._process_value_according_to_param_policy(
                    param, value, param_id, param_groups, key
                )
            if isinstance(value, dict):
                return {
                    k: _cast(param, v, param_id=param_id, param_groups=param_groups, key=k)
                    for k, v in value.items()
                }
            if isinstance(value, Iterable):
                return type(value)(_cast(param, v, param_id=param_id, param_groups=param_groups) for v in value)  # type: ignore[call-arg]
            return value

        state: DefaultDict[torch.Tensor, Dict[Any, Any]] = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = _cast(param, v, param_id=k, param_groups=state_dict["param_groups"])
            else:
                state[k] = v

        def update_group(group: Dict[str, Any], new_group: Dict[str, Any]) -> Dict[str, Any]:
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

    @staticmethod
    def _process_value_according_to_param_policy(
        param: torch.Tensor,
        value: torch.Tensor,
        param_id: int,
        param_groups: List[Dict[Any, Any]],
        key: Hashable = None,
    ) -> torch.Tensor:
        fused = False
        capturable = False
        assert param_groups is not None
        for pg in param_groups:
            if param_id in pg["params"]:
                fused = pg["fused"] if "fused" in pg else False
                capturable = pg["capturable"] if "capturable" in pg else False
                break

        if key == "step":
            if capturable or fused:
                return value.to(dtype=torch.float32, device=param.device)
            return value

        supported = {
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }
        if value.dtype not in supported:
            raise ValueError(f"Unsupported optimizer state dtype: {value.dtype}")
        return value.to(device=param.device)

    @torch.no_grad()
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            scale_exp_avgs = []
            expand_exp_avgs = []
            sqrt_minmax_exp_avgs = []
            exp_avg_sqs = []
            scale_exp_avg_sqs = []
            expand_exp_avg_sqs = []
            sqrt_minmax_exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            amsgrad = group["amsgrad"]
            use_expansion = _use_expansion_mode(self.qargs.first_order_expansion)
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                use_expansion,
                exp_avgs,
                scale_exp_avgs,
                expand_exp_avgs,
                sqrt_minmax_exp_avgs,
                exp_avg_sqs,
                scale_exp_avg_sqs,
                expand_exp_avg_sqs,
                sqrt_minmax_exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                step_t = state_steps[i]
                step_t += 1
                step = int(step_t.item())

                bias_correction1 = 1.0 - beta1**step
                bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
                step_size = group["lr"] / bias_correction1
                wd_lr = group["lr"] * group["weight_decay"]

                if use_expansion:
                    triton_fp8_adamw_expand_step(
                        param,
                        grad,
                        exp_avgs[i],
                        scale_exp_avgs[i],
                        expand_exp_avgs[i],
                        sqrt_minmax_exp_avgs[i],
                        exp_avg_sqs[i],
                        scale_exp_avg_sqs[i],
                        expand_exp_avg_sqs[i],
                        sqrt_minmax_exp_avg_sqs[i],
                        beta1=beta1,
                        beta2=beta2,
                        step_size=step_size,
                        bias_correction2_sqrt=bias_correction2_sqrt,
                        eps=group["eps"],
                        wd_lr=wd_lr,
                        expand_min=self.qargs.expand_min,
                        qgroup_size=self.qargs.qgroup_size,
                    )
                else:
                    triton_fp8_adamw_step(
                        param,
                        grad,
                        exp_avgs[i],
                        scale_exp_avgs[i],
                        exp_avg_sqs[i],
                        scale_exp_avg_sqs[i],
                        beta1=beta1,
                        beta2=beta2,
                        step_size=step_size,
                        bias_correction2_sqrt=bias_correction2_sqrt,
                        eps=group["eps"],
                        wd_lr=wd_lr,
                        qgroup_size=self.qargs.qgroup_size,
                    )

        return loss
