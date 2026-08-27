import torch

from optim.fp8_state import (
    FP8StateDictMixin,
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)

from .soap.soap_harvard import SOAP


class FP8SOAP(FP8StateDictMixin, SOAP):
    """SOAP with FP8 first- and second-order Adam moments.

    Shampoo preconditioner matrices and eigenbases remain in their native
    precision because SOAP applies eigendecomposition and QR updates to them.
    """

    def __init__(self, params, qargs, **kwargs):
        super().__init__(params, **kwargs)
        self.qargs = qargs

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]

                if "step" not in state:
                    state["step"] = 0

                if "fp8_exp_avg" not in state:
                    init_fp8_state(
                        state,
                        "fp8_exp_avg",
                        grad,
                        self.qargs,
                        order="first",
                    )
                    init_fp8_state(
                        state,
                        "fp8_exp_avg_sq",
                        grad,
                        self.qargs,
                        order="second",
                    )

                if "Q" not in state:
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=(
                            group["shampoo_beta"]
                            if group["shampoo_beta"] >= 0
                            else group["betas"][1]
                        ),
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_embed_debed=group["precondition_embed_debed"],
                    )
                    self.update_preconditioner(
                        grad,
                        state,
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"],
                        precondition_embed_debed=group["precondition_embed_debed"],
                    )
                    continue

                state["exp_avg"] = dequantize_fp8_state(
                    state,
                    "fp8_exp_avg",
                    self.qargs,
                    signed=True,
                )
                state["exp_avg_sq"] = dequantize_fp8_state(
                    state,
                    "fp8_exp_avg_sq",
                    self.qargs,
                    signed=False,
                )

                try:
                    grad_projected = self.project(
                        grad,
                        state,
                        merge_dims=group["merge_dims"],
                        max_precond_dim=group["max_precond_dim"],
                        precondition_embed_debed=group["precondition_embed_debed"],
                    )

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]
                    state["step"] += 1

                    exp_avg.mul_(beta1).add_(grad_projected, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).add_(
                        grad_projected.square(),
                        alpha=1.0 - beta2,
                    )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size *= bias_correction2**0.5 / bias_correction1

                    norm_grad = self.project_back(
                        exp_avg / denom,
                        state,
                        merge_dims=group["merge_dims"],
                        max_precond_dim=group["max_precond_dim"],
                        precondition_embed_debed=group["precondition_embed_debed"],
                    )
                    if group["normalize_grads"]:
                        norm_grad = norm_grad / (
                            1e-30 + torch.mean(norm_grad**2) ** 0.5
                        )

                    param.add_(norm_grad.to(dtype=param.dtype), alpha=-step_size)
                    if group["weight_decay"] > 0.0:
                        param.add_(
                            param,
                            alpha=-group["lr"] * group["weight_decay"],
                        )

                    self.update_preconditioner(
                        grad,
                        state,
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"],
                        precondition_embed_debed=group["precondition_embed_debed"],
                    )

                    quantize_fp8_state_(
                        state,
                        "fp8_exp_avg",
                        state["exp_avg"],
                        self.qargs,
                        signed=True,
                    )
                    quantize_fp8_state_(
                        state,
                        "fp8_exp_avg_sq",
                        state["exp_avg_sq"],
                        self.qargs,
                        signed=False,
                    )
                finally:
                    state.pop("exp_avg", None)
                    state.pop("exp_avg_sq", None)

        return loss
