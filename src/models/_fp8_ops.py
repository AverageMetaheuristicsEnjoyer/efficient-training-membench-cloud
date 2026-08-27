"""
FP8 autograd Functions for the unified Llama block.

These are the numerical kernels previously embedded inside the nn.Module
wrappers in ``fp8_llama.py``. They take weight tensors directly so the
surrounding block can own the master ``nn.Linear`` modules and keep a single
parameterization across BF16 and FP8 modes.
"""

import sys
import os

import torch

_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
_SRC_PARENT   = os.path.dirname(_SRC_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_PARENT)
for _p in [_SRC_PARENT, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from third_party.coat.activation.real_quantization import (
    fp8_add_Ifp_Ifp_Ofp_Og16,
    fp8_add_Ifp_Ifp_Ofp_Opt,
    fp8_division,
    fp8_division_transpose,
    fp8_linear_backward,
    fp8_linear_forward,
    fp8_mul_backward,
    fp8_mul_forward,
    fp8_quantize_pertensor,
    fp8_quantize_pertensor_transpose,
    fp8_rmsnorm_backward,
    fp8_rmsnorm_forward,
    fp8_silu_backward,
    fp8_silu_forward,
    fp8_transpose,
)


class FP8BeforeAttentionResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        re_x, in_x, in_s,
        w1_origin, w1, w1_t, w1_s,
        w2_origin, w2, w2_t, w2_s,
        w3_origin, w3, w3_t, w3_s,
        rmsnorm_weight,
        group_size, fwobits, qargs,
        eps=1e-5,
    ):
        in_x = in_x.view(torch.float8_e4m3fn)

        ln_x, ln_s, ln_x_t, ln_utils = fp8_rmsnorm_forward(
            in_x, in_s, rmsnorm_weight, group_size, eps, transpose_output_2d=True
        )

        w1, w1_s = fp8_division(w1_origin, qargs.group_size, fwobits["fwbit"], w1_s)
        w2, w2_s = fp8_division(w2_origin, qargs.group_size, fwobits["fwbit"], w2_s)
        w3, w3_s = fp8_division(w3_origin, qargs.group_size, fwobits["fwbit"], w3_s)

        q = fp8_linear_forward(ln_x, ln_s, w1, w1_s, False, group_size)
        k = fp8_linear_forward(ln_x, ln_s, w2, w2_s, False, group_size)
        v = fp8_linear_forward(ln_x, ln_s, w3, w3_s, False, group_size)

        ctx.save_for_backward(in_x, in_s, ln_x_t, ln_s)
        ctx.weight = (w1_origin, w1_s, w2_origin, w2_s, w3_origin, w3_s)
        ctx.group_size = group_size
        ctx.ln_utils = ln_utils
        ctx.fwobits = fwobits
        ctx.qargs = qargs

        return re_x, q, k, v

    @staticmethod
    def backward(ctx, fp_grad, q_g, k_g, v_g):
        in_x, in_s, ln_x_t, ln_s = ctx.saved_tensors
        w1_t, w1_s, w2_t, w2_s, w3_t, w3_s = ctx.weight
        group_size = ctx.group_size
        rms_weight, rstd, num_warps = ctx.ln_utils
        fwobits = ctx.fwobits
        qargs = ctx.qargs

        q_g, q_gs, q_g_t = fp8_quantize_pertensor_transpose(
            q_g, group_size, fwobits["babit"], transpose_output_2d=True, stochastic=False
        )
        k_g, k_gs, k_g_t = fp8_quantize_pertensor_transpose(
            k_g, group_size, fwobits["babit"], transpose_output_2d=True, stochastic=False
        )
        v_g, v_gs, v_g_t = fp8_quantize_pertensor_transpose(
            v_g, group_size, fwobits["babit"], transpose_output_2d=True, stochastic=False
        )

        w1_t, w1_s = fp8_division_transpose(w1_t, qargs.group_size, fwobits["fwbit"], w1_s, only_transposed=True)
        w2_t, w2_s = fp8_division_transpose(w2_t, qargs.group_size, fwobits["fwbit"], w2_s, only_transposed=True)
        w3_t, w3_s = fp8_division_transpose(w3_t, qargs.group_size, fwobits["fwbit"], w3_s, only_transposed=True)

        fc_g1, wg1 = fp8_linear_backward(ln_x_t, ln_s, q_g, q_gs, q_g_t, w1_t, w1_s, group_size)
        fc_g2, wg2 = fp8_linear_backward(ln_x_t, ln_s, k_g, k_gs, k_g_t, w2_t, w2_s, group_size)
        fc_g3, wg3 = fp8_linear_backward(ln_x_t, ln_s, v_g, v_gs, v_g_t, w3_t, w3_s, group_size)
        fc_g = fc_g1 + fc_g2 + fc_g3

        in_g, rms_wg = fp8_rmsnorm_backward(in_x, in_s, fc_g, rms_weight, rstd, group_size, num_warps)

        re_g, (in_g, in_sg, in_sg_g16) = fp8_add_Ifp_Ifp_Ofp_Opt(
            fp_grad, in_g, group_size, fwobits["babit"], stochastic=False
        )
        in_g = in_g.view(torch.float8_e4m3fn)

        return (
            re_g, in_g, in_sg_g16,
            wg1, None, None, None,
            wg2, None, None, None,
            wg3, None, None, None,
            rms_wg,
            None, None, None,
        )


class FP8AfterAttentionResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, re_x, flash_x, w4_origin, w4, w4_t, w4_s, group_size, fwobits, qargs):
        flash_qx, flash_s, _ = fp8_quantize_pertensor(flash_x, group_size, fwobits["fabit"])

        w4, w4_s = fp8_division(w4_origin, qargs.group_size, fwobits["fwbit"], w4_s)
        fc4_x = fp8_linear_forward(flash_qx, flash_s, w4, w4_s, False, group_size)

        fp_x, (out_x, out_s) = fp8_add_Ifp_Ifp_Ofp_Og16(re_x, fc4_x, flash_qx.dtype, group_size)

        ctx.save_for_backward(flash_x, flash_s)
        ctx.weight = (w4_origin, w4_s)
        ctx.group_size = group_size
        ctx.fwobits = fwobits
        ctx.qargs = qargs

        out_x = out_x.view(torch.float8_e4m3fn)
        return fp_x, out_x, out_s

    @staticmethod
    def backward(ctx, fp_grad, out_g, out_gs):
        flash_x, flash_s = ctx.saved_tensors
        w4_t, w4_s = ctx.weight
        group_size = ctx.group_size
        fwobits = ctx.fwobits
        qargs = ctx.qargs

        out_g = out_g.view(torch.float8_e5m2)
        out_gs_max = out_gs.max()

        out_g_t = fp8_transpose(out_g, transpose_output_2d=True)
        flash_x_t, flash_s = fp8_division_transpose(
            flash_x, group_size, fwobits["fabit"], flash_s, stochastic=False, only_transposed=True
        )
        w4_t, w4_s = fp8_division_transpose(w4_t, qargs.group_size, fwobits["fwbit"], w4_s, only_transposed=True)
        fc4_g, w4_wg = fp8_linear_backward(flash_x_t, flash_s, out_g, out_gs_max, out_g_t, w4_t, w4_s, group_size)

        return fp_grad, fc4_g, w4_wg, None, None, None, None, None, None


class FP8MLPResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        re_x, in_x, in_s,
        w1_origin, w1, w1_t, w1_s,
        w2_origin, w2, w2_t, w2_s,
        w3_origin, w3, w3_t, w3_s,
        rmsnorm_weight,
        group_size, fwobits, qargs,
        eps=1e-5,
    ):
        in_x = in_x.view(torch.float8_e4m3fn)

        ln_x, ln_s, ln_x_t, ln_utils = fp8_rmsnorm_forward(
            in_x, in_s, rmsnorm_weight, group_size, eps, transpose_output_2d=True
        )

        w1, w1_s = fp8_division(w1_origin, qargs.group_size, fwobits["fwbit"], w1_s)
        w2, w2_s = fp8_division(w2_origin, qargs.group_size, fwobits["fwbit"], w2_s)
        w3, w3_s = fp8_division(w3_origin, qargs.group_size, fwobits["fwbit"], w3_s)

        gate_x, gate_s = fp8_linear_forward(ln_x, ln_s, w1, w1_s, True, group_size)
        up_x,   up_s   = fp8_linear_forward(ln_x, ln_s, w2, w2_s, True, group_size)

        silu_x, silu_s = fp8_silu_forward(gate_x, gate_s, group_size)

        mul_x, mul_s, mul_x_t = fp8_mul_forward(
            silu_x, silu_s, up_x, up_s, group_size, transpose_output_2d=True
        )

        fc3_x = fp8_linear_forward(mul_x, mul_s, w3, w3_s, False, group_size)

        fp_x, (out_x, out_s) = fp8_add_Ifp_Ifp_Ofp_Og16(re_x, fc3_x, mul_x.dtype, group_size)

        ctx.save_for_backward(
            in_x, in_s, ln_x_t, ln_s,
            gate_x, gate_s, up_x, up_s,
            silu_x, silu_s, mul_x_t, mul_s,
        )
        ctx.weight = (w1_origin, w1_s, w2_origin, w2_s, w3_origin, w3_s)
        ctx.group_size = group_size
        ctx.ln_utils = ln_utils
        ctx.fwobits = fwobits
        ctx.qargs = qargs

        out_x = out_x.view(torch.float8_e4m3fn)
        return fp_x, out_x, out_s

    @staticmethod
    def backward(ctx, fp_grad, out_g, out_gs):
        (in_x, in_s, ln_x_t, ln_s,
         gate_x, gate_s, up_x, up_s,
         silu_x, silu_s, mul_x_t, mul_s) = ctx.saved_tensors
        w1_t, w1_s, w2_t, w2_s, w3_t, w3_s = ctx.weight
        group_size = ctx.group_size
        rms_weight, rstd, num_warps = ctx.ln_utils
        fwobits = ctx.fwobits
        qargs = ctx.qargs

        out_g = out_g.view(torch.float8_e5m2)
        out_gs_max = out_gs.max()

        out_g_t = fp8_transpose(out_g, transpose_output_2d=True)
        w3_t, w3_s = fp8_division_transpose(w3_t, qargs.group_size, fwobits["fwbit"], w3_s, only_transposed=True)
        fc3_g, wg3 = fp8_linear_backward(mul_x_t, mul_s, out_g, out_gs_max, out_g_t, w3_t, w3_s, group_size)
        del out_g, out_g_t, w3_t

        mul_g1, (mul_g2, mul_gs2, mul_g2_t) = fp8_mul_backward(
            silu_x, silu_s, up_x, up_s, fc3_g, group_size, fwobits["babit"],
            output_quantized_transpose=True,
        )

        silu_g, silu_gs, silu_g_t = fp8_silu_backward(
            gate_x, gate_s, mul_g1, group_size, fwobits["babit"],
            output_quantized_transpose=True,
        )

        w1_t, w1_s = fp8_division_transpose(w1_t, qargs.group_size, fwobits["fwbit"], w1_s, only_transposed=True)
        w2_t, w2_s = fp8_division_transpose(w2_t, qargs.group_size, fwobits["fwbit"], w2_s, only_transposed=True)
        fc1_g, wg1 = fp8_linear_backward(ln_x_t, ln_s, silu_g, silu_gs, silu_g_t, w1_t, w1_s, group_size)
        fc2_g, wg2 = fp8_linear_backward(ln_x_t, ln_s, mul_g2, mul_gs2, mul_g2_t, w2_t, w2_s, group_size)
        fc_g = fc1_g + fc2_g

        in_g, rms_wg = fp8_rmsnorm_backward(in_x, in_s, fc_g, rms_weight, rstd, group_size, num_warps)

        re_g, (in_g, in_sg, in_sg_g16) = fp8_add_Ifp_Ifp_Ofp_Opt(
            fp_grad, in_g, group_size, fwobits["babit"], stochastic=False
        )
        in_g = in_g.view(torch.float8_e4m3fn)

        return (
            re_g, in_g, in_sg_g16,
            wg1, None, None, None,
            wg2, None, None, None,
            wg3, None, None, None,
            rms_wg,
            None, None, None,
        )
