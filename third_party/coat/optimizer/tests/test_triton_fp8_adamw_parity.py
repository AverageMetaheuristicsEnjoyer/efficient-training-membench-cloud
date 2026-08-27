import unittest

import torch

from third_party.coat.utils._fp8_quantization_config import QuantizationConfig

CoatAdamW = None
TritonCoatAdamW = None


def _require_cuda_stack():
    global CoatAdamW, TritonCoatAdamW

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for FP8 optimizer parity tests.")

    try:
        import triton  # noqa: F401
    except Exception as exc:
        raise unittest.SkipTest(f"Triton is unavailable: {exc}") from exc

    try:
        import qoptim_cuda  # noqa: F401
    except Exception as exc:
        raise unittest.SkipTest(f"qoptim_cuda is unavailable: {exc}") from exc

    if CoatAdamW is None or TritonCoatAdamW is None:
        try:
            from third_party.coat.optimizer.fp8_adamw import CoatAdamW as _CoatAdamW
            from third_party.coat.optimizer.triton_fp8_adamw import TritonCoatAdamW as _TritonCoatAdamW
        except Exception as exc:
            raise unittest.SkipTest(f"Optimizer imports failed: {exc}") from exc
        CoatAdamW = _CoatAdamW
        TritonCoatAdamW = _TritonCoatAdamW


def _build_qargs(expansion: str, first_order_bit: str = "E4M3", second_order_bit: str = "E4M3") -> QuantizationConfig:
    return QuantizationConfig(
        quantize_model="none",
        first_order_expansion=expansion,
        second_order_expansion=expansion,
        first_order_bit=first_order_bit,
        second_order_bit=second_order_bit,
        qgroup_size=128,
        expand_min=16,
    )


def _dequant_basic(fp8_tensor: torch.Tensor, scale: torch.Tensor, qgroup_size: int) -> torch.Tensor:
    flat = fp8_tensor.view(-1).to(torch.float32)
    idx = torch.arange(flat.numel(), device=flat.device) // qgroup_size
    s = scale.to(torch.float32)[idx]
    return flat * s


def _dequant_expand(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
    expand: torch.Tensor,
    sqrt_minmax: torch.Tensor,
    qgroup_size: int,
    *,
    signed: bool,
) -> torch.Tensor:
    flat = fp8_tensor.view(-1).to(torch.float32)
    idx = torch.arange(flat.numel(), device=flat.device) // qgroup_size
    s = scale.to(torch.float32)[idx]
    e = expand.to(torch.float32)[idx].clamp_min(1e-20)
    z = sqrt_minmax.to(torch.float32)[idx].clamp_min(1e-20)

    raw = flat * s
    if signed:
        abs_raw = raw.abs()
        out = torch.sign(raw) * torch.pow(abs_raw.clamp_min(1e-20), 1.0 / e) * z
        out = torch.where(abs_raw > 0, out, torch.zeros_like(out))
    else:
        raw_pos = raw.clamp_min(0)
        out = torch.pow(raw_pos.clamp_min(1e-20), 1.0 / e) * z
        out = torch.where(raw_pos > 0, out, torch.zeros_like(out))
    return out


class TestTritonCoatAdamWParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _require_cuda_stack()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def _run_parity_case(self, numel: int, expansion_mode: str):
        qargs = _build_qargs(expansion=expansion_mode)
        lr = 3e-4
        betas = (0.9, 0.95)
        eps = 1e-8
        wd = 0.1
        qgroup_size = qargs.qgroup_size

        p_ref = torch.nn.Parameter(torch.randn(numel, device="cuda", dtype=torch.float32))
        p_tri = torch.nn.Parameter(p_ref.detach().clone())
        grad = torch.randn_like(p_ref)

        opt_ref = CoatAdamW([p_ref], lr=lr, betas=betas, eps=eps, weight_decay=wd, qargs=qargs)
        opt_tri = TritonCoatAdamW([p_tri], lr=lr, betas=betas, eps=eps, weight_decay=wd, qargs=qargs)

        p_ref.grad = grad.clone()
        p_tri.grad = grad.clone()
        opt_ref.step()
        opt_tri.step()

        torch.testing.assert_close(p_tri, p_ref, atol=2e-3, rtol=2e-3)

        s_ref = opt_ref.state[p_ref]
        s_tri = opt_tri.state[p_tri]

        self.assertEqual(set(s_ref.keys()), set(s_tri.keys()))
        torch.testing.assert_close(s_tri["scale_exp_avg"], s_ref["scale_exp_avg"], atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(s_tri["scale_exp_avg_sq"], s_ref["scale_exp_avg_sq"], atol=2e-3, rtol=2e-3)

        if expansion_mode.lower() in {"true", "expand", "expansion"}:
            torch.testing.assert_close(s_tri["expand_exp_avg"], s_ref["expand_exp_avg"], atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(
                s_tri["sqrt_minmax_exp_avg"], s_ref["sqrt_minmax_exp_avg"], atol=2e-3, rtol=2e-3
            )
            torch.testing.assert_close(
                s_tri["expand_exp_avg_sq"], s_ref["expand_exp_avg_sq"], atol=2e-3, rtol=2e-3
            )
            torch.testing.assert_close(
                s_tri["sqrt_minmax_exp_avg_sq"], s_ref["sqrt_minmax_exp_avg_sq"], atol=2e-3, rtol=2e-3
            )

            m_ref = _dequant_expand(
                s_ref["exp_avg"],
                s_ref["scale_exp_avg"],
                s_ref["expand_exp_avg"],
                s_ref["sqrt_minmax_exp_avg"],
                qgroup_size,
                signed=True,
            )
            m_tri = _dequant_expand(
                s_tri["exp_avg"],
                s_tri["scale_exp_avg"],
                s_tri["expand_exp_avg"],
                s_tri["sqrt_minmax_exp_avg"],
                qgroup_size,
                signed=True,
            )
            v_ref = _dequant_expand(
                s_ref["exp_avg_sq"],
                s_ref["scale_exp_avg_sq"],
                s_ref["expand_exp_avg_sq"],
                s_ref["sqrt_minmax_exp_avg_sq"],
                qgroup_size,
                signed=False,
            )
            v_tri = _dequant_expand(
                s_tri["exp_avg_sq"],
                s_tri["scale_exp_avg_sq"],
                s_tri["expand_exp_avg_sq"],
                s_tri["sqrt_minmax_exp_avg_sq"],
                qgroup_size,
                signed=False,
            )
        else:
            m_ref = _dequant_basic(s_ref["exp_avg"], s_ref["scale_exp_avg"], qgroup_size)
            m_tri = _dequant_basic(s_tri["exp_avg"], s_tri["scale_exp_avg"], qgroup_size)
            v_ref = _dequant_basic(s_ref["exp_avg_sq"], s_ref["scale_exp_avg_sq"], qgroup_size)
            v_tri = _dequant_basic(s_tri["exp_avg_sq"], s_tri["scale_exp_avg_sq"], qgroup_size)

        torch.testing.assert_close(m_tri, m_ref, atol=3e-2, rtol=5e-2)
        torch.testing.assert_close(v_tri, v_ref, atol=3e-2, rtol=5e-2)

    def test_basic_partial_group_parity(self):
        self._run_parity_case(numel=383, expansion_mode="false")

    def test_expand_partial_group_parity(self):
        self._run_parity_case(numel=383, expansion_mode="expand")

    def test_basic_full_group_parity(self):
        self._run_parity_case(numel=512, expansion_mode="false")

    def test_expand_full_group_parity(self):
        self._run_parity_case(numel=512, expansion_mode="expand")


class TestTritonCoatAdamWE5M2Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _require_cuda_stack()
        if not hasattr(torch, "float8_e5m2"):
            raise unittest.SkipTest("torch.float8_e5m2 is not available in this torch build.")

    def test_e5m2_state_smoke(self):
        qargs = _build_qargs(expansion="false", first_order_bit="E5M2", second_order_bit="E5M2")
        p = torch.nn.Parameter(torch.randn(257, device="cuda", dtype=torch.float32))
        p.grad = torch.randn_like(p)

        opt = TritonCoatAdamW([p], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, qargs=qargs)
        opt.step()

        self.assertFalse(torch.isnan(p).any().item())
        self.assertFalse(torch.isinf(p).any().item())

        state = opt.state[p]
        self.assertEqual(state["exp_avg"].dtype, torch.float8_e5m2)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.float8_e5m2)


if __name__ == "__main__":
    unittest.main()
