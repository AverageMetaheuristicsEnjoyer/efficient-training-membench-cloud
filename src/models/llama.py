import math
import sys
import os

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base import GPTBase


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    return torch.stack((cos_freqs, sin_freqs), dim=-1)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape[:-1] == (x.shape[1], x.shape[-2])
    shape = [
        1 if i != 1 and i != ndim - 2 else d for i, d in enumerate(x.shape[:-1])
    ] + [2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    q = q.float().reshape(*q.shape[:-1], -1, 2)
    k = k.float().reshape(*k.shape[:-1], -1, 2)

    freqs_cis = _reshape_for_broadcast(freqs_cis, q)

    q_cos = q[..., 0] * freqs_cis[..., 0] - q[..., 1] * freqs_cis[..., 1]
    q_sin = q[..., 0] * freqs_cis[..., 1] + q[..., 1] * freqs_cis[..., 0]
    k_cos = k[..., 0] * freqs_cis[..., 0] - k[..., 1] * freqs_cis[..., 1]
    k_sin = k[..., 0] * freqs_cis[..., 1] + k[..., 1] * freqs_cis[..., 0]

    q_out = torch.stack((q_cos, q_sin), dim=-1).reshape(q.shape).flatten(3)
    k_out = torch.stack((k_cos, k_sin), dim=-1).reshape(k.shape).flatten(3)

    return q_out, k_out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def _mlp_hidden_dim(config) -> int:
    hidden_dim = config.n_embd * 4
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = config.multiple_of * (
        (hidden_dim + config.multiple_of - 1) // config.multiple_of
    )
    return hidden_dim


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_dim = config.n_embd // config.n_head
        inner_dim = config.n_head * head_dim
        self.q_proj = nn.Linear(config.n_embd, inner_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, inner_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, config.n_embd, bias=False)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = _mlp_hidden_dim(config)
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)


class LlamaBlock(nn.Module):
    """Unified transformer block. BF16 or FP8 path is selected at runtime."""

    def __init__(self, config, qargs=None, layer_idx: int = 0):
        super().__init__()
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout  = config.dropout
        self.qkv_clipping = getattr(config, "qkv_clipping", False)
        self.qkv_clipping_factor = getattr(config, "qkv_clipping_factor", 1.0)

        # Canonical construction order — identical across BF16 and FP8:
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn = Attention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp  = MLP(config)

        # FP8-only cache — adds no nn.Parameter / nn.Linear children, so it
        # cannot perturb seed parity or optimizer-facing parameter structure.
        self.fp8 = qargs is not None
        if self.fp8:
            # Lazy import so BF16 runs don't require triton/COAT deps.
            from third_party.coat.utils._fp8_weightcache import FP8CacheWeightModule

            self.qargs = qargs
            self.fwobits = {
                "fabit": qargs.fabit, "fwbit": qargs.fwbit, "fobit": qargs.fobit,
                "babit": qargs.babit, "bwbit": qargs.bwbit, "bobit": qargs.bobit,
            }
            self.fp8_cache = FP8CacheWeightModule(None, qargs, layer_idx)
            # prepare_weight() reads self.fp8_cache.fwobits["fwbit"] internally.
            self.fp8_cache.fwobits = self.fwobits

    def _project_qkv(self, x):
        B, T, C = x.size()
        x_norm = self.ln_1(x)
        q = self.attn.q_proj(x_norm)
        k = self.attn.k_proj(x_norm)
        v = self.attn.v_proj(x_norm)

        if self.qkv_clipping:
            c = self.qkv_clipping_factor
            q = q.clamp(min=-c, max=c)
            k = k.clamp(min=-c, max=c)
            v = v.clamp(min=-c, max=c)

        return x_norm, q, k, v

    def _non_fp8_forward(self, x, freqs_cis):
        B, T, C = x.size()
        _, q, k, v = self._project_qkv(x)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn.o_proj(y)

        h = self.ln_2(x)
        x = x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))
        return x

    def _fp8_forward(self, x, Qx, Sx, freqs_cis):
        from third_party.coat.utils._fp8manager import FP8Manager
        from models._fp8_ops import (
            FP8BeforeAttentionResidual,
            FP8AfterAttentionResidual,
            FP8MLPResidual,
        )

        B, T, C = x.size()
        is_first = FP8Manager.is_first_microbatch
        gs = self.qargs.group_size

        with torch.no_grad():
            w_q_s = self.fp8_cache.prepare_weight(self.attn.q_proj.weight, "q_proj", is_first)
            w_k_s = self.fp8_cache.prepare_weight(self.attn.k_proj.weight, "k_proj", is_first)
            w_v_s = self.fp8_cache.prepare_weight(self.attn.v_proj.weight, "v_proj", is_first)
            w_o_s = self.fp8_cache.prepare_weight(self.attn.o_proj.weight, "o_proj", is_first)
            w_g_s = self.fp8_cache.prepare_weight(self.mlp.gate_proj.weight, "gate_proj", is_first)
            w_u_s = self.fp8_cache.prepare_weight(self.mlp.up_proj.weight,   "up_proj",   is_first)
            w_d_s = self.fp8_cache.prepare_weight(self.mlp.down_proj.weight, "down_proj", is_first)

        residual, q, k, v = FP8BeforeAttentionResidual.apply(
            x, Qx, Sx,
            self.attn.q_proj.weight, None, None, w_q_s,
            self.attn.k_proj.weight, None, None, w_k_s,
            self.attn.v_proj.weight, None, None, w_v_s,
            self.ln_1.weight,
            gs, self.fwobits, self.qargs,
        )

        if self.qkv_clipping:
            c = self.qkv_clipping_factor
            q = q.clamp(min=-c, max=c)
            k = k.clamp(min=-c, max=c)
            v = v.clamp(min=-c, max=c)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        hidden, Qx, Sx = FP8AfterAttentionResidual.apply(
            residual, attn_out,
            self.attn.o_proj.weight, None, None, w_o_s,
            gs, self.fwobits, self.qargs,
        )

        hidden, Qx, Sx = FP8MLPResidual.apply(
            hidden, Qx, Sx,
            self.mlp.gate_proj.weight, None, None, w_g_s,
            self.mlp.up_proj.weight,   None, None, w_u_s,
            self.mlp.down_proj.weight, None, None, w_d_s,
            self.ln_2.weight,
            gs, self.fwobits, self.qargs,
        )

        return hidden, Qx, Sx

    def forward(self, x, Qx, Sx, freqs_cis):
        if self.fp8 and self.training:
            return self._fp8_forward(x, Qx, Sx, freqs_cis)
        return self._non_fp8_forward(x, freqs_cis), None, None


class Llama(GPTBase):
    def __init__(self, config):
        nn.Module.__init__(self)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.fp8 = bool(getattr(config, "fp8", False))
        qargs = getattr(config, "qargs", None) if self.fp8 else None
        if self.fp8:
            assert qargs is not None, (
                "Llama with --fp8 requires a QuantizationConfig on args.qargs "
                "(built by src/main.py when --fp8 is passed)."
            )
        self.qargs = qargs
        self._tokenizer = None

        self.head_dim  = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([
                LlamaBlock(config, qargs, i) for i in range(config.n_layer)
            ]),
            ln_f = RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Init after shared parameters are constructed. FP8-only modules below
        # have no nn.Parameter children and are registered afterwards so they
        # cannot perturb RNG consumption.
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=self.config.init_std / math.sqrt(2 * config.n_layer),
                )

        if self.fp8:
            from third_party.coat.activation.real_quantization import (
                Coat_quantize_bgn, Coat_quantize_end,
            )
            self.quantize_input  = Coat_quantize_bgn(qargs)
            self.quantize_output = Coat_quantize_end(qargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def get_num_params(self, non_embedding=True):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.config.sequence_length}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        x = self.transformer.drop(self.transformer.wte(idx))
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        if self.fp8:
            x, Qx, Sx = self.quantize_input(x)
        else:
            Qx, Sx = None, None

        for block in self.transformer.h:
            x, Qx, Sx = block(x, Qx, Sx, freqs_cis)

        if self.fp8:
            x = self.quantize_output(x, Qx, Sx)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return {
            "logits": logits if get_logits else None,
            "loss": loss,
        }
