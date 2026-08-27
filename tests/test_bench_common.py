"""The parts of the dense sweep that can be wrong without a GPU noticing."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.common import (  # noqa: E402
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TOKENS_PER_STEP,
    MICROBATCHES,
    MODEL_SPECS,
    VARIANTS,
    accumulation_steps,
    expected_parameters,
    mlp_hidden_dim,
    model_spec,
    variant_spec,
)


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda spec: spec["name"])
def test_recorded_parameter_count_follows_from_the_geometry(spec):
    """The worker refuses a model that misses this count, so the count must be right."""
    assert expected_parameters(spec) == spec["parameters_expected"]


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda spec: spec["name"])
def test_head_dimension_divides(spec):
    assert spec["n_embd"] % spec["n_head"] == 0


def test_mlp_hidden_matches_the_upstream_rounding():
    # src/models/llama.py::_mlp_hidden_dim, at the three widths the sweep uses.
    assert mlp_hidden_dim(1024) == 2816
    assert mlp_hidden_dim(1280) == 3584
    assert mlp_hidden_dim(2048) == 5632


def test_257m_agrees_with_the_sibling_benchmark():
    """The same Llama at the same geometry is measured in effective-muon-membench."""
    assert model_spec("257m")["parameters_expected"] == 257_188_864
    assert model_spec("1p4b")["parameters_expected"] == 1_439_270_912


def test_every_micro_batch_processes_the_same_tokens_per_step():
    tokens = set()
    for microbatch in MICROBATCHES:
        accumulation = accumulation_steps(
            DEFAULT_TOKENS_PER_STEP, microbatch, DEFAULT_SEQUENCE_LENGTH
        )
        tokens.add(microbatch * accumulation * DEFAULT_SEQUENCE_LENGTH)
    assert tokens == {DEFAULT_TOKENS_PER_STEP}


def test_a_micro_batch_that_does_not_divide_the_step_is_refused():
    with pytest.raises(ValueError, match="not divisible"):
        accumulation_steps(DEFAULT_TOKENS_PER_STEP, 3, DEFAULT_SEQUENCE_LENGTH)


def test_the_variant_grid_is_three_optimizers_by_three_precisions():
    assert len(VARIANTS) == 9
    assert {variant["optimizer"] for variant in VARIANTS} == {"adamw", "muon", "soap"}
    for optimizer in ("adamw", "muon", "soap"):
        names = {variant["name"] for variant in VARIANTS if variant["optimizer"] == optimizer}
        assert names == {
            f"{optimizer}_bf16_state_fp32",
            f"{optimizer}_fp8gemm_state_fp32",
            f"{optimizer}_bf16_state_fp8",
        }


def test_variant_names_match_the_moe_half_of_the_benchmark():
    """H-MoE-Part-cloud's stage3_moe.ARMS, so the two tables join on this column."""
    moe_arms = {
        "adamw_bf16_state_fp32",
        "adamw_bf16_state_fp8",
        "muon_bf16_state_fp32",
        "muon_bf16_state_fp8",
        "adamw_fp8gemm_state_fp32",
        "muon_fp8gemm_state_fp32",
    }
    assert moe_arms <= {variant["name"] for variant in VARIANTS}


def test_exactly_one_variant_per_optimizer_is_the_baseline():
    baselines = [variant for variant in VARIANTS
                 if variant["gemm"] == "bf16" and variant["state"] == "fp32"]
    assert len(baselines) == 3


def test_unknown_names_are_refused():
    with pytest.raises(KeyError):
        model_spec("nope")
    with pytest.raises(KeyError):
        variant_spec("nope")
