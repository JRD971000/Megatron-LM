# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""R1 unit test for the Gemma4 packed-SFT per-document position-reset helpers.

Under packing a single ``[1, s]`` sequence concatenates several documents, and Gemma4
RoPE reads ``position_ids`` directly, so positions MUST reset to 0 at every document
boundary (TASK-001 R1). ``gemma4_model`` provides two helpers:

  * ``_reset_position_ids_from_cu_seqlens`` derives per-document reset positions from a
    (padded) cu_seqlens, e.g. ``[0, 3, 7] -> [0, 1, 2, 0, 1, 2, 3]`` (parity path);
  * ``_assert_position_ids_reset`` verifies a caller-supplied ``position_ids`` already
    resets at the cu_seqlens boundaries, raising on a flat ``arange`` (training path).

Runtime environment: importing ``megatron.core`` triggers the package ``__init__`` chain
that asserts on the container's ``nvidia-resiliency-ext`` pre-release ``__version__``.
``examples/gemma4/gemma4_common`` applies the ``__version__`` shim and sets up ``sys.path``,
so it MUST be imported before ``megatron`` (see the nvrx note in TASK-003). Run with
``--noconftest``.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "examples", "gemma4"))

# Apply the nvrx __version__ shim + sys.path setup BEFORE importing megatron.
import gemma4_common  # noqa: F401,E402

from megatron.core.models.gemma4.gemma4_model import (  # noqa: E402
    _assert_position_ids_reset,
    _reset_position_ids_from_cu_seqlens,
)


def test_reset_position_ids_from_cu_seqlens_basic():
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    pos = _reset_position_ids_from_cu_seqlens(cu)
    assert pos.tolist() == [0, 1, 2, 0, 1, 2, 3]


def test_reset_position_ids_single_segment_equals_flat():
    # One document -> reset positions equal a flat arange (no boundary to reset at).
    cu = torch.tensor([0, 5], dtype=torch.int32)
    pos = _reset_position_ids_from_cu_seqlens(cu)
    assert pos.tolist() == [0, 1, 2, 3, 4]


def test_reset_position_ids_many_segments():
    cu = torch.tensor([0, 2, 4, 8], dtype=torch.int32)
    pos = _reset_position_ids_from_cu_seqlens(cu)
    assert pos.tolist() == [0, 1, 0, 1, 0, 1, 2, 3]


def test_assert_accepts_reset_positions():
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    reset = _reset_position_ids_from_cu_seqlens(cu).unsqueeze(0)  # [1, s]
    # Should not raise.
    _assert_position_ids_reset(reset, cu)


def test_assert_raises_on_flat_positions():
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    flat = torch.arange(7, dtype=torch.int64).unsqueeze(0)  # [0..6], never resets
    with pytest.raises(AssertionError, match="R1"):
        _assert_position_ids_reset(flat, cu)


def test_assert_single_segment_flat_ok():
    # With one document, a flat arange is a valid reset (start position is 0).
    cu = torch.tensor([0, 5], dtype=torch.int32)
    flat = torch.arange(5, dtype=torch.int64).unsqueeze(0)
    _assert_position_ids_reset(flat, cu)


if __name__ == "__main__":
    test_reset_position_ids_from_cu_seqlens_basic()
    test_reset_position_ids_single_segment_equals_flat()
    test_reset_position_ids_many_segments()
    test_assert_accepts_reset_positions()
    test_assert_raises_on_flat_positions()
    test_assert_single_segment_flat_ok()
    print("ALL R1 POSITION-RESET TESTS PASSED")
