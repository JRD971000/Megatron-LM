# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""VA unit test for the Gemma4 ``gemma`` assistant-only SFT loss-masking format.

The Gemma4 (E4B) chat template is stateful and forward-scanning (role:tool
messages render inside the preceding model turn), so the per-turn
re-tokenization masking loop used by the nemotron formats does not apply. The
``gemma`` format instead renders with a copy of the REAL template augmented with
``{% generation %}...{% endgeneration %}`` markers around ONLY the
model-generated emits, then uses HF ``return_assistant_tokens_mask`` to build
the loss target.

This test asserts:
  (a) token-identity: the augmented-template render is token-identical to the
      original-template render (both passed explicitly via ``chat_template=``) --
      generation markers are jinja no-ops, so any diff is a transcription error;
  (b) the assistant mask is 1 EXACTLY on the reasoning (thought channel),
      tool_call, final-content and model-turn-closing spans, and 0 on the
      system/tools block, user, tool responses and structural tokens.

Model-turn-closing decision: the closing ``<turn|>`` of a *model* turn IS
trained (mask == 1) so the model learns to stop; the ``<turn|>`` of user turns
and the system-block ``<turn|>`` are masked. This matches the STAGE-3.5 spec
recommendation.

Notes on the runtime environment:
  * ``sft_tokenizer.py`` has no ``megatron`` imports, so it is loaded standalone
    via ``importlib`` to avoid the ``megatron.core`` package ``__init__`` chain
    (which asserts on an ``nvidia-resiliency-ext`` pre-release version in the
    26.06 container). Run with ``--noconftest`` for the same reason.
  * The test requires the real E4B fast tokenizer on lustre; it skips otherwise.
"""
import importlib.util
import os

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

TOKENIZER_PATH = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/"
    "ataghibakhsh/gemma4-playground/weights/gemma-4-E4B-it"
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(TOKENIZER_PATH),
    reason=f"Gemma4 E4B tokenizer not found at {TOKENIZER_PATH}",
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SFT_PATH = os.path.join(
    _REPO_ROOT, "megatron/core/tokenizers/text/libraries/sft_tokenizer.py"
)


def _load_sft_module():
    spec = importlib.util.spec_from_file_location("sft_tokenizer_standalone", _SFT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SFT = _load_sft_module()
SFTTokenizer = _SFT.SFTTokenizer
IGNORE_INDEX = _SFT.IGNORE_INDEX
GEMMA_TEMPLATE = _SFT.gemma4_assistant_masked_template


# ---- fixture conversation (trimmed from the target tool-calling dataset's first record) ----
SYSTEM = "You are a helpful construction-dispute assistant."
USER = "I want to dispute weather-delay charges for project COM-3388."
REASONING = (
    "I need to help this user with their dispute about weather-related delay "
    "charges for project COM-3388. First, I should understand the project "
    "details and status. Let me get the project details first."
)
TOOL_RESP = '{"contract_amount": 600000, "contingency_pct": 10, "location": "123 Main St"}'
FINAL = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT FOR ASSISTANCE."

CONVERSATION = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "id": "call_abc",
                "function": {
                    "name": "get_project_details",
                    "arguments": '{"project_id": "COM-3388"}',
                },
            }
        ],
        "reasoning_content": REASONING,
        "content": "",
    },
    {"role": "tool", "tool_call_id": "call_abc", "content": TOOL_RESP},
    {"role": "assistant", "content": FINAL, "reasoning_content": ""},
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_project_details",
            "description": (
                "Access project specifications including contract amount and contingency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project identifier"}
                },
                "required": ["project_id"],
            },
        },
    }
]


@pytest.fixture(scope="module")
def raw_tokenizer():
    return transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)


@pytest.fixture(scope="module")
def sft_tokenizer():
    return SFTTokenizer(TOKENIZER_PATH, "gemma")


def _apply(tokenizer, chat_template):
    out = tokenizer.apply_chat_template(
        CONVERSATION,
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        chat_template=chat_template,
    )
    return list(out["input_ids"])


def _masked_runs(mask):
    runs = []
    i = 0
    while i < len(mask):
        if mask[i] == 1:
            j = i
            while j < len(mask) and mask[j] == 1:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def test_augmented_template_is_token_identical(raw_tokenizer, sft_tokenizer):
    """The generation markers must be jinja no-ops: augmented == original tokens."""
    original_ids = _apply(raw_tokenizer, raw_tokenizer.chat_template)
    augmented_ids = _apply(raw_tokenizer, GEMMA_TEMPLATE)

    # Guard against the degenerate no-template render (a handful of tokens).
    assert len(original_ids) > 50, "original-template render looks degenerate"
    assert augmented_ids == original_ids, "augmented template perturbs tokenization"

    # The SFTTokenizer 'gemma' path must reproduce exactly these tokens.
    sft_tokens = sft_tokenizer.tokenize_conversation(
        CONVERSATION, return_target=False, add_generation_prompt=False, tools=TOOLS
    )
    assert list(np.asarray(sft_tokens)) == original_ids


def test_assistant_mask_spans(raw_tokenizer, sft_tokenizer):
    tokens, target = sft_tokenizer.tokenize_conversation(
        CONVERSATION, return_target=True, add_generation_prompt=False, tools=TOOLS
    )
    tokens = np.asarray(tokens)
    target = np.asarray(target)

    assert tokens.shape == target.shape
    mask = (target != IGNORE_INDEX).astype(int)
    # Where trained, target must equal the input token (identity), never a shifted/other id.
    assert np.array_equal(target[mask == 1], tokens[mask == 1])

    runs = _masked_runs(mask)
    decoded_runs = [raw_tokenizer.decode(tokens[a:b].tolist()) for (a, b) in runs]

    # Exactly two trained spans:
    #   run 0: reasoning (thought channel) + tool_call  (the reasoning turn)
    #   run 1: final assistant content + the model-turn-closing <turn|>
    assert len(runs) == 2, f"expected 2 trained runs, got {len(runs)}: {decoded_runs}"

    run0, run1 = decoded_runs

    # --- run 0: reasoning thought channel + tool_call span ---
    assert run0.startswith("<|channel>thought")
    assert REASONING in run0
    assert "<channel|>" in run0
    assert "<|tool_call>call:get_project_details" in run0
    assert run0.endswith("<tool_call|>")

    # --- run 1: final content + model-turn-closing token (trained so the model stops) ---
    assert run1 == FINAL + "<turn|>\n", repr(run1)

    # --- everything else is masked (mask == 0) ---
    zero_text = raw_tokenizer.decode(tokens[mask == 0].tolist())
    # structural / prompt / environment content must NOT be trained:
    assert SYSTEM in zero_text
    assert USER in zero_text
    assert "contract_amount" in zero_text  # tool response (environment-generated)
    assert "declaration:get_project_details" in zero_text  # tools block
    assert "<|turn>user" in zero_text  # role prefix
    assert "<|tool_response>" in zero_text  # forward-scanned tool response wrapper
    # and the model-generated content must NOT leak into the masked region:
    assert REASONING not in zero_text
    assert FINAL not in zero_text
