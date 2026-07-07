# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import atexit
import json
import os
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split

IGNORE_INDEX = -100


def _strip_none(obj):
    """Recursively drop dict keys whose value is None -- EXCEPT inside
    ``tool_calls[].function.arguments`` subtrees.

    Raw tool-calling data legitimately carries null message fields (e.g.
    ``"content": null`` on tool-call-only assistant turns) that the chat template
    cannot render ('can only concatenate str (not "NoneType")'), so nulls are
    pruned: for the template an absent key and a ``None`` key are equivalent (it
    uses ``.get(...)``).

    Null VALUES inside a tool call's ``arguments`` dict are different: they are
    semantic argument values. The HF reference pipeline never strips them and the
    ground-truth chat template renders them via jinja stringification
    (``city:None``), so stripping there would silently drop supervised argument
    keys from the training target. The ``arguments`` subtree is therefore
    preserved verbatim. MUST stay in sync with
    examples/gemma4/pack_sft_dataset.py::strip_none.
    """
    if isinstance(obj, dict):
        return {
            k: (v if k == "arguments" else _strip_none(v))
            for k, v in obj.items()
            if v is not None
        }
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


class SFTLowLevelDataset:
    """The low-level dataset loading jsonl data for SFT

    Reads the jsonl directly via a byte-offset index (seek + json.loads per row)
    instead of ``datasets.load_dataset("json", ...)``. The Arrow-backed loader is
    unusable for real tool-calling corpora: Arrow unifies one schema over the whole
    file and (a) fails outright on mixed-type fields (e.g. a tool-schema ``const``
    that is a string in one record and a number in another:
    "Column(...anyOf/[]/const) changed from string to number"), (b) injects null
    fields into every record (union of all keys), and (c) needs tens of GB of RAM
    and cache to generate the split for a 30GB+ file. The offset index costs one
    sequential scan, is cached next to the jsonl (``<path>.idx.npy``), and rows
    round-trip byte-exact.

    ``_strip_none`` is still applied on read: real data legitimately carries null
    fields (e.g. ``"content": null`` on tool-call-only assistant turns) that the
    chat template cannot render.

    Args:
        dataset_path (str): The path to jsonl data
            Each line of the jsonl must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
            A jsonl line can contain multiple conversations packed together into on list. Each
            conversation starts with the system role, and conversations can have multiple turns
            of the user and assistant roles.
    """

    def __init__(self, dataset_path: str) -> None:
        self._path = dataset_path
        self._offsets = self._load_or_build_index(dataset_path)
        self._handles: Dict[int, Any] = {}  # pid -> file handle (fork/worker safe)

    @staticmethod
    def _load_or_build_index(dataset_path: str) -> np.ndarray:
        """Byte offset of every line start; cached as <path>.idx.npy.

        The cache is only reused when newer than the jsonl. Concurrent builders
        (multiple ranks) all compute identical content and publish via atomic
        rename, so races are benign.
        """
        idx_path = dataset_path + ".idx.npy"
        try:
            if os.path.getmtime(idx_path) >= os.path.getmtime(dataset_path):
                return np.load(idx_path)
        except OSError:
            pass
        offsets = []
        pos = 0
        with open(dataset_path, "rb") as f:
            for line in f:
                offsets.append(pos)
                pos += len(line)
        arr = np.array(offsets, dtype=np.int64)
        try:
            tmp = f"{idx_path}.{os.getpid()}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, idx_path)
        except OSError:
            pass  # read-only location etc. -- index just isn't cached
        return arr

    def _row(self, idx: int) -> Dict[str, Any]:
        pid = os.getpid()
        handle = self._handles.get(pid)
        if handle is None:
            handle = open(self._path, "rb")
            self._handles[pid] = handle
        handle.seek(int(self._offsets[idx]))
        return json.loads(handle.readline())

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> list:
        # Strip null-valued keys so the chat template renders the original
        # per-record structure -- see _strip_none.
        return _strip_none(self._row(idx)["messages"])

    def get_tools(self, idx: int) -> Optional[list]:
        """Return the optional per-record tool definitions (or None)."""
        return _strip_none(self._row(idx).get("tools"))

    def get_tools_per_conversation(self, idx: int) -> Optional[list]:
        """Return the optional per-conversation tool definitions (or None).

        Pre-packed rows (see examples/gemma4/pack_sft_dataset.py) concatenate
        conversations from different source records, each with its own tools. Such
        rows carry a ``tools_per_conversation`` list aligned with the conversations
        obtained by splitting ``messages`` on system-role messages; element ``i``
        (a tool list or None) applies to conversation ``i`` only. Takes precedence
        over the row-level ``tools`` key when present.
        """
        return _strip_none(self._row(idx).get("tools_per_conversation"))


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def _split_conversations(self, merged_conversations):
        split_conversations = []
        current = []
        for msg in merged_conversations:
            # Whenever we see a new system message, start a new conversation
            if msg["role"] == "system":
                if current:  # If previously accumulating a conversation, then store it
                    split_conversations.append(current)
                current = [msg]  # Then start the new conversation
            else:
                current.append(msg) # Continue accumulating the current conversation
        if current:  # Store any remaining conversation
            split_conversations.append(current)
        return split_conversations

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length

        row_idx = int(self.indices[idx % len(self.indices)])
        merged_conversations = self.dataset[row_idx]
        # Optional per-record tool definitions (used by tool-calling chat templates,
        # e.g. the Gemma4 assistant-masked format).
        tools = self.dataset.get_tools(row_idx) if hasattr(self.dataset, "get_tools") else None
        # Pre-packed rows carry per-conversation tools (element i applies to
        # conversation i); they take precedence over the row-level ``tools``.
        tools_per_conversation = (
            self.dataset.get_tools_per_conversation(row_idx)
            if hasattr(self.dataset, "get_tools_per_conversation")
            else None
        )
        split_conversations = self._split_conversations(merged_conversations)
        if tools_per_conversation is not None:
            assert len(tools_per_conversation) == len(split_conversations), (
                f"tools_per_conversation has {len(tools_per_conversation)} entries but "
                f"messages split into {len(split_conversations)} conversations "
                f"(row {row_idx})"
            )

        def extend_with_padding(tokens, targets, positions, pad_len):
            tokens.extend([pad] * pad_len)
            targets.extend([pad] * pad_len)
            positions.extend(range(positions[-1]+1, positions[-1]+1+pad_len))

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]
        eod = tokenizer.eod
        pad = tokenizer.pad
        # TODO(duncan): Track number of convs dropped and/or truncated and amount of end-padding
        for conv_idx, conversation in enumerate(split_conversations):

            conv_tools = (
                tools_per_conversation[conv_idx]
                if tools_per_conversation is not None
                else tools
            )
            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True, add_generation_prompt=False, tools=conv_tools
            )

            tokens_list = tokens.tolist()
            targets_list = targets.tolist()


            pack_tokens.extend(tokens_list)
            pack_targets.extend(targets_list)

            assert not self.config.reset_position_ids
            pack_positions.extend(range(len(tokens_list)))

            if self.config.context_parallel_size > 1:
                pad_granularity = self.config.context_parallel_size * 2
                mod_token_count = len(pack_tokens) % pad_granularity
                if mod_token_count != 0:
                    pad_len = pad_granularity - mod_token_count
                    extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)

            # TODO(duncan): Consider also padding to multiple of number of tokens here. This might
            # be needed for efficiency (and potentially set via command-line argument).

            cu_seqlens.append(len(pack_tokens))

            # Handle any necessary truncation
            if len(pack_tokens) >= pack_length + 1:  # +1 here to account for later alignment
                # A packer-merged row (carries tools_per_conversation) is sized to fit
                # sequence_length by construction; overflowing here means the packer ran
                # with a different --seq-length or --pad-granularity (must be
                # 2*context_parallel_size) than this training config. Truncating would
                # silently drop the tail conversations' supervised tokens, so fail loudly.
                assert tools_per_conversation is None, (
                    f"pre-packed row {row_idx} overflows sequence_length={pack_length} at "
                    f"train time (packed with a different --seq-length or a "
                    f"--pad-granularity != 2*context_parallel_size?)"
                )
                # Truncate on the right
                max_body = pack_length
                pack_tokens = pack_tokens[:max_body]
                pack_targets = pack_targets[:max_body]
                pack_tokens.append(pad)
                pack_targets.append(pad)
                pack_positions = pack_positions[:pack_length+1]
                # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
                cu_seqlens[-1] = len(pack_tokens) - 1
                break

        # Handle any necessary padding
        if len(pack_tokens) < pack_length + 1:  # +1 here to account for later alignment
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)
            # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
            cu_seqlens[-1] = len(pack_tokens) - 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1

        # Align and convert to tensors
        input_ids    = torch.tensor(pack_tokens[:-1],  dtype=torch.int64)
        labels       = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        # Loss mask.
        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0  # Mask paddings
        loss_mask[labels == IGNORE_INDEX] = 0.0  # mask prompts

        # TODO(duncan): Optionally create an attention mask
        assert not self.config.create_attention_mask and not self.config.reset_attention_mask
        # attention_mask = None

        assert len(cu_seqlens) >= 2
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        # Calculating max_seqlen here, rather than incrementally above, because of possible
        # effects of truncation and padding
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()  # max_seqlen is a 0-D tensor

        return {
            'tokens': input_ids,
            'labels': labels,
            # 'attention_mask': attention_mask,  # PyTorch collate cannot handle NoneType
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
        }
