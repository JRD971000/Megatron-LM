# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Gradient clipping."""

from typing import List, Optional, Union

import torch
from torch import inf

try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
        multi_tensor_scale,
        multi_tensor_scale_tensor,
    )

    l2_norm_impl = multi_tensor_l2norm
    multi_tensor_scale_impl = multi_tensor_scale
    multi_tensor_scale_tensor_impl = multi_tensor_scale_tensor
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier

        l2_norm_impl = amp_C.multi_tensor_l2norm
        multi_tensor_scale_impl = amp_C.multi_tensor_scale
        multi_tensor_scale_tensor_impl = None
    except ImportError:
        import warnings

        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of multi_tensor_applier, '
            'multi_tensor_l2norm, and multi_tensor_scale'
        )

        from megatron.core.utils import (
            local_multi_tensor_applier,
            local_multi_tensor_l2_norm,
            local_multi_tensor_scale,
        )

        multi_tensor_applier = local_multi_tensor_applier
        l2_norm_impl = local_multi_tensor_l2_norm
        multi_tensor_scale_impl = local_multi_tensor_scale
        multi_tensor_scale_tensor_impl = None


from ..tensor_parallel import param_is_not_tensor_parallel_duplicate
from ..transformer.module import param_is_not_shared
from ..utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor

# TE/Apex multi-tensor kernels store each tensor's numel in a 32-bit int
# (TensorListMetadata.sizes); a tensor with numel > 2**31 - 1 wraps negative and
# the kernel reads/writes out of bounds (observed: CUDA illegal memory access in
# multi_tensor_l2norm on a 2.82B-element replicated-embedding main grad at DP=1,
# where the distributed optimizer cannot shard it below 2**31).
_MULTI_TENSOR_INT32_NUMEL_LIMIT = 2**31 - 1
_MULTI_TENSOR_SPLIT_CHUNK = 2**30


def _split_int32_oversized_tensors(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Split tensors too large for the int32 multi-tensor kernels into views.

    Returns flat sub-views (< 2**31 elements each) sharing the original storage,
    so in-place ops (scale) still hit the real tensor and reductions (l2 norm)
    are exact. Returns the input list unchanged when nothing is oversized.
    """
    if all(t.numel() <= _MULTI_TENSOR_INT32_NUMEL_LIMIT for t in tensors):
        return tensors
    split = []
    for t in tensors:
        if t.numel() <= _MULTI_TENSOR_INT32_NUMEL_LIMIT:
            split.append(t)
            continue
        assert t.is_contiguous(), (
            "cannot split a non-contiguous tensor with numel > 2**31-1 for the "
            "int32 multi-tensor kernels"
        )
        split.extend(t.view(-1).split(_MULTI_TENSOR_SPLIT_CHUNK))
    return split


def get_grad_norm_fp32(
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    norm_type: Union[int, float] = 2,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Calculate the p-norm of gradients in FP32 precision.

    This function is adapted from `torch.nn.utils.clip_grad.clip_grad_norm_`
    and extends it with functionality to handle model-parallel parameters.
    It ensures that the norm is correctly computed and reduced across
    the specified process group (typically the model-parallel group for
    non-distributed optimizers or the entire world for distributed optimizers).

    Args:
        grads_for_norm (Union[List[torch.Tensor], torch.Tensor]): An iterable
            of Tensors or a single Tensor used to calculate the gradient norm.
        norm_type (Union[int, float]): The type of the p-norm to use. Can be
            'inf' for infinity norm. Defaults to 2.
        grad_stats_parallel_group (ProcessGroup, optional): The process group
            used for reducing gradient statistics (e.g., norms and zero counts).

    Returns:
        float: The total norm of the parameters, treated as a single vector.
    """

    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    data_parallel_group = None
    for grad in grads_for_norm:
        data_parallel_group = get_data_parallel_group_if_dtensor(grad, data_parallel_group)

    grads_for_norm = [to_local_if_dtensor(grad) for grad in grads_for_norm]

    # Norm parameters.
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
            )
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=grad_stats_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:
                grad_norm, _ = multi_tensor_applier(
                    l2_norm_impl,
                    dummy_overflow_buf,
                    # Split >2**31-1-element grads into views (int32 kernel limit);
                    # exact for the whole-list norm computed here.
                    [_split_int32_oversized_tensors(grads_for_norm)],
                    False,  # no per-parameter norm
                )
            else:
                grad_norm = torch.zeros(1, dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm**norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm**norm_type

        # Sum across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
            )
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
        )
        if multi_tensor_scale_tensor_impl is not None:
            total_norm = total_norm.pow(1.0 / norm_type)
        else:
            total_norm = total_norm.item() ** (1.0 / norm_type)

    return total_norm


def clip_grad_by_total_norm_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    max_norm: Union[int, float],
    total_norm: float,
    use_decoupled_grad: bool = False,
):
    """Clips the gradients of an iterable of parameters in FP32 by total norm.

    Note that the gradients are modified in-place.

    Args:
        parameters (Union[List[torch.Tensor], torch.Tensor]): An iterable of
            Tensors or a single Tensor that will have gradients normalized.
        max_norm (Union[int, float]): The maximum permissible total norm
            of the gradients.
        total_norm (float): The current total norm of the gradients.
        use_decoupled_grad (bool, optional): Whether to read from the
            '.decoupled_grad' attribute instead of the standard '.grad'.
            Defaults to False.
    """
    # Grads.
    params = []
    grads = []
    for param in parameters:
        if use_decoupled_grad:
            if hasattr(param, "decoupled_grad") and param.decoupled_grad is not None:
                assert param.decoupled_grad.dtype in [torch.float32, torch.bfloat16]
                params.append(param)
                grads.append(to_local_if_dtensor(param.decoupled_grad).detach())
        else:
            if param.grad is not None:
                assert param.grad.type() == 'torch.cuda.FloatTensor'
                params.append(param)
                grads.append(to_local_if_dtensor(param.grad).detach())

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')
    # Split >2**31-1-element grads into views (int32 kernel limit); the scale is
    # elementwise in-place on views of the same storage, so this is exact.
    grads = _split_int32_oversized_tensors(grads)
    if isinstance(clip_coeff, torch.Tensor):
        clip_coeff.clamp_max_(1.0)
        assert (
            multi_tensor_scale_tensor_impl is not None
        ), "clip_coeff is tensor type. But multi_tensor_scale_tensor not available."
        multi_tensor_applier(
            multi_tensor_scale_tensor_impl, dummy_overflow_buf, [grads, grads], clip_coeff
        )
    elif clip_coeff < 1.0:
        multi_tensor_applier(
            multi_tensor_scale_impl, dummy_overflow_buf, [grads, grads], clip_coeff
        )


def count_zeros_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    grad_stats_parallel_group: torch.distributed.ProcessGroup,
    use_decoupled_grad: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Counts the number of zero values in the gradients of the given parameters.

    The count is performed in FP32. This method filters parameters to ensure
    gradients are not double-counted by checking if the gradient is not None,
    the parameter is not shared, and the parameter is not a replica due
    to tensor model parallelism. It also handles parameters managed by
    Megatron FSDP specifically.

    Args:
        parameters (Union[List[torch.Tensor], torch.Tensor]): An iterable of
            Tensors or a single Tensor whose gradients will be checked for zeros.
        grad_stats_parallel_group (ProcessGroup): The process group used for
            reducing the zero count across distributed ranks.
        use_decoupled_grad (bool, optional): If True, reads from the
            '.decoupled_grad' attribute instead of the standard '.grad'.
            Defaults to False.

    Returns:
        float: The total number of zeros in the gradients across the process group.
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = torch.zeros(1, dtype=torch.int64, device='cuda')
    data_parallel_group = None
    use_megatron_fsdp = False
    for param in parameters:
        grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
        grad_not_none = hasattr(param, grad_attr) and getattr(param, grad_attr) is not None
        if getattr(param, "__fsdp_param__", False) and grad_not_none:
            # If the parameter is managed by Megatron FSDP, the gradient is
            # an FSDP-sharded DTensor, and we should use the local shard.
            use_megatron_fsdp = True
            grad = getattr(param, grad_attr)._local_tensor
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros += num_zeros
            continue
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param, tp_group=tp_group)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grad_obj = getattr(param, grad_attr)
            data_parallel_group = get_data_parallel_group_if_dtensor(grad_obj, data_parallel_group)
            grad = to_local_if_dtensor(grad_obj).detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    if use_megatron_fsdp and data_parallel_group is not None:
        # Megatron-FSDP does not need to all-reduce the gradient across DP,
        # as FSDP has already handled the DP reduction during BWD.
        raise ValueError(
            "Unexpected use of Megatron FSDP with data parallel group. "
            "Please ensure that the parameters are properly managed by Megatron FSDP."
        )

    # Sum across all data-parallel GPUs if using FSDP.
    if data_parallel_group:
        torch.distributed.all_reduce(
            total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(
        total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
    )

    total_num_zeros = total_num_zeros.item()

    return total_num_zeros
