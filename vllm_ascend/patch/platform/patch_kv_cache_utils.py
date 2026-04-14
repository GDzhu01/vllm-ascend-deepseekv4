from collections import defaultdict

import vllm
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import (create_kv_cache_group_specs,
                                         get_num_blocks, get_uniform_page_size,
                                         may_override_num_blocks,
                                         unify_hybrid_kv_cache_specs,
                                         max_memory_usage_bytes,
                                         _get_kv_cache_groups_uniform_type,
                                         _report_kv_cache_config,
                                         get_kv_cache_config_from_groups,
                                         _check_enough_kv_cache_memory,
                                         )
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.utils.mem_constants import GiB_bytes
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from dataclasses import replace
from vllm_ascend.patch.platform.patch_kv_cache_interface import get_all_kvcache_specs_from_list
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import USE_MULTI_GROUPS_KV_CACHE


logger = init_logger(__name__)

def estimate_max_model_len_with_multi_groups(
    vllm_config: VllmConfig,
    kv_cache_spec_list: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.
    """
    # Save the original max_model_len to restore after estimation
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        all_kv_cache_specs = get_all_kvcache_specs_from_list(kv_cache_spec_list)
        memory_needed = max_memory_usage_bytes(vllm_config, all_kv_cache_specs)
        return memory_needed <= available_memory

    try:
        # Binary search for the maximum model length
        left, right = 1, original_max_model_len

        # If even the smallest model length doesn't fit, return 0
        if not fits_in_memory(left):
            return 0

        # Binary search for the maximum model length that fits
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # Always restore the original max_model_len to avoid side effects
        vllm_config.model_config.max_model_len = original_max_model_len

def check_enough_kv_cache_memory_with_multi_groups(
    vllm_config: VllmConfig,
    kv_cache_spec_list: dict[str, list[KVCacheSpec]],
    available_memory: int,
):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    # No need to check for available memory if the kv_cache_spec is empty
    if kv_cache_spec_list:
        _check_enough_kv_cache_memory(
            available_memory,
            lambda: max_memory_usage_bytes(vllm_config, get_all_kvcache_specs_from_list(kv_cache_spec_list)),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len_with_multi_groups(vllm_config, kv_cache_spec_list, am),
        )

def create_kv_cache_group_specs_with_multi_groups(
    kv_cache_spec_list: dict[str, list[KVCacheSpec]], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """
    Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same
    KVCacheSpec.

    Args:
        kv_cache_spec:
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layer_names:
            A list of kv cache groups, where each element is a list of layer
            names that belong to the same group and should share the same
            KVCacheSpec.
    Returns:
        A list of KVCacheGroupSpec objects, one for each group.
    """
    kv_cache_groups = []

    # TODO(cmq): REFACTOR ME
    for layer_names_one_group in grouped_layer_names:
        skip = False
        for kv_group in kv_cache_groups:
            if layer_names_one_group == kv_group.layer_names:
                skip = True
        if skip:
            continue
        # TODO(cmq): REFACTOR ME: `layer_specs_list` should be initialized with the length of groups
        layer_specs_list:list[list[KVCacheSpec]] = [[],[]]
        for layer_name in layer_names_one_group:
            layer_spec_list = kv_cache_spec_list[layer_name]
            for idx, layer_spec in enumerate(layer_spec_list):
                layer_specs_list[idx].append(layer_spec)
        for layer_specs in layer_specs_list:
            if len(layer_specs) == 0:
                continue
            merged_layer_spec = layer_specs[0].merge(layer_specs)
            kv_cache_groups.append(
                KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
            )

    return kv_cache_groups

def is_kv_cache_spec_uniform_with_multi_groups(kv_cache_spec_list: dict[str, list[KVCacheSpec]]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    if not kv_cache_spec_list:
        # Encoder-only models do not have KV cache, kv_cache_type can be
        # regarded as uniform.
        return True
    try:
        kv_cache_spec_values = get_all_kvcache_specs_from_list(kv_cache_spec_list)
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True

def _get_kv_cache_groups_uniform_spec_with_multi_groups(
    kv_cache_specs_list: dict[str, list[KVCacheSpec]],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with the same KV cache
    spec for all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs
    """
    # Only one spec in a layer, thus grouped_layer_names has no need to take spec list in one
    # layer into account
    return create_kv_cache_group_specs(kv_cache_specs_list, [list(kv_cache_specs_list.keys())])

def unify_kv_cache_spec_page_size_with_multi_groups(
    kv_cache_spec_list: dict[str, list[KVCacheSpec]],
) -> dict[str, list[KVCacheSpec]]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Raise
    NotImplementedError if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """

    page_sizes = set()
    for layer_name, layer_spec_list in kv_cache_spec_list.items():
        for layer_spec in layer_spec_list:
            page_sizes.add(layer_spec.page_size_bytes)
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec_list

    max_page_size = max(page_sizes)
    new_kv_cache_spec_list = {}
    for layer_name, layer_spec_list in kv_cache_spec_list.items():
        for layer_spec in layer_spec_list:
            if layer_spec.page_size_bytes == max_page_size:
                if layer_name in new_kv_cache_spec_list:
                    new_kv_cache_spec_list[layer_name].append(layer_spec)
                else:
                    new_kv_cache_spec_list[layer_name] = [layer_spec]
            else:
                layer_page_size = layer_spec.page_size_bytes
                print(f"{layer_name=}")
                print(f"{max_page_size=}")
                print(f"{layer_spec=}")
                print(f"{layer_page_size=}")
                if max_page_size % layer_page_size != 0:
                    raise NotImplementedError(
                        "The page size of the layer is not divisible by the "
                        "maximum page size. Cannot unify by adjusting block_size."
                    )
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
                assert new_spec.page_size_bytes == max_page_size

                if layer_name in new_kv_cache_spec_list:
                    new_kv_cache_spec_list[layer_name].append(new_spec)
                else:
                    new_kv_cache_spec_list[layer_name] = [new_spec]
    return new_kv_cache_spec_list

def is_kv_cache_type_attention_free_with_multi_groups(kv_cache_spec_list: dict[str, list[KVCacheSpec]]) -> bool:
    # kv_cache_spec is an empty dict for attention free models
    return not kv_cache_spec_list


def _get_kv_cache_groups_uniform_page_size_with_multi_groups(
    kv_cache_spec_list: dict[str, list[KVCacheSpec]],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer
    in the group.
    For example:
    1. A model only uses full attention. The pattern is
    (num_hidden_layers * full), so there is only one group and the block table
    is shared by all layers. It is already handled by
    `_get_kv_cache_config_uniform_type`.
    2. A model with 10 full attention layers and 20 sliding window
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so
    there are 3 kv_cache_groups, each of which represents 10 layers.

    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups.
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV
    cache group, but within each KV cache group, all layers must share the same
    block size.
    3. Physical memory per token per layer: This property is decided by model
    config. Currently we only support models that have the same physical memory
    per token per layer for all layers. Can be relaxed with a simple extension,
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers.
    Can be relaxed with a simple extension, but still need to keep physical
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    attention type. One exception is that, when
    `--disable-hybrid-kv-cache-manager` is true, the single group for full
    attention layers may also include attention layers using sliding window or
    LLaMA 4 local attention. See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is
    general to an arbitrary number of attention types. But
    `find_longest_cache_hit` only supports one attention type or two
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and
    number of layers per group are the same now, we can ensure that physical
    memory per block is the same for all groups.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs
    """
    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec_list in kv_cache_spec_list.items():
        for layer_spec in layer_spec_list:
            same_type_layers[layer_spec].append(layer_name)

    # layer0 --> spec0, spec1
    # layer1 --> spec2
    # spec0, spec1, spec2

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    min_num_layers = min([len(layers) for layers in same_type_layers.values()])
    # TODO(cmq): REFACTOR ME to more general logic

    max_num_layers = max([len(layers) for layers in same_type_layers.values()])
    if max_num_layers < min_num_layers * 1.25:
        # If the number of layers is not much larger than the minimum number of layers,
        # use the maximum number of layers as the group size to avoid too many padding
        # layers. A typical example is gpt-oss-20b + eagle, with 12 sw + 13 full. We
        # pad it to (13 sw, 13 full) instead of (12 sw, 24 full). 1.25 is just a
        # magic number to avoid too many padding layers.
        group_size = max_num_layers
    group_size = 22
    grouped_layers = []
    group_layer_specs = []
    for layer_spec, layers in same_type_layers.items():
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
            group_layer_specs.append(layer_spec)
    kv_cache_groups = []
    for group_layer_spec, layer_names_one_group in zip(group_layer_specs, grouped_layers):
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, group_layer_spec)
        )
    return kv_cache_groups
    # TODO (wjq) refactor me later
    # return create_kv_cache_group_specs(kv_cache_spec_list, grouped_layers)

def unify_hybrid_kv_cache_specs_with_multi_groups(kv_cache_spec_list: dict[str, list[KVCacheSpec]]):
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    if is_kv_cache_spec_uniform_with_multi_groups(
        kv_cache_spec_list
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec_list):
        return


    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )

    # when hybrid is disabled, multi-specs in one layer is also disabled.
    assert is_one_spec_type_in_list(kv_cache_spec_list), \
        "Only one spec type in one layer is required when hybrid kvcache is disabled"

    kv_cache_spec: dict[str, KVCacheSpec] = {}
    for layer_name, layer_specs in kv_cache_spec_list.items():
        kv_cache_spec[layer_name] = layer_specs[0]

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in kv_cache_spec.values()
    )
    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    sliding_window=spec.sliding_window,
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                )

    if not (
        is_kv_cache_spec_uniform_with_multi_groups(kv_cache_spec)
        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)
    ):
        raise ValueError(
            "Hybrid KV cache manager is disabled but failed to "
            "convert the KV cache specs to one unified type."
        )

def is_one_spec_type_in_list(kv_cache_specs_list: dict[str, list[KVCacheSpec]]):
    for _, layer_specs in kv_cache_specs_list.items():
        if len(layer_specs) == 1:
            # Different specs in one layer, not uniform
            return False
    return True


def get_kv_cache_groups_with_multi_groups(
    vllm_config: VllmConfig, kv_cache_spec_list: dict[str, list[KVCacheSpec]]
) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs_with_multi_groups(kv_cache_spec_list)

    if is_kv_cache_type_attention_free_with_multi_groups(kv_cache_spec_list):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    if is_kv_cache_spec_uniform_with_multi_groups(kv_cache_spec_list):
        assert is_one_spec_type_in_list(kv_cache_spec_list), "Only one spec type is required in uniform spec"
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec_with_multi_groups(kv_cache_spec_list)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec_list):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)

    # As KVCacheManager can only allocate memory of one size, we need to unify
    # the page size of the layers. For cases cannot be unified, this function
    # will raise an error.
    kv_cache_spec_list = unify_kv_cache_spec_page_size_with_multi_groups(kv_cache_spec_list)
    # Model contains multiple attention types, but KV cache of all layers
    # have the same physical memory per block per layer. Split the layers
    # into groups with the same number of layers, and thus same total page
    # size.
    return _get_kv_cache_groups_uniform_page_size_with_multi_groups(kv_cache_spec_list)

def get_kv_cache_configs_with_multi_groups(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, list[KVCacheSpec]]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    merged_kv_cache_specs_list: dict[str, list[KVCacheSpec]] = {}
    for kv_cache_spec_list_one_worker in kv_cache_specs:
        for layer_name, layer_spec_list in kv_cache_spec_list_one_worker.items():
            if layer_name not in merged_kv_cache_specs_list:
                merged_kv_cache_specs_list[layer_name] = layer_spec_list
            else:
                assert merged_kv_cache_specs_list[layer_name] == layer_spec_list, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs_list may be modified in-place.
    # TODO
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs_list)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    # TODO
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec_list)
        for worker_spec_list in kv_cache_specs
    ]

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    # Check if the available memory is enough per worker.
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_list_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        all_layer_names = []
        for layer_name, layer_spec_list in kv_cache_spec_list_one_worker.items():
            all_layer_names.extend([layer_name] * len(layer_spec_list))

        assert sum(len(group.layer_names) for group in projected_groups) == len(
            all_layer_names
        ), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # Shrink tensor size proportionally
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

        if len(kv_cache_config.kv_cache_groups) > 0:
            _report_kv_cache_config(vllm_config, kv_cache_config)

    return kv_cache_configs

if USE_MULTI_GROUPS_KV_CACHE:
    vllm.v1.core.kv_cache_utils.estimate_max_model_len = estimate_max_model_len_with_multi_groups
    vllm.v1.core.kv_cache_utils.check_enough_kv_cache_memory = check_enough_kv_cache_memory_with_multi_groups
    vllm.v1.core.kv_cache_utils.create_kv_cache_group_specs = create_kv_cache_group_specs_with_multi_groups
    vllm.v1.core.kv_cache_utils.is_kv_cache_spec_uniform = is_kv_cache_spec_uniform_with_multi_groups
    vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_spec = _get_kv_cache_groups_uniform_spec_with_multi_groups
    vllm.v1.core.kv_cache_utils.unify_kv_cache_spec_page_size = unify_kv_cache_spec_page_size_with_multi_groups
    vllm.v1.core.kv_cache_utils.is_kv_cache_type_attention_free = is_kv_cache_type_attention_free_with_multi_groups
    vllm.v1.core.kv_cache_utils.unify_hybrid_kv_cache_specs = unify_hybrid_kv_cache_specs_with_multi_groups
    vllm.v1.core.kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups_with_multi_groups
    vllm.v1.core.kv_cache_utils.get_kv_cache_configs = get_kv_cache_configs_with_multi_groups
    vllm.v1.core.kv_cache_utils.get_all_kvcache_specs_from_list = get_all_kvcache_specs_from_list
else:
    def _get_kv_cache_groups_uniform_block_size(
            kv_cache_spec: dict[str, KVCacheSpec], ) -> list[KVCacheGroupSpec]:
        '''
        Generates the KV cache groups with same block size,
        and there maybe multiple groups with different spec,
        each group has their own block_pool and each layer
        of each group has their own kv_cache_tensor.

        :param kv_cache_spec: The KVCacheSpecs of all the layers
        :type kv_cache_spec: dict[str, KVCacheSpec]
        :return: a list of KVCacheGroupSpecs, there is one type of KVCacheSpec in each group
        :rtype: list[KVCacheGroupSpec]
        '''
        same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
        _, first_kv_cache_config = next(iter(kv_cache_spec.items()))
        block_size = first_kv_cache_config.block_size
        for layer_name, layer_spec in kv_cache_spec.items():
            assert block_size == layer_spec.block_size, "Layer block size is not equal."
            same_type_layers[layer_spec].append(layer_name)
        grouped_layers = list(same_type_layers.values())
        return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


    def check_uniform_page_size(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
        kv_cache_specs = [group.kv_cache_spec for group in kv_cache_groups]
        page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
        return len(page_sizes) == 1


    def get_kv_cache_groups(
            vllm_config: VllmConfig,
            kv_cache_spec: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
        """
        Split the layers in the model into groups with the same KV cache spec.

        Args:
            vllm_config: The global VllmConfig
            kv_cache_spec: The kv cache spec of each attention layer in the model

        Returns:
            The generated KVCacheGroups
        """

        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            unify_hybrid_kv_cache_specs(kv_cache_spec)

        # kv cache group spec with multi groups and same block size without share hybrid blocks
        return _get_kv_cache_groups_uniform_block_size(kv_cache_spec)


    def get_kv_cache_config_from_groups(
            vllm_config: VllmConfig,
            kv_cache_groups: list[KVCacheGroupSpec],
            available_memory: int,
    ) -> KVCacheConfig:
        """
        Generate the KV cache configuration from the KV cache groups and spec
        of each layer.

        Args:
            vllm_config: The global VllmConfig
            kv_cache_groups: The KV cache groups
            available_memory: Memory available for KV cache in bytes
        Returns:
            The generated KVCacheConfig
        """
        if len(kv_cache_groups) == 0:
            # Attention free models do not have KV cache.
            # Return num_blocks=1 as BlockPool always needs a null_block.
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
            )

        # Determine how model runners should initialize the KV cache tensors.
        if len(kv_cache_groups) == 1 and isinstance(
                kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs):
            # Special case: all layers have the same type of KV cache but with
            # different hidden size. Allocate different amount of memory for each
            # layer based on its hidden size.
            num_blocks = (available_memory //
                          kv_cache_groups[0].kv_cache_spec.page_size_bytes)
            num_blocks = may_override_num_blocks(vllm_config, num_blocks)
            per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
            kv_cache_tensors = [
                KVCacheTensor(
                    size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                    shared_by=[layer_name],
                ) for layer_name in kv_cache_groups[0].layer_names
            ]
        elif check_uniform_page_size(kv_cache_groups) is False:
            # Special case: there are multiple groups of KV cache, and the block
            # size of them keeps the same. We will still have `num_layers` memory
            # pools, this means the memory pools won't be shared across the groups.
            total_page_size_bytes = 0
            for kv_cache_group in kv_cache_groups:
                num_layers = len(kv_cache_group.layer_names)
                page_size = kv_cache_group.kv_cache_spec.page_size_bytes
                total_page_size_bytes += page_size * num_layers
            num_blocks = available_memory // total_page_size_bytes
            # TODO(zxr): DONT use magic number
            num_blocks = num_blocks // 128 * 128
            assert num_blocks > 0
            kv_cache_tensors = []
            for i in range(len(kv_cache_groups)):
                for layer_name in kv_cache_groups[i].layer_names:
                    shared_by = [layer_name]
                    kv_cache_tensors.append(
                        KVCacheTensor(
                            size=kv_cache_groups[i].kv_cache_spec.page_size_bytes *
                                 num_blocks,
                            shared_by=shared_by))
        else:
            # General case:
            # We will have group_size memory pools, each is shared by one layer from
            # each group. As layers of different groups have different block table,
            # they will use different parts of the shared Tensor.
            # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
            # (sw.1, padding) will be: (group_size = 2)
            # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
            # full.1, sw.2: share another Tensor with size=available_memory//2
            group_size = max(len(group.layer_names) for group in kv_cache_groups)

            page_size = get_uniform_page_size(
                [group.kv_cache_spec for group in kv_cache_groups])
            assert group_size > 0, "group_size must be greater than 0"
            num_blocks = get_num_blocks(vllm_config, group_size, available_memory,
                                        page_size)
            kv_cache_tensors = []
            for i in range(group_size):
                shared_by = []
                for j in range(len(kv_cache_groups)):
                    if i < len(kv_cache_groups[j].layer_names):
                        shared_by.append(kv_cache_groups[j].layer_names[i])
                kv_cache_tensors.append(
                    KVCacheTensor(size=page_size * num_blocks,
                                  shared_by=shared_by))

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )
    vllm.v1.core.kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups
    vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
