# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
import vllm.v1.core.kv_cache_utils as _kv_cache_utils
from vllm.v1.core.kv_cache_utils import (
    KVCacheGroupSpec,
    KVCacheTensor,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_num_blocks,
    is_kv_cache_spec_uniform,
    is_kv_cache_type_attention_free,
    unify_hybrid_kv_cache_specs,
)


_ORIG_GET_KV_CACHE_CONFIG_FROM_GROUPS = (
    _kv_cache_utils.get_kv_cache_config_from_groups
)
_ORIG_GET_KV_CACHE_GROUPS = _kv_cache_utils.get_kv_cache_groups


def approximate_gcd(values: list[int], *, lower_bound: int | None = None) -> int:
    """Pick a group size that minimizes total upward padding."""
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {values!r}")

    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d
    return best_d


def _has_svf_mla_and_swa_specs(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    has_mla = False
    has_swa = False
    compress_ratios: set[int] = set()
    for spec in kv_cache_spec.values():
        if isinstance(spec, SlidingWindowMLASpec):
            has_swa = True
        elif isinstance(spec, MLAAttentionSpec):
            has_mla = True
            compress_ratios.add(spec.compress_ratio)
    return has_mla and has_swa and len(compress_ratios) >= 2


def _has_svf_uniform_groups(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    if not all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ):
        return False

    has_swa = False
    compress_ratios: set[int] = set()
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for spec in group.kv_cache_spec.kv_cache_specs.values():
            if isinstance(spec, SlidingWindowMLASpec):
                has_swa = True
            elif isinstance(spec, MLAAttentionSpec):
                compress_ratios.add(spec.compress_ratio)
    return has_swa and len(compress_ratios) >= 2


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> _kv_cache_utils.KVCacheConfig:
    if not _has_svf_uniform_groups(kv_cache_groups):
        return _ORIG_GET_KV_CACHE_CONFIG_FROM_GROUPS(
            vllm_config, kv_cache_groups, available_memory
        )

    # Special case (only SVF for now): all groups are UniformTypeKVCacheSpecs.
    # They must already be page_size aligned and #(layer_tuples) aligned.
    # Here we allocate one KV cache tensor per (layer_tuple, page_size) "slot".
    # Layers across groups with the same page size at the same tuple index share
    # the same backing tensor.
    def get_real_layer_name(spec_layer_name: str) -> str:
        return ".".join(spec_layer_name.split(".")[:3])

    layer_specs: dict[str, KVCacheSpec] = {}
    page_kv_cache_groups: dict[int, list[KVCacheGroupSpec]] = defaultdict(list)
    for group in kv_cache_groups:
        group_page_size_spec_layers: dict[int, list[str]] = defaultdict(list)
        for spec_layer_name in group.layer_names:
            assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            layer_single_spec = group.kv_cache_spec.kv_cache_specs[spec_layer_name]
            group_page_size_spec_layers[layer_single_spec.page_size_bytes].append(
                spec_layer_name
            )
            layer_specs[spec_layer_name] = layer_single_spec
        for page_size, spec_layer_names in group_page_size_spec_layers.items():
            # NOTE(zxr): we assume that layers with same page size in one group
            # use same kv_cache_spec
            page_group_spec = layer_specs[spec_layer_names[0]]
            page_kv_cache_group = KVCacheGroupSpec(spec_layer_names, page_group_spec)
            page_kv_cache_groups[page_size].append(page_kv_cache_group)
    kv_cache_tensors: list[KVCacheTensor] = []
    max_group_size = 0
    total_page_size = 0
    for page_size, kv_cache_group_lists in page_kv_cache_groups.items():
        page_group_size = max(len(g.layer_names) for g in kv_cache_group_lists)
        max_group_size = max(max_group_size, page_group_size)
        total_page_size += page_size
    num_blocks = get_num_blocks(
        vllm_config, max_group_size, available_memory, total_page_size
    )
    for page_size, kv_cache_group_lists in page_kv_cache_groups.items():
        page_group_size = max(len(g.layer_names) for g in kv_cache_group_lists)
        allocate_complete_layers: list[str] = []
        used_layer_kv_cache_group_idx: dict[str, set[int]] = defaultdict(set)
        layer_kv_cache_group_idx: dict[str, set[int]] = defaultdict(set)
        layer_to_spec_layer_names: dict[str, list[str]] = defaultdict(list)
        for i, group in enumerate(kv_cache_group_lists):
            for layer_name in group.layer_names:
                real_layer_name = get_real_layer_name(layer_name)
                layer_kv_cache_group_idx[real_layer_name].add(i)
                layer_to_spec_layer_names[real_layer_name].append(layer_name)
        for _ in range(page_group_size):
            shared_by: list[str] = []
            used_group_idx_set: list[int] = []
            for j in range(len(kv_cache_group_lists)):
                for layer_name in kv_cache_group_lists[j].layer_names:
                    real_layer_name = get_real_layer_name(layer_name)
                    if real_layer_name in allocate_complete_layers:
                        continue
                    group_used = False
                    for gid in layer_kv_cache_group_idx[real_layer_name]:
                        if gid in used_group_idx_set:
                            group_used = True
                            break
                        used_layer_kv_cache_group_idx[real_layer_name].add(gid)
                    if group_used:
                        continue
                    shared_by.extend(layer_to_spec_layer_names[real_layer_name])
                    used_group_idx_set.extend(layer_kv_cache_group_idx[real_layer_name])
                    if len(used_layer_kv_cache_group_idx[real_layer_name]) == len(
                        layer_kv_cache_group_idx[real_layer_name]
                    ):
                        allocate_complete_layers.append(real_layer_name)
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    return _kv_cache_utils.KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for SVF.
    """
    if not _has_svf_mla_and_swa_specs(kv_cache_spec):
        return None

    ratio_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            ratio_specs[spec.compress_ratio][name] = spec

    mla_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in ratio_specs.values():
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(
        isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs
    )
    full_mla_specs: list[UniformTypeKVCacheSpecs] = []
    swa_mla_specs: list[UniformTypeKVCacheSpecs] = []
    for group in grouped_specs:
        first_spec = next(iter(group.kv_cache_specs.values()))
        if isinstance(first_spec, MLAAttentionSpec):
            assert all(
                isinstance(spec, MLAAttentionSpec)
                for spec in group.kv_cache_specs.values()
            )
            full_mla_specs.append(group)
        else:
            assert isinstance(first_spec, SlidingWindowMLASpec)
            assert all(
                isinstance(spec, SlidingWindowMLASpec)
                for spec in group.kv_cache_specs.values()
            )
            swa_mla_specs.append(group)

    assert full_mla_specs and swa_mla_specs
    full_mla_groups = [
        KVCacheGroupSpec(
            layer_names=list(full_mla_spec.kv_cache_specs.keys()),
            kv_cache_spec=full_mla_spec,
        )
        for full_mla_spec in full_mla_specs
    ]

    num_layer_tuples_per_group: list[int] = [
        g_spec.get_num_layer_tuples() for g_spec in grouped_specs
    ]
    num_layer_tuples = approximate_gcd(
        num_layer_tuples_per_group,
        lower_bound=full_mla_specs[0].get_num_layer_tuples(),
    )

    all_page_sizes = sorted(
        {
            page_size
            for full_mla_spec in full_mla_specs
            for page_size in full_mla_spec.get_page_sizes()
        }
    )
    swa_mla_groups: list[KVCacheGroupSpec] = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        for sm_page_size in sm_page_sizes:
            candidate = min(x for x in all_page_sizes if x >= sm_page_size)
            if sm_page_size < candidate:
                for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
                    if layer_spec.page_size_bytes != sm_page_size:
                        continue
                    object.__setattr__(layer_spec, "page_size_padded", candidate)
            for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
                if layer_spec.page_size_bytes == candidate:
                    layers_per_size[candidate].append(layer_name)
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            group_layer_names = [
                name for layer_tuple in group_layer_tuples for name in layer_tuple
            ]
            group_layer_specs = {
                name: sm_spec.kv_cache_specs[name] for name in group_layer_names
            }
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [*full_mla_groups, *swa_mla_groups]


def get_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.
    """
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        return _kv_cache_utils._get_kv_cache_groups_uniform_spec(kv_cache_spec)
    grouped_specs = group_and_unify_kv_cache_specs(kv_cache_spec)
    if grouped_specs:
        return _get_kv_cache_groups_uniform_groups(grouped_specs)
    return _ORIG_GET_KV_CACHE_GROUPS(vllm_config, kv_cache_spec)


_kv_cache_utils.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
_kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups
