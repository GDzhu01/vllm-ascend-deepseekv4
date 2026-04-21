import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Tuple, Type, TypeVar

import torch
import torch.nn.functional as F
import torch_npu
import vllm.envs as envs_vllm
from vllm.v1.attention.backend import AttentionBackend
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.attention.backend import (AttentionCGSupport,
                                        AttentionMetadataBuilder)
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         split_decodes_and_prefills)
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (AscendDeviceType, attention_calculation_stream,
                               get_ascend_device_type, npu_stream_switch,
                               olora_tp_enable)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_ascend.ops.triton.rms_norm import triton_q_rms

if HAS_TRITON:
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms  # noqa: F811
else:
    triton_q_rms = None  # type: ignore

BUILD_METADATA_STEP_PREFILL = 0
BUILD_METADATA_STEP_DECODE = 1


def hadamard_transform_ref(x: torch.Tensor, hadamard: torch.Tensor, scale:int =1.0,):
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x,
                                  hadamard=hadamard,
                                  scale=hidden_size**-0.5)


def pad_to_blocks(x: torch.Tensor,
                  length_list: torch.Tensor,
                  block_size: int = 128):
    """
    Pads a ragged/packed tensor into fixed-size blocks.

    Args:
        x: Input tensor of shape [t, n, d] where t = sum(length_list).
        length_list: Tensor of shape [bs] containing valid sequence lengths.
        block_size: The size of each block (default 128).

    Returns:
        padded_blocks: Tensor of shape [total_blocks, block_size, n, d].
    """
    # 1. Validation
    if x.shape[0] != length_list.sum():
        raise ValueError(
            f"Input dimension 0 ({x.shape[0]}) does not match sum of length_list ({length_list.sum()})"
        )

    bs = length_list.shape[0]
    n, d = x.shape[1], x.shape[2]

    # 2. Calculate how many blocks are needed for each request
    # Formula: ceil(length / block_size) -> (length + block_size - 1) // block_size
    blocks_per_req = (length_list + block_size - 1) // block_size
    total_blocks = blocks_per_req.sum() + 1

    # 3. Allocate output tensor with zeros (this handles the padding automatically)
    # Shape: [total_blocks, block_size, n, d]
    out = torch.zeros((total_blocks, block_size, n, d),
                      dtype=x.dtype,
                      device=x.device)

    # 4. Fill data
    input_offset = 0
    block_offset = 1

    for i in range(bs):
        length = length_list[i]
        num_blocks = blocks_per_req[i]

        if length > 0:
            # Slice the valid data for this request from the packed input
            # Shape: [length, n, d]
            req_data = x[input_offset:input_offset + length]

            # Select the assigned blocks in the output
            # Shape: [num_blocks, block_size, n, d]
            target_blocks = out[block_offset:block_offset + num_blocks]

            # View as a flat sequence to easily copy the data
            # Shape: [num_blocks * block_size, n, d]
            target_flat = target_blocks.view(-1, n, d)

            # Copy valid data into the beginning of the allocated blocks
            # The rest remains zeros
            target_flat[:length] = req_data

        # Update pointers
        input_offset += length
        block_offset += num_blocks

    return out


class AscendDSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "ASCEND_DSA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_builder_cls():
        return AscendDSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_scale_shape(num_blocks: int, block_size: int,
                        scale_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, scale_size

    @staticmethod
    def get_impl_cls() -> Type["DSAAttentionImpl"]:
        return AscendDSAImpl

    @staticmethod
    def get_supported_block_size() -> list[int]:
        return [32, 64, 128, 1024]


@dataclass
class AscendDSAPrefillMetadata:
    """ Prefill Specific Metadata for Ascend"""
    attn_mask: torch.Tensor
    query_lens: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    input_positions: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_query_len: int
    max_seq_lens: int

    sin: torch.Tensor = None
    cos: torch.Tensor = None
    compress_sin: torch.Tensor = None
    compress_cos: torch.Tensor = None
    start_pos: Optional[torch.Tensor] = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None
    cu_c4_cmp_seqlen_list: torch.Tensor = None
    cu_c128_cmp_seqlen_list: torch.Tensor = None


@dataclass
class AscendDSADecodeMetadata:
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen_kv: int
    max_seqlen_q: int
    seq_lens_list: list[int]
    max_seq_lens: int
    slot_mapping: torch.Tensor

    query_start_loc: torch.tensor = None
    query_start_loc_cpu: torch.tensor = None
    attn_mask: Optional[torch.Tensor] = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    compress_sin: torch.Tensor = None
    compress_cos: torch.Tensor = None
    cp_seq_len: torch.Tensor = None
    batch_seq_mask: torch.Tensor = None
    start_pos: torch.Tensor = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None


@dataclass
class AscendDSAMetadata:
    """Metadata for MLACommon.
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    query_lens: Optional[list[int]] = None
    # The dimension of the attention heads
    head_dim: Optional[int] = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    decode: Optional[AscendDSADecodeMetadata] = None
    prefill: Optional[AscendDSAPrefillMetadata] = None
    reshape_cache_event: torch.npu.Event = None

    # metadata for dsv4 indexer

    hadamard: Optional[torch.Tensor] = None

    start_pos: Optional[torch.Tensor] = None

    def __post_init__(self):
        pass


M = TypeVar("M", bound=AscendDSAMetadata)


class AscendDSAMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH
    hadamard = None
    start_pos_prefill: Optional[torch.Tensor] = None
    start_pos_decode: Optional[torch.Tensor] = None
    decode_sas_metadata: Optional[torch.Tensor] = None
    decode_qli_metadata: Optional[torch.Tensor] = None
    prefill_ratio_to_sas_metadata: Optional[dict] = None
    decode_ratio_to_sas_metadata: Optional[dict] = None
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec: MLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendDSAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.metadata_cls = (metadata_cls if metadata_cls is not None else
                             AscendDSAMetadata)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        scheduler_config = vllm_config.scheduler_config
        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len +
                           self.block_size - 1) // self.block_size

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

        self.reorder_batch_threshold = self.decode_threshold
        self.rope_dim = self.model_config.hf_text_config.rope_head_dim
        self.cos_cache = None
        self.sin_cache = None

        self.cu_seq_lens_cpu: torch.Tensor = None
        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.context_lens_cpu: torch.Tensor = None
        self.num_actual_tokens: Optional[int] = None
        self.block_table: torch.Tensor = None
        self.slot_mapping: torch.Tensor = None
        self.graph_pad_size = 0
        self.query_lens: torch.Tensor = None
        self.seq_lens: torch.Tensor = None
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

        layer_idx = extract_layer_index(layer_names[0])
        if layer_idx in [0, 1]:
            self.compressor_ratio = 1
        elif layer_idx % 2 == 0:
            self.compressor_ratio = 4
        else:
            self.compressor_ratio = 128

        if AscendDSAMetadataBuilder.hadamard is None:
            hf_config = self.model_config.hf_config
            if hf_config.model_type == 'deepseek_v4':
                indexer_head_dim = hf_config.index_head_dim
                try:
                    from scipy.linalg import hadamard
                except ImportError as e:
                    raise ImportError("Please install scipy") from e
                log_dim = math.ceil(math.log2(indexer_head_dim))
                dim_padded = 2**log_dim
                AscendDSAMetadataBuilder.hadamard = torch.tensor(
                    hadamard(dim_padded, dtype=float),
                    dtype=torch.float,
                    device=self.device).to(torch.bfloat16)
        self.start_pos_prefill = torch.zeros(scheduler_config.max_num_seqs,
                                             dtype=torch.int32,
                                             device=self.device)
        self.start_pos_decode = torch.zeros(scheduler_config.max_num_seqs,
                                            dtype=torch.int32,
                                            device=self.device)
        self.decode_sas_metadata = torch.zeros(1024,
                                                  dtype=torch.int32,
                                                  device=self.device)
        self.decode_qli_metadata = torch.zeros(1024,
                                               dtype=torch.int32,
                                               device=self.device)
        self.cu_seqlens_ori_kv = torch.tensor([], device=self.device)
        self.cu_seqlens_cmp_kv = torch.tensor([], device=self.device)
        self.seqused_q = torch.tensor([], device=self.device)
        ascend_config = get_ascend_config()
        self.enable_kv_tnd = ascend_config.enable_kv_tnd

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendDSAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if num_tokens <= self.decode_threshold:
                decodes.append(i)
            else:
                prefills.append(i)

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill],
                                        decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        return modified_batch

    def set_num_actual_tokens(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        num_reqs_actual = kwargs.get("num_reqs_actual", None)
        self.prefill_ratio_to_sas_metadata = kwargs.get("prefill_ratio_to_sas_metadata", None)
        self.decode_ratio_to_sas_metadata = kwargs.get("decode_ratio_to_sas_metadata", None)
        self.common_ratio_to_sas_metadata = kwargs.get("common_ratio_to_sas_metadata", None)

        if self.common_ratio_to_sas_metadata.get("num_decodes", None) is None:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = \
                split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
            self.common_ratio_to_sas_metadata["num_decodes"] = self.num_decodes
            self.common_ratio_to_sas_metadata["num_prefills"] = self.num_prefills
            self.common_ratio_to_sas_metadata["num_decode_tokens"] = self.num_decode_tokens
            self.common_ratio_to_sas_metadata["num_prefill_tokens"] = self.num_prefill_tokens
            self.set_num_actual_tokens(common_attn_metadata)
            assert self.num_decodes + self.num_prefills == num_reqs
            assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens
            num_input_tokens = common_attn_metadata.num_input_tokens
            input_positions = common_attn_metadata.positions[:
                                                         num_input_tokens].long(
                                                         )
            self.common_ratio_to_sas_metadata["input_positions"] = input_positions
            if self.num_prefills:
                cos, sin = get_cos_and_sin_dsa(input_positions)
            else:
                cos, sin = get_cos_and_sin_dsa(input_positions, True)
            self.common_ratio_to_sas_metadata["cos"] = cos
            self.common_ratio_to_sas_metadata["sin"] = sin
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            self.common_ratio_to_sas_metadata["seq_lens"] = self.seq_lens
        else:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = \
                self.common_ratio_to_sas_metadata["num_decodes"], \
                self.common_ratio_to_sas_metadata["num_prefills"], \
                self.common_ratio_to_sas_metadata["num_decode_tokens"], \
                self.common_ratio_to_sas_metadata["num_prefill_tokens"]
            self.set_num_actual_tokens(common_attn_metadata)
            num_input_tokens = common_attn_metadata.num_input_tokens
            input_positions = self.common_ratio_to_sas_metadata["input_positions"]
            cos, sin = self.common_ratio_to_sas_metadata["cos"], self.common_ratio_to_sas_metadata["sin"]
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        self.query_lens = query_seq_lens_cpu[:num_reqs]

        self.graph_pad_size = common_attn_metadata.graph_pad_size
        block_table_size = self.get_block_table_size(
            common_attn_metadata, BUILD_METADATA_STEP_PREFILL)
        self.block_table = common_attn_metadata.block_table_tensor[:
                                                                   block_table_size]
        self.slot_mapping = common_attn_metadata.slot_mapping[:
                                                              num_input_tokens]

        prefill_metadata = None
        if self.num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata(
                common_prefix_len, common_attn_metadata)

        decode_metadata = None

        if self.num_decodes > 0:
            decode_metadata = self.build_decode_metadata(
                common_prefix_len, common_attn_metadata, num_reqs_actual)

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            query_lens=self.query_lens,
            slot_mapping=None,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_mask=self.attn_mask_builder.get_final_mla_mask(
                self.model_config),
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            block_tables=None,
            seq_lens=self.seq_lens,
            cos=cos,
            sin=sin,
            hadamard=AscendDSAMetadataBuilder.hadamard,
        )

    def build_prefill_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendDSAPrefillMetadata:
        query_start_loc = common_attn_metadata.query_start_loc

        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        input_positions = common_attn_metadata.positions[:self.
                                                         num_actual_tokens].long(
                                                         )

        # reqs_start: the start request position of prefill request
        reqs_start = self.num_decodes
        # reqs_start: the start token position of prefill request
        tokens_start = self.num_decode_tokens

        max_query_len = self.query_lens[reqs_start:].max().item()
        max_seq_lens = common_attn_metadata.seq_lens_cpu[reqs_start:].max(
        ).item()
        prefill_query_start_loc = query_start_loc[
            reqs_start:] - query_start_loc[reqs_start]

        prefill_input_positions = input_positions[tokens_start:]
        cos, sin = get_cos_and_sin_dsa(prefill_input_positions)

        def _get_padded_compressed_position(prefill_input_positions,
                                            compress_ratio):
            if compress_ratio == 1:
                return prefill_input_positions
            mask = ((prefill_input_positions + 1) % compress_ratio) == 0
            input_positions = prefill_input_positions[mask]
            input_positions = (input_positions + 1) - compress_ratio
            target_shape = (min(
                self.num_prefill_tokens,
                self.num_prefill_tokens // compress_ratio +
                self.num_prefills), )
            pad_right = target_shape[0] - input_positions.shape[0]
            pad_positions = F.pad(input_positions, (0, pad_right), value=0.0)
            return pad_positions

        def _get_cmp_seq_lens(prefill_seq_lens, compress_ratio):
            _cmp_seq_lens = prefill_seq_lens // compress_ratio
            return torch.concat(
                (torch.tensor([0], device=_cmp_seq_lens.device),
                 torch.cumsum(_cmp_seq_lens, -1)),
                dim=-1)

        compress_cos, compress_sin = get_cos_and_sin_dsa(
            _get_padded_compressed_position(prefill_input_positions,
                                            self.compressor_ratio))

        prefill_seq_lens = self.seq_lens[reqs_start:]
        num_prefill = prefill_seq_lens.shape[0]

        decode_input_positions = input_positions[:tokens_start]

        def _get_compressed_decode_token_start_and_end(decode_input_positions,
                                                       compress_ratio):
            # TODO(cmq): decode_input_positions is a device tensor,
            # this will introduce sync operation. Refactor me to torch.where instead
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            compressed_decode_num = mask.sum()

            end = min(
                self.num_prefill_tokens,
                self.num_prefill_tokens // compress_ratio + self.num_prefills)
            return compressed_decode_num, end

        compressed_tokens_start, compressed_tokens_end = _get_compressed_decode_token_start_and_end(
            decode_input_positions, self.compressor_ratio)

        prefill_slot_mapping = self.slot_mapping[
            compressed_tokens_start:compressed_tokens_end +
            compressed_tokens_start]

        assert self.start_pos_prefill is not None
        self.start_pos_prefill.fill_(0)
        seq_lens_q = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
        self.start_pos_prefill[:num_prefill] = self.seq_lens[
            reqs_start:] - seq_lens_q

        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        index_topk = self.model_config.hf_config.index_topk

        if self.enable_kv_tnd:
            cu_c4_cmp_seqlen_list = _get_cmp_seq_lens(prefill_seq_lens, 4)
            cu_c128_cmp_seqlen_list = _get_cmp_seq_lens(prefill_seq_lens, 128)
        else:
            cu_c4_cmp_seqlen_list = None
            cu_c128_cmp_seqlen_list = None

        if self.compressor_ratio == 1:
            if self.prefill_ratio_to_sas_metadata.get("c1") is None:
                self.prefill_ratio_to_sas_metadata["c1"] = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=prefill_query_start_loc,
                    cu_seqlens_ori_kv=prefill_query_start_loc,
                    cu_seqlens_cmp_kv=None,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[reqs_start:],
                    max_seqlen_q=seq_lens_q.max(),
                    max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                    batch_size=len(self.seq_lens[reqs_start:]),
                    cmp_ratio=1,
                    ori_mask_mode=4,  # 4:sliding window
                    ori_win_left=self.model_config.hf_config.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="TND" if self.enable_kv_tnd else "PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=False,
                    device=str(self.seqused_q.device))
            sas_metadata = self.prefill_ratio_to_sas_metadata["c1"]
        elif self.compressor_ratio == 4:
            if self.prefill_ratio_to_sas_metadata.get("c4") is None:
                self.prefill_ratio_to_sas_metadata["c4"] = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=prefill_query_start_loc,
                    cu_seqlens_ori_kv=prefill_query_start_loc,
                    cu_seqlens_cmp_kv=cu_c4_cmp_seqlen_list,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[reqs_start:],
                    max_seqlen_q=seq_lens_q.max(),
                    max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                    batch_size=len(self.seq_lens[reqs_start:]),
                    cmp_topk=index_topk,
                    # topk=index_topk,
                    cmp_ratio=4,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="TND" if self.enable_kv_tnd else "PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                    device=str(self.seqused_q.device))
            sas_metadata = self.prefill_ratio_to_sas_metadata["c4"]
        else:
            if self.prefill_ratio_to_sas_metadata.get("c128") is None:
                self.prefill_ratio_to_sas_metadata["c128"] = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=prefill_query_start_loc,
                    cu_seqlens_ori_kv=prefill_query_start_loc,
                    cu_seqlens_cmp_kv=cu_c128_cmp_seqlen_list,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[reqs_start:],
                    max_seqlen_q=seq_lens_q.max(),
                    max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                    batch_size=len(self.seq_lens[reqs_start:]),
                    cmp_ratio=128,  #
                    ori_mask_mode=4,  # 4:sliding window
                    cmp_mask_mode=3,  # 3:causal
                    ori_win_left=self.model_config.hf_config.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="TND" if self.enable_kv_tnd else "PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                    device=str(self.seqused_q.device))
            sas_metadata = self.prefill_ratio_to_sas_metadata["c128"]
        if self.prefill_ratio_to_sas_metadata.get("qli") is None:
            self.prefill_ratio_to_sas_metadata["qli"] = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=prefill_query_start_loc[1:].clone(),
                actual_seq_lengths_key=self.seq_lens[reqs_start:].clone(),
                num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
                num_heads_k=1,
                head_dim=self.model_config.hf_config.index_head_dim,  # 128
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=len(self.seq_lens[reqs_start:]),
                max_seqlen_q=seq_lens_q.max().item(),
                max_seqlen_k=self.seq_lens[reqs_start:].max().item(),
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.model_config.hf_config.index_topk,  # 512
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=4,
                device=str(self.seqused_q.device))
        qli_metadata = self.prefill_ratio_to_sas_metadata.get("qli")

        return AscendDSAPrefillMetadata(
            attn_mask=self.attn_mask_builder.get_final_mla_mask(
                self.model_config),
            query_lens=self.query_lens[reqs_start:].to(torch.int32),
            seq_lens=self.seq_lens[reqs_start:],
            context_lens=self.seq_lens[reqs_start:],
            input_positions=prefill_input_positions,
            block_table=self.block_table[reqs_start:, ...],
            slot_mapping=prefill_slot_mapping,
            max_query_len=max_query_len,
            max_seq_lens=max_seq_lens,
            query_start_loc=prefill_query_start_loc,
            sin=sin,
            cos=cos,
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            start_pos=self.start_pos_prefill[:num_prefill],
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
            cu_c4_cmp_seqlen_list=cu_c4_cmp_seqlen_list,
            cu_c128_cmp_seqlen_list=cu_c128_cmp_seqlen_list)

    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        num_reqs_actual: Optional[int],
    ) -> AscendDSADecodeMetadata:
        if self.decode_ratio_to_sas_metadata.get("query_start_loc", None) is None:
            query_start_loc = common_attn_metadata.query_start_loc[:self.
                                                                num_decodes + 1]
            self.decode_ratio_to_sas_metadata["query_start_loc"] = query_start_loc
            input_positions = common_attn_metadata.positions[:self.
                                                            num_actual_tokens].long(
                                                            )
            self.decode_ratio_to_sas_metadata["input_positions"] = input_positions
            input_positions = input_positions[:self.num_decode_tokens]
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True)
            self.decode_ratio_to_sas_metadata["cos"] = cos
            self.decode_ratio_to_sas_metadata["sin"] = sin
        else:
            query_start_loc = self.decode_ratio_to_sas_metadata["query_start_loc"]
            input_positions = self.decode_ratio_to_sas_metadata["input_positions"]
            cos = self.decode_ratio_to_sas_metadata["cos"]
            sin = self.decode_ratio_to_sas_metadata["sin"]

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:self.
                                                                       num_decodes
                                                                       + 1]
        input_positions_cpu = common_attn_metadata.positions_cpu[:self.
                                                                 num_actual_tokens].long(
                                                                 )
        input_positions_cpu = input_positions_cpu[:self.num_decode_tokens]

        max_seq_lens = common_attn_metadata.seq_lens_cpu[:self.
                                                         num_decodes].max(
                                                         ).item()
        block_table_size = self.get_block_table_size(
            common_attn_metadata, BUILD_METADATA_STEP_DECODE)

        seq_lens_list = common_attn_metadata.seq_lens_cpu[:self.
                                                          num_decodes].tolist(
                                                          )

        cp_seq_len, batch_seq_mask = None, None

        decode_input_positions = input_positions_cpu

        def _get_padded_compressed_position(decode_input_positions,
                                            compress_ratio, device):
            if compress_ratio == 1:
                return decode_input_positions
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            input_positions = decode_input_positions[mask]
            input_positions = (input_positions + 1) - compress_ratio
            target_shape = (min(
                self.num_decode_tokens,
                self.num_decode_tokens // compress_ratio + self.num_decodes), )
            pad_right = target_shape[0] - input_positions.shape[0]
            pad_positions = F.pad(input_positions, (0, pad_right), value=0.0)
            gpu_pad_positions = pad_positions.pin_memory().to(
                device, non_blocking=True)
            return gpu_pad_positions

        layer_name = f"c{self.compressor_ratio}"
        if self.decode_ratio_to_sas_metadata.get(layer_name + "_cos", None) is None:
            compress_cos, compress_sin = get_cos_and_sin_dsa(
                {
                    layer_name:
                    _get_padded_compressed_position(decode_input_positions,
                                                    self.compressor_ratio,
                                                    input_positions.device)
                },
                use_cache=True)
            self.decode_ratio_to_sas_metadata[layer_name + "_cos"] = compress_cos
            self.decode_ratio_to_sas_metadata[layer_name + "_sin"] = compress_sin
        else:
            compress_cos = self.decode_ratio_to_sas_metadata[layer_name + "_cos"]
            compress_sin = self.decode_ratio_to_sas_metadata[layer_name + "_sin"]

        def _get_compressed_decode_token_start(decode_input_positions,
                                               compress_ratio):
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            compressed_decode_num = mask.sum().item()
            return compressed_decode_num

        compressed_tokens_start = _get_compressed_decode_token_start(
            decode_input_positions, self.compressor_ratio)

        slot_mapping = self.slot_mapping[:compressed_tokens_start]

        max_seqlen_kv = torch.max(
            common_attn_metadata.seq_lens_cpu[:self.num_decodes]).item()
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()

        assert self.start_pos_decode is not None
        self.start_pos_decode.fill_(0)
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        self.start_pos_decode[:self.
                              num_decodes] = self.seq_lens[:self.
                                                           num_decodes] - seq_lens_q

        if num_reqs_actual is not None and num_reqs_actual < self.num_decodes:
            self.start_pos_decode[num_reqs_actual:].fill_(0)
            self.block_table[num_reqs_actual:self.num_decodes, ...].fill_(0)

        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        index_topk = 512

        assert self.decode_sas_metadata is not None
        if self.compressor_ratio == 1:
            if self.decode_ratio_to_sas_metadata.get("c1") is None:
                self.decode_ratio_to_sas_metadata["c1"] = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=query_start_loc, # cached
                    cu_seqlens_ori_kv=self.cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=self.cu_seqlens_cmp_kv,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[:self.num_decodes], # cached
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=len(self.seq_lens[:self.num_decodes]), # cached
                    cmp_ratio=1,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=False,
                    device=str(self.seqused_q.device))
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata["c1"]
        elif self.compressor_ratio == 4:
            if self.decode_ratio_to_sas_metadata.get("c4") is None:
                self.decode_ratio_to_sas_metadata["c4"] = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=query_start_loc, # cached
                    cu_seqlens_ori_kv=self.cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=self.cu_seqlens_cmp_kv,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[:self.num_decodes], # cached
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=len(self.seq_lens[:self.num_decodes]), # cached
                    cmp_topk=index_topk,
                    # topk=index_topk,
                    cmp_ratio=4,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                    device=str(self.seqused_q.device))
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata["c4"]
        else:
            if self.decode_ratio_to_sas_metadata.get("c128") is None:
                self.decode_ratio_to_sas_metadata["c128"] = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=query_start_loc, # cached
                    cu_seqlens_ori_kv=self.cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=self.cu_seqlens_cmp_kv,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[:self.num_decodes], # cached
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=len(self.seq_lens[:self.num_decodes]), # cached
                    cmp_ratio=128,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                    device=str(self.seqused_q.device))
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata["c128"]
        assert self.decode_qli_metadata is not None
        if self.decode_ratio_to_sas_metadata.get("qli") is None:
            self.decode_ratio_to_sas_metadata["qli"] = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=query_start_loc[1:].clone(), # cached
                actual_seq_lengths_key=self.seq_lens[:self.num_decodes].clone(),
                num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
                num_heads_k=1,
                head_dim=self.model_config.hf_config.index_head_dim,  # 128
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=len(self.seq_lens[:self.num_decodes]), # cached
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.model_config.hf_config.index_topk,  # 512
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=4,
                device=str(self.seqused_q.device))
        self.decode_qli_metadata[:1024] = self.decode_ratio_to_sas_metadata.get("qli")
        decode_metadata = AscendDSADecodeMetadata(
            input_positions=input_positions,
            block_table=self.block_table[:block_table_size, ...],
            slot_mapping=slot_mapping,
            seq_lens=self.seq_lens[:self.num_decodes], # cached
            seq_lens_list=seq_lens_list,
            max_seq_lens=max_seq_lens,
            max_seqlen_kv=max_seqlen_kv,
            max_seqlen_q=max_seqlen_q,
            attn_mask=self.attn_mask_builder.get_splitfuse_attn_mask(),
            query_start_loc=query_start_loc, # cached
            query_start_loc_cpu=query_start_loc_cpu,
            sin=sin[:self.num_decode_tokens, ...],
            cos=cos[:self.num_decode_tokens, ...],
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            cp_seq_len=cp_seq_len,
            batch_seq_mask=batch_seq_mask,
            start_pos=self.start_pos_decode[:self.num_decodes], # cached
            sas_metadata = self.decode_sas_metadata,
            qli_metadata=self.decode_qli_metadata)
        return decode_metadata

    def get_block_table_size(
            self, common_attn_metadata: AscendCommonAttentionMetadata,
            build_metadata_step: int):
        if build_metadata_step == BUILD_METADATA_STEP_PREFILL:
            # If graph_pad_size > -1, mean is running in fullgraph mode.
            # NOTE: Maybe this block_table change can be removed when graph_pad_size > 1.
            # if self.graph_pad_size > common_attn_metadata.num_reqs and self.speculative_config.disable_padded_drafter_batch:
            #     return self.graph_pad_size
            return common_attn_metadata.num_reqs
        return self.num_decodes

    def build_for_graph_capture(
            self,
            common_attn_metadata: AscendCommonAttentionMetadata,
            attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
            **kwargs):
        if attn_state in {
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding
        }:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state"
            )

        assert attn_metadata is not None
        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendDSAImpl(DSAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        **kwargs,
    ):
        self.num_heads = n_heads
        self.n_local_heads = n_local_heads
        self.scale = scale
        self.o_lora_rank = o_lora_rank
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.head_dim = head_dim
        self.n_group = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim**-0.5

        # MLA Args
        self.wq_a = kwargs['wq_a']
        self.wq_b = kwargs['wq_b']
        self.wkv = kwargs['wkv']
        self.q_norm = kwargs['q_norm']
        self.kv_norm = kwargs['kv_norm']

        self.indexer = kwargs.get('indexer', None)
        self.compressor = kwargs.get('compressor', None)

        self.wo_a = kwargs['wo_a']
        self.wo_b = kwargs['wo_b']

        self.eps = kwargs['eps']

        self.attn_sink = kwargs['attn_sink']

        ascend_config = get_ascend_config()
        self.multistream_dsa_preprocess = ascend_config.multistream_dsa_preprocess
        self.enable_kv_tnd = ascend_config.enable_kv_tnd

        self.vllm_config = get_current_vllm_config()

        # indexer param
        if self.indexer is not None:
            self.indexer_heads: int = self.indexer.n_heads
            self.inderxer_dim: int = self.indexer.head_dim
            self.inderxer_wq_b = self.indexer.wq_b
            self.weights_proj = self.indexer.weights_proj
            self.indexer_softmax_scale = self.inderxer_dim**-0.5

            self.indexer_compress = self.indexer.compressor

            # indexer_compressor
            self.indexcom_ape = self.indexer.compressor.ape
            self.indexcom_wkv = self.indexer.compressor.wkv
            self.indexcom_wgate = self.indexer.compressor.wgate
            self.indexcom_norm = self.indexer.compressor.norm

            self.indexcom_head_dim = self.indexer.compressor.head_dim
            self.indexcom_rotate = self.indexer.compressor.rotate
            self.index_topk = 512

        # compress param
        if self.compressor is not None:
            self.compressor_head_dim = self.compressor.head_dim
            self.compressor_overlap = self.compressor.overlap
            self.compressor_rotate = self.compressor.rotate

            self.compressor_ape = self.compressor.ape
            self.compressor_wkv = self.compressor.wkv
            self.compressor_wgate = self.compressor.wgate
            self.compressor_norm = self.compressor.norm
            self.compressor_norm_eps = self.compressor.norm_eps

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass

    # TODO: cast to bfloat16 to speed up
    def rope_single(self, x, cos, sin, inverse=False):
        if inverse:
            sin = -sin
        tnd_layout = 1
        if len(x.shape) == 3:
            num_tokens, num_heads, rotary_dim = x.shape
        else:
            tnd_layout = 0
            _, num_tokens, num_heads, rotary_dim = x.shape
        x_rot = torch_npu.npu_rotary_mul(x.reshape(num_tokens, num_heads, 1,
                                                   rotary_dim),
                                         cos,
                                         sin,
                                         rotary_mode="interleave")
        if tnd_layout:
            x = x_rot.reshape(num_tokens, -1, rotary_dim)
        else:
            x = x_rot.reshape(1, num_tokens, -1, rotary_dim)
        return x

    def forward(  # type: ignore[override]
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: list[M],
        need_gather_q_kv: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        if not isinstance(attn_metadata, list):
            attn_metadata = [attn_metadata]
        output_padded = output
        # Process for Flash Comm V1
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states, need_gather_q_kv)
        has_prefill = attn_metadata[0].num_prefills > 0
        has_decode = attn_metadata[0].num_decodes > 0
        decode_tokens = attn_metadata[0].num_decode_tokens
        actual_tokens = attn_metadata[0].num_actual_tokens
        prefill_hidden_states = hidden_states[decode_tokens:actual_tokens]
        decode_hidden_states = hidden_states[:decode_tokens]

        forward_context = get_forward_context()
        o_proj_input_shape = (forward_context.num_tokens, self.n_local_heads,
                              self.head_dim)
        o_proj_input = torch.empty(o_proj_input_shape,
                                   dtype=hidden_states.dtype,
                                   device=hidden_states.device)

        if has_prefill:
            assert attn_metadata[0].prefill is not None
            output_prefill = self._forward_prefill(layer_name,
                                                   prefill_hidden_states,
                                                   kv_cache, attn_metadata)
            o_proj_input[decode_tokens:actual_tokens] = output_prefill
            cos = attn_metadata[0].prefill.cos[layer_name]
            sin = attn_metadata[0].prefill.sin[layer_name]

        if has_decode:
            assert attn_metadata[0].decode is not None
            output_decode = self._forward_decode(layer_name,
                                                 decode_hidden_states,
                                                 kv_cache, attn_metadata)
            o_proj_input[:decode_tokens] = output_decode
            cos = attn_metadata[0].decode.cos[layer_name]
            sin = attn_metadata[0].decode.sin[layer_name]

        cos = attn_metadata[0].cos[layer_name]
        sin = attn_metadata[0].sin[layer_name]
        num_tokens = o_proj_input.shape[0]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            o_proj_input.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # o
        o_proj_input = o_proj_input.view(num_tokens, self.n_local_groups, -1)
        if olora_tp_enable():
            o_proj_input = self.wo_a(o_proj_input)
        else:
            # wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            # o = torch.einsum("tgd,grd->tgr", o, wo_a)
            o_proj_input = torch_npu.npu_transpose_batchmatmul(
                o_proj_input,
                self.wo_a.weight,
                bias=None,
                scale=None,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
                batch_split_factor=1)
            o_proj_input = o_proj_input.reshape(num_tokens, -1)
        output[...] = self.wo_b(o_proj_input)

        return output_padded

    def _forward_prefill(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: Tuple,
        attn_metadata: AscendDSAMetadata,
    ):
        compress_common_attn_metadata = None
        if self.compress_ratio == 4:
            (swa_cache, compressor_attn_cache, _, _, compressor_kv_state, compressor_score_state, _, _) = kv_cache
            (swa_metadata, compressor_attn_metadata, _, compressor_kv_state_metadata, compressor_score_state_metadata, _, _) = attn_metadata
            compress_common_attn_metadata = compressor_attn_metadata
        elif self.compress_ratio == 128:
            (swa_cache, compressor_attn_cache, compressor_kv_state, compressor_score_state) = kv_cache
            (swa_metadata, compressor_attn_metadata, compressor_kv_state_metadata, compressor_score_state_metadata) = attn_metadata
            compress_common_attn_metadata = compressor_attn_metadata
        else:
            (swa_cache,) = kv_cache
            (swa_metadata,) = attn_metadata
            compress_common_attn_metadata = swa_metadata
        
        assert compress_common_attn_metadata.prefill is not None
        cos = compress_common_attn_metadata.prefill.cos[layer_name]
        sin = compress_common_attn_metadata.prefill.sin[layer_name]
        actual_seq_lengths_query = compress_common_attn_metadata.prefill.query_start_loc
        actual_seq_lengths_key = compress_common_attn_metadata.prefill.seq_lens

        # mlaprolog
        # q
        qr = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = triton_q_rms(q, self.eps)

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        # win kv & tok_dis
        kv = self.wkv(hidden_states)
        kv = self.kv_norm(kv)
        assert self.rope_head_dim is not None
        kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # swa exec kv
        torch_npu.npu_scatter_nd_update_(
            swa_cache.view(-1, kv.shape[-1]),
            swa_metadata.prefill.slot_mapping.unsqueeze(-1), kv)
        compress_cos = compress_common_attn_metadata.prefill.compress_cos[layer_name]
        compress_sin = compress_common_attn_metadata.prefill.compress_sin[layer_name]
        if self.compress_ratio > 1:
            compress_topk_idxs = None
            if self.compress_ratio == 4:
                compress_topk_idxs = self.indexer_select_qli(
                    x=hidden_states,
                    qr=qr,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    cos=cos,
                    sin=sin,
                    compressed_cos=compress_cos,
                    compressed_sin=compress_sin,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    actual_seq_lengths_key=actual_seq_lengths_key,
                    with_prefill=True)

            coff = 2 if self.compressor_overlap else 1

            # compressor
            compressed_kv, _, _, _, _ = torch.ops._C_ascend.compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                compressor_kv_state.squeeze(-2),
                compressor_score_state.squeeze(-2),
                self.compressor_ape,
                self.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                kv_block_table=compressor_kv_state_metadata.prefill.block_table,
                score_block_table=compressor_score_state_metadata.prefill.block_table,
                cu_seqlens=actual_seq_lengths_query,
                seqused=None,
                start_pos=compress_common_attn_metadata.prefill.start_pos,
                rope_head_dim=self.rope_head_dim,
                cmp_ratio=self.compress_ratio,
                coff=coff,
                norm_eps=self.compressor_norm_eps,
                rotary_mode=2,
                enable_grad=False)

            if compressed_kv.numel() == 0:
                compressed_kv = None

            # kv_compress_epilog
            torch_npu.npu_scatter_nd_update_(
                compressor_attn_cache.view(-1, compressed_kv.shape[-1]),
                compressor_attn_metadata.prefill.slot_mapping.unsqueeze(-1),
                compressed_kv.view(-1, compressed_kv.shape[-1]))

        sliding_window_kv = kv if self.enable_kv_tnd else swa_cache

        if self.compress_ratio == 1:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=sliding_window_kv,
                ori_block_table=swa_metadata.prefill.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                cu_seqlens_ori_kv=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compress_common_attn_metadata.prefill.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="TND" if self.enable_kv_tnd else "PA_ND")[0]
        elif self.compress_ratio == 4:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=sliding_window_kv,
                cmp_kv=compressed_kv.unsqueeze(1)
                if self.enable_kv_tnd else compressor_attn_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_metadata.prefill.block_table,
                cmp_block_table=compressor_attn_metadata.prefill.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                cu_seqlens_ori_kv=actual_seq_lengths_query,
                cu_seqlens_cmp_kv=compress_common_attn_metadata.prefill.cu_c4_cmp_seqlen_list,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compress_common_attn_metadata.prefill.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="TND" if self.enable_kv_tnd else "PA_ND")[0]
        else:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=sliding_window_kv,
                cmp_kv=compressed_kv.unsqueeze(1)
                if self.enable_kv_tnd else compressor_attn_cache,
                ori_block_table=swa_metadata.prefill.block_table,
                cmp_block_table=compressor_attn_metadata.prefill.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                cu_seqlens_ori_kv=actual_seq_lengths_query,
                cu_seqlens_cmp_kv=compress_common_attn_metadata.prefill.
                cu_c128_cmp_seqlen_list,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compressor_attn_metadata.prefill.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="TND" if self.enable_kv_tnd else "PA_ND")[0]
        return attn_output

    def _forward_decode(
            self,
            layer_name,
            hidden_states: torch.Tensor,
            kv_cache: Tuple,
            attn_metadata: AscendDSAMetadata,
    ):
        assert attn_metadata[0].decode is not None
        compress_common_attn_metadata = None
        if self.compress_ratio == 4:
            (swa_cache, compressor_attn_cache, _, _, compressor_kv_state, compressor_score_state, _, _) = kv_cache
            (swa_metadata, compressor_attn_metadata, _, compressor_kv_state_metadata, compressor_score_state_metadata, _, _) = attn_metadata
            compress_common_attn_metadata = compressor_attn_metadata
        elif self.compress_ratio == 128:
            (swa_cache, compressor_attn_cache, compressor_kv_state, compressor_score_state) = kv_cache
            (swa_metadata, compressor_attn_metadata, compressor_kv_state_metadata, compressor_score_state_metadata) = attn_metadata
            compress_common_attn_metadata = compressor_attn_metadata
        else:
            (swa_cache,) = kv_cache
            (swa_metadata,) = attn_metadata
            compress_common_attn_metadata = swa_metadata
        cos = compress_common_attn_metadata.decode.cos[layer_name]
        sin = compress_common_attn_metadata.decode.sin[layer_name]
        actual_seq_lengths_query = compress_common_attn_metadata.decode.query_start_loc
        actual_seq_lengths_key = compress_common_attn_metadata.decode.seq_lens

        wait_hidden_state_cal_event = torch.npu.current_stream().record_event() \
            if self.multistream_dsa_preprocess else None

        # q
        if (not isinstance(self.wq_b.quant_method, AscendUnquantizedLinearMethod)) and \
                isinstance(self.wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod):
            q_a = self.wq_a(hidden_states)
            qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                q_a, self.q_norm.weight, epsilon=self.eps)
            q = torch_npu.npu_quant_matmul(
                qr,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            ).unflatten(-1, (self.n_local_heads, self.head_dim))
        else:
            qr = q = self.q_norm(self.wq_a(hidden_states))
            q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
            qr_pertoken_scale = None

        q = triton_q_rms(q, self.eps)

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        with npu_stream_switch(attention_calculation_stream(),
                               enabled=self.multistream_dsa_preprocess):
            if wait_hidden_state_cal_event:
                torch.npu.current_stream().wait_event(
                    wait_hidden_state_cal_event)

            # win kv & tok_dis
            kv = self.wkv(hidden_states)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )

            # swa exec kv
            torch_npu.npu_scatter_nd_update_(
                swa_cache.view(-1, kv.shape[-1]),
                swa_metadata.decode.slot_mapping.unsqueeze(-1), kv)

            wait_attention_cal_event = torch.npu.current_stream().record_event() \
                if self.multistream_dsa_preprocess else None

        if wait_attention_cal_event:
            torch.npu.current_stream().wait_event(wait_attention_cal_event)

        if self.compress_ratio > 1:
            compress_cos = compress_common_attn_metadata.decode.compress_cos[layer_name]
            compress_sin = compress_common_attn_metadata.decode.compress_sin[layer_name]
            compress_topk_idxs = None
            if self.compress_ratio == 4:
                compress_topk_idxs = self.indexer_select_qli(
                    x=hidden_states,
                    qr=qr,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    cos=cos,
                    sin=sin,
                    compressed_cos=compress_cos,
                    compressed_sin=compress_sin,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    actual_seq_lengths_key=actual_seq_lengths_key,
                    with_prefill=False,
                    qr_pertoken_scale=qr_pertoken_scale)

            coff = 2 if self.compressor_overlap else 1

            # compressor
            compressed_kv, _, _, _, _ = torch.ops._C_ascend.compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                compressor_kv_state.squeeze(-2),
                compressor_score_state.squeeze(-2),
                self.compressor_ape,
                self.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                kv_block_table=compressor_kv_state_metadata.decode.block_table,
                score_block_table=compressor_score_state_metadata.decode.block_table,
                cu_seqlens=actual_seq_lengths_query,
                seqused=None,
                start_pos=compress_common_attn_metadata.decode.start_pos,
                rope_head_dim=self.rope_head_dim,
                cmp_ratio=self.compress_ratio,
                coff=coff,
                norm_eps=self.compressor_norm_eps,
                rotary_mode=2,
                enable_grad=False)
            # kv_compress_epilog
            torch_npu.npu_scatter_nd_update_(
                compressor_attn_cache.view(-1, compressed_kv.shape[-1]),
                compressor_attn_metadata.decode.slot_mapping.unsqueeze(-1),
                compressed_kv.view(-1, compressed_kv.shape[-1]))
        if self.compress_ratio == 1:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_cache,
                ori_block_table=swa_metadata.decode.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=swa_metadata.decode.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND")[0]
        elif self.compress_ratio == 4:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_cache,
                cmp_kv=compressor_attn_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_metadata.decode.block_table,
                cmp_block_table=compressor_attn_metadata.decode.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compressor_attn_metadata.decode.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND")[0]
        else:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_cache,
                cmp_kv=compressor_attn_cache,
                ori_block_table=swa_metadata.decode.block_table,
                cmp_block_table=compressor_attn_metadata.decode.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compressor_attn_metadata.decode.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND")[0]
        return attn_output

    def indexer_select_qli(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: list[M],
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        with_prefill: bool = False,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        (_, _, indexer_kv_cache, indexer_scale_cache, _, _, indexer_kv_state, indexer_kv_score_state) = kv_cache
        (_, _, indexer_kv_scale_metadata, _, _, indexer_kv_state_metadata, indexer_kv_score_state_metadata) = attn_metadata

        if (not isinstance(self.inderxer_wq_b.quant_method, AscendUnquantizedLinearMethod)) and \
            isinstance(self.inderxer_wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod) and \
            qr_pertoken_scale is not None:
            q = torch_npu.npu_quant_matmul(
                qr,
                self.inderxer_wq_b.weight,
                self.inderxer_wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.inderxer_wq_b.bias,
                output_dtype=x.dtype,
            )
        else:
            q = self.inderxer_wq_b(qr)
        q = q.view(-1, self.indexer_heads, self.indexcom_head_dim)  # [T, N, D]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[
                self.indexcom_head_dim - self.rope_head_dim,
                self.indexcom_head_dim
            ],
        )

        q = rotate_activation(q, indexer_kv_scale_metadata.hadamard)
        coff = 2 if self.compressor_overlap else 1

        if with_prefill:
            assert indexer_kv_scale_metadata.prefill is not None
            kv_block_table = indexer_kv_state_metadata.prefill.block_table
            score_block_table = indexer_kv_score_state_metadata.prefill.block_table
            start_pos = indexer_kv_scale_metadata.prefill.start_pos
        else:
            assert indexer_kv_scale_metadata.decode is not None
            kv_block_table = indexer_kv_state_metadata.decode.block_table
            score_block_table = indexer_kv_score_state_metadata.decode.block_table
            start_pos = indexer_kv_scale_metadata.decode.start_pos

        kv, _, _, _, _ = torch.ops._C_ascend.compressor(
            x,
            self.indexcom_wkv.weight,
            self.indexcom_wgate.weight,
            indexer_kv_state.squeeze(-2),
            indexer_kv_score_state.squeeze(-2),
            self.indexcom_ape,
            self.indexcom_norm.weight,
            compressed_sin.view(-1, compressed_sin.shape[-1]),
            compressed_cos.view(-1, compressed_cos.shape[-1]),
            kv_block_table=kv_block_table,
            score_block_table=score_block_table,
            cu_seqlens=actual_seq_lengths_query,
            seqused=None,
            start_pos=start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=coff,
            norm_eps=self.compressor_norm_eps,
            rotary_mode=2,
            enable_grad=False)

        if kv.numel() == 0:
            kv = None
        elif self.indexer.compressor.rotate:
            kv = rotate_activation(kv, indexer_kv_scale_metadata.hadamard)

        weights = self.weights_proj(x) * (self.indexer_softmax_scale *
                                          self.indexer_heads ** -0.5)

        soc_version = get_ascend_device_type()
        dst_type = torch.float8_e4m3fn if soc_version in {AscendDeviceType.A5
                                                          } else torch.int8

        q, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=dst_type)
        if kv is not None:
            kv, kv_scale = torch_npu.npu_dynamic_quant(kv, dst_type=dst_type)
            kv_scale = kv_scale.unsqueeze(-1)

        if soc_version not in {AscendDeviceType.A5}:
            q_scale = q_scale.to(torch.float16)
            if kv is not None:
                kv_scale = kv_scale.to(torch.float16)
                kv_scale = kv_scale.unsqueeze(-1)

        if with_prefill:
            assert indexer_kv_scale_metadata.prefill is not None
            if kv is not None:
                torch_npu.npu_scatter_nd_update_(
                    indexer_kv_cache.view(-1, kv.shape[-1]),
                    indexer_kv_scale_metadata.prefill.slot_mapping.unsqueeze(-1),
                    kv.view(-1, kv.shape[-1]))
                torch_npu.npu_scatter_nd_update_(
                    indexer_scale_cache.view(-1, kv_scale.shape[-1]),
                    indexer_kv_scale_metadata.prefill.slot_mapping.unsqueeze(-1),
                    kv_scale.view(-1, kv_scale.shape[-1]))
        else:
            assert indexer_kv_scale_metadata.decode is not None
            if kv is not None:
                torch_npu.npu_scatter_nd_update_(
                    indexer_kv_cache.view(-1, kv.shape[-1]),
                    indexer_kv_scale_metadata.decode.slot_mapping.unsqueeze(-1),
                    kv.view(-1, kv.shape[-1]))
                torch_npu.npu_scatter_nd_update_(
                    indexer_scale_cache.view(-1, kv_scale.shape[-1]),
                    indexer_kv_scale_metadata.decode.slot_mapping.unsqueeze(-1),
                    kv_scale.view(-1, kv_scale.shape[-1]))

        if with_prefill:
            assert indexer_kv_scale_metadata.prefill is not None
            qlens = indexer_kv_scale_metadata.prefill.query_start_loc[1:]
            kvlens = indexer_kv_scale_metadata.prefill.seq_lens
            block_table = indexer_kv_scale_metadata.prefill.block_table
            qli_metadata = indexer_kv_scale_metadata.prefill.qli_metadata
        else:
            assert indexer_kv_scale_metadata.decode is not None
            qlens = indexer_kv_scale_metadata.decode.query_start_loc[1:]
            kvlens = indexer_kv_scale_metadata.decode.seq_lens
            block_table = indexer_kv_scale_metadata.decode.block_table
            qli_metadata = indexer_kv_scale_metadata.decode.qli_metadata

        topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer(
            query=q,
            key=indexer_kv_cache,
            weights=weights.to(torch.float16),
            query_dequant_scale=q_scale,
            key_dequant_scale=indexer_scale_cache.squeeze(-2),
            actual_seq_lengths_query=qlens,
            actual_seq_lengths_key=kvlens,
            block_table=block_table,
            metadata=qli_metadata,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=512,
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            return_value=False)
        return topk_idxs
