from dataclasses import dataclass
from typing import (TYPE_CHECKING, ClassVar, NamedTuple, Optional, Tuple, Type,
                    TypeVar)

import numpy as np
import math
import torch
import torch_npu
import torch.nn.functional as F
import vllm.envs as envs_vllm
from vllm.attention.backends.abstract import AttentionBackend, DSAAttentionImpl
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.math_utils import cdiv, round_down
# from vllm.v1.attention.backends.mla.common import MLACommonMetadataBuilder
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         enable_cp,
                                         maybe_save_kv_layer_to_connector,
                                         split_decodes_and_prefills,
                                         trans_rope_weight, transdata,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params, get_graph_params,
    update_draft_graph_params_workspaces, update_graph_params_workspaces)
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.ops.triton.rms_norm import triton_q_rms
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, npu_stream_switch, attention_calculation_stream

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

BUILD_METADATA_STEP_PREFILL = 0
BUILD_METADATA_STEP_DECODE = 1


def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    def _get_window_topk_idxs():
        if start_pos >= window_size - 1:
            return torch.arange(window_size)
        elif start_pos > 0:
            return F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
        else:
            base = torch.arange(seqlen).unsqueeze(1)
            matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
            matrix = torch.where(matrix > base, -1, matrix)
            return matrix
    return _get_window_topk_idxs().unsqueeze(0).expand(bsz, -1, -1)


def hadamard_transform_ref(x: torch.Tensor, scale=1.0, attn_metadata=None):
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, attn_metadata.hadamard)
    out = x * scale
    return out[..., :dim].reshape(*x_shape)

def rotate_activation(x: torch.Tensor, attn_metadata) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, scale=hidden_size ** -0.5, attn_metadata=attn_metadata)
import math


def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    def _get_compress_topk_idxs():
        if start_pos > 0:
            return torch.arange(0, start_pos // ratio) + offset
        else:
            matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
            mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            matrix = torch.where(mask, -1, matrix + offset)
            return matrix
    return _get_compress_topk_idxs().unsqueeze(0).expand(bsz, -1, -1)


def pad_to_blocks(x: torch.Tensor, length_list: torch.Tensor, block_size: int = 128):
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
    out = torch.zeros(
        (total_blocks, block_size, n, d),
        dtype=x.dtype,
        device=x.device
    )

    # 4. Fill data
    input_offset = 0
    block_offset = 1

    for i in range(bs):
        length = length_list[i]
        num_blocks = blocks_per_req[i]

        if length > 0:
            # Slice the valid data for this request from the packed input
            # Shape: [length, n, d]
            req_data = x[input_offset: input_offset + length]

            # Select the assigned blocks in the output
            # Shape: [num_blocks, block_size, n, d]
            target_blocks = out[block_offset: block_offset + num_blocks]

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

def generate_cache_and_block_table_and_slot_mapping(x: torch.Tensor, length_list: torch.Tensor, block_size: int = 128):
    # 1. Validation
    if x.shape[0] != length_list.sum():
        raise ValueError(
            f"Input dimension 0 ({x.shape[0]}) does not match sum of length_list ({length_list.sum()})"
        )
        
    block_num_list = []
    for length in length_list.tolist():
        block_num_list.append((length + block_size - 1) // block_size)
        
    cache = torch.zeros((sum(block_num_list) + 1, block_size, x.shape[-2], 640), dtype=torch.float8_e4m3fn).to(x.device)
    block_table = torch.zeros((len(block_num_list), max(block_num_list)), dtype=torch.int32, device=x.device)
    
    # build blocktable
    num_blocks = 0
    for i, block_num in enumerate(block_num_list):
        block_table[i, :block_num] = torch.arange(num_blocks, num_blocks + block_num) + 1  # 不用第一个block
        num_blocks += block_num
        
    block_num_list_cumsum = torch.Tensor(block_num_list).to(torch.int32).cumsum(-1).tolist()
    
    # build slot_mapping
    slot_mapping_list = []
    for i, length in enumerate(length_list):
        if i == 0:
            slot_mapping_list.extend([j for j in range(0, length)])
        else:
            slot_mapping_list.extend([j for j in range(block_num_list_cumsum[i - 1] * block_size, block_num_list_cumsum[i - 1] * block_size + length)])
    slot_mapping = torch.Tensor(slot_mapping_list).to(torch.int32).to(x.device)
    slot_mapping += 128  # 不用第一个block
            
    return cache, block_table, slot_mapping



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
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            return num_blocks, block_size, num_kv_heads, 640
        else:
            return num_blocks, block_size, num_kv_heads, head_size
        
    @staticmethod
    def get_indexer_kcache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_scale_shape(num_blocks: int, block_size: int, scale_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, scale_size

    @staticmethod
    def get_impl_cls() -> Type["DSAAttentionImpl"]:
        return AscendDSAImpl

    @staticmethod
    def get_supported_block_size() -> list[int]:
        return [128]


@dataclass
class ChunkedContextMetadata:
    # New for MLA (compared to FlashAttention)
    # For handling chunked prefill
    cu_seq_lens: torch.Tensor
    starts: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    workspace: torch.Tensor
    chunk_seq_lens: torch.Tensor
    chunk_seq_lens_npu: torch.Tensor

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
    prefill_swa_block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_query_len: int
    max_seq_lens: int

    swa_slot_mapping: torch.Tensor
    swa_block_table: torch.Tensor
    state_block_table: torch.Tensor

    chunked_context: Optional[ChunkedContextMetadata] = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    compress_sin: torch.Tensor = None
    compress_cos: torch.Tensor = None
    start_pos: Optional[torch.Tensor] = None
    sas_c1_metadata: torch.Tensor = None
    sas_c4_metadata: torch.Tensor = None
    sas_c128_metadata: torch.Tensor = None

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

    swa_slot_mapping: torch.Tensor
    swa_block_table: torch.Tensor
    state_block_table: torch.Tensor

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
    sas_c1_metadata: torch.Tensor = None
    sas_c4_metadata: torch.Tensor = None
    sas_c128_metadata: torch.Tensor = None


@dataclass
class AscendDSAMetadata:
    """Metadata for MLACommon.
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor
    swa_slot_mapping: torch.Tensor
    swa_block_table: torch.Tensor
    state_block_table: torch.Tensor


    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
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
    indexer_metadata: Optional[torch.Tensor] = None

    hadamard: Optional[torch.Tensor] = None

    start_pos: Optional[torch.Tensor] = None

    def __post_init__(self):
        pass


M = TypeVar("M", bound=AscendDSAMetadata)


class AscendDSAMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH
    indexer_metadata = None
    hadamard = None
    start_pos_prefill: Optional[torch.Tensor] = None
    start_pos_decode: Optional[torch.Tensor] = None
    decode_sas_c1_metadata: Optional[torch.Tensor] = None
    decode_sas_c4_metadata: Optional[torch.Tensor] = None
    decode_sas_c128_metadata: Optional[torch.Tensor] = None
    
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
        self.metadata_cls = (metadata_cls if metadata_cls is not None else
                             AscendDSAMetadata)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        scheduler_config = vllm_config.scheduler_config
        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len +
                           self.block_size - 1) // self.block_size
        # NOTE: For deepseek v4, this is disabled by default now in `check_and_update_config`
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

        self.reorder_batch_threshold = self.decode_threshold
        if self.chunked_prefill_enabled:
            self.chunked_prefill_workspace_size = min(
                # Max sure there is enough for 8 full length request or at least
                # 4 pages of cache per request
                max(8 * self.model_config.max_model_len,
                    4 * scheduler_config.max_num_seqs * self.block_size),
                # For long-context models try not to over-allocate limiting
                # kv-cache space, limiting it to 64k tokens,
                # which would result in the workspace being:
                #   2*(576)*(64*1024) = 144mb
                # (assuming 576 MLA head dim, and fp16)
                # which would result in up-projected context being
                #   2*(192*128)*(64*1024) = 3gb
                # (assuming 192 QK head dim, 128 heads, and fp16)
                128 * 1024)
            assert self.chunked_prefill_workspace_size >= \
                   scheduler_config.max_num_seqs * self.block_size
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size,
                 self.model_config.get_head_size()),
                dtype=self.model_config.dtype,
                device=device,
            )
        self.rope_dim = self.model_config.hf_text_config.rope_head_dim
        self.cos_cache = None
        self.sin_cache = None

        self.chunk_seq_lens: torch.Tensor = None
        self.cu_seq_lens_cpu: torch.Tensor = None
        self.num_chunks: torch.Tensor = None
        self.max_context_chunk = 0
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

        self.compressor_ratio = 1
        if AscendDSAMetadataBuilder.indexer_metadata is None:
            # TODO(cmq): What does the magic number for?
            AscendDSAMetadataBuilder.indexer_metadata = torch.zeros((2048),
                                                                    dtype = torch.int32,
                                                                    device=self.device)
            usedCoreNum = 24
            mBaseSize = 256
            s2BaseSize = 2048
            AscendDSAMetadataBuilder.indexer_metadata[:3] = torch.tensor(
                [usedCoreNum, mBaseSize, s2BaseSize],
                dtype=torch.int32,
                device=self.device)

            # AscendDSAMetadataBuilder.metadata[3:27] = torch.zeros((usedCoreNum), dtype = torch.int32, device=q.device) #bN2End 每个核处理数据的BN2结束点
            # AscendDSAMetadataBuilder.metadata[27:51] = torch.zeros((usedCoreNum), dtype = torch.int32, device=q.device) #mEnd 每个核处理数据的M结束点
            # AscendDSAMetadataBuilder.metadata[51:75] = torch.zeros((usedCoreNum), dtype = torch.int32, device=q.device) #s2End 每个核处理数据的S2结束点

        if AscendDSAMetadataBuilder.hadamard is None:
            hf_config = self.model_config.hf_config
            if hf_config.model_type == 'deepseek_v4':
                indexer_head_dim = hf_config.index_head_dim
                try:
                    from scipy.linalg import hadamard
                except ImportError as e:
                    raise ImportError("Please install scipy") from e
                log_dim = math.ceil(math.log2(indexer_head_dim))
                dim_padded = 2 ** log_dim
                AscendDSAMetadataBuilder.hadamard = torch.tensor(hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device)        
        AscendDSAMetadataBuilder.start_pos_prefill = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
        AscendDSAMetadataBuilder.start_pos_decode = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
        if AscendDSAMetadataBuilder.decode_sas_c1_metadata is None:
            AscendDSAMetadataBuilder.decode_sas_c1_metadata = torch.zeros(2048, dtype=torch.int32, device=self.device)
        if AscendDSAMetadataBuilder.decode_sas_c4_metadata is None:
            AscendDSAMetadataBuilder.decode_sas_c4_metadata = torch.zeros(2048, dtype=torch.int32, device=self.device)
        if AscendDSAMetadataBuilder.decode_sas_c128_metadata is None:
            AscendDSAMetadataBuilder.decode_sas_c128_metadata = torch.zeros(2048, dtype=torch.int32, device=self.device)


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


    def build_chunked_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        if not self.chunked_prefill_enabled:
            return None
        num_reqs = common_attn_metadata.num_reqs

        num_computed_tokens_cpu = (self.seq_lens - self.query_lens)
        reqs_start = self.num_decodes

        self.context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
        max_context_len_cpu = self.context_lens_cpu.max().item()
        if not max_context_len_cpu > 0:
            return None
        num_prefills_with_context_cpu = (self.context_lens_cpu
                                         > 0).sum().item()
        self.max_context_chunk = (self.chunked_prefill_workspace_size //
                                  num_prefills_with_context_cpu)
        self.max_context_chunk = round_down(self.max_context_chunk,
                                            self.block_size)

        assert self.max_context_chunk > 0
        self.num_chunks = cdiv(max_context_len_cpu, self.max_context_chunk)
        chunk_starts = torch.arange(self.num_chunks, dtype=torch.int32) \
                           .unsqueeze(1).expand(-1, self.num_prefills) * self.max_context_chunk
        chunk_ends = torch.min(self.context_lens_cpu.unsqueeze(0),
                               chunk_starts + self.max_context_chunk)
        self.chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)
        self.cu_seq_lens_cpu = torch.zeros(self.num_chunks,
                                           self.num_prefills + 1,
                                           dtype=torch.int32,
                                           pin_memory=True)
        torch.cumsum(self.chunk_seq_lens,
                     dim=1,
                     out=self.cu_seq_lens_cpu[:, 1:],
                     dtype=torch.int32)
        return ChunkedContextMetadata(
            cu_seq_lens=self.cu_seq_lens_cpu.pin_memory().to(
                self.device, non_blocking=True),
            starts=chunk_starts.pin_memory().to(self.device,
                                                non_blocking=True),
            seq_tot=self.chunk_seq_lens.sum(dim=1).tolist(),
            max_seq_lens=self.chunk_seq_lens.max(dim=1).values.tolist(),
            chunk_seq_lens=self.chunk_seq_lens,
            chunk_seq_lens_npu=self.chunk_seq_lens.npu(),
            workspace=self.chunked_prefill_workspace,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc

        self.compressor_ratio =  kwargs.get('compress_ratio', 1)

        self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
        self.set_num_actual_tokens(common_attn_metadata)
        assert self.num_decodes + self.num_prefills == num_reqs
        assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens

        # zyl TODO: remove
        num_input_tokens = common_attn_metadata.num_input_tokens
        input_positions = common_attn_metadata.positions[:
                                                         num_input_tokens].long(
                                                         )
        if self.num_prefills:
            cos, sin = get_cos_and_sin_dsa(input_positions)
        else:
            cos, sin = get_cos_and_sin_dsa(input_positions, True)


        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        self.slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        self.query_lens = query_seq_lens_cpu[:num_reqs]

        self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]

        self.graph_pad_size = common_attn_metadata.graph_pad_size
        block_table_size = self.get_block_table_size(
            common_attn_metadata, BUILD_METADATA_STEP_PREFILL)
        self.block_table = common_attn_metadata.block_table_tensor[:block_table_size]

        prefill_metadata = None
        if self.num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata(
                common_prefix_len, common_attn_metadata)

        decode_metadata = None

        if self.num_decodes > 0:
            decode_metadata = self.build_decode_metadata(
                common_prefix_len, common_attn_metadata)

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            query_lens=self.query_lens,
            slot_mapping=self.slot_mapping,
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
            block_tables=self.block_table,
            seq_lens=self.seq_lens,
            cos=cos,
            sin=sin,
            swa_slot_mapping=common_attn_metadata.swa_slot_mapping,
            swa_block_table=common_attn_metadata.swa_block_table,
            state_block_table=common_attn_metadata.state_block_table,
            indexer_metadata=AscendDSAMetadataBuilder.indexer_metadata,
            hadamard = AscendDSAMetadataBuilder.hadamard,
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

        chunked_context_metadata = self.build_chunked_metadata(
            common_prefix_len, common_attn_metadata)
        # reqs_start: the start request position of prefill request
        reqs_start = self.num_decodes
        # reqs_start: the start token postion of prefill request
        tokens_start = self.num_decode_tokens

        max_query_len = self.query_lens[reqs_start:].max().item()
        max_seq_lens = common_attn_metadata.seq_lens_cpu[reqs_start:].max().item()
        prefill_query_start_loc = query_start_loc[
            reqs_start:] - query_start_loc[reqs_start]

        prefill_input_positions = input_positions[tokens_start:]
        cos, sin = get_cos_and_sin_dsa(prefill_input_positions)

        def _get_padded_compressed_position(prefill_input_positions, compress_ratio):
            # TODO(lxs): refactor me to get_compressed_pos_and_indices
            if compress_ratio == 1:
                return prefill_input_positions
            mask = ((prefill_input_positions + 1) % compress_ratio) == 0
            input_positions = prefill_input_positions[mask]
            input_positions = (input_positions + 1) - compress_ratio
            target_shape = (min(self.num_prefill_tokens,
                                self.num_prefill_tokens // compress_ratio + self.num_prefills),)
            pad_right = target_shape[0] - input_positions.shape[0]
            pad_positions = F.pad(input_positions, (0, pad_right), value=0.0)
            return pad_positions

        compress_cos, compress_sin = get_cos_and_sin_dsa(_get_padded_compressed_position(prefill_input_positions, self.compressor_ratio))

        # tmp swa_block
        prefill_seq_lens = self.seq_lens[reqs_start:]
        # TODO(cmq): refactor this magic number
        prefill_swa_block = (prefill_seq_lens + 128 - 1) // 128
        cumsum_prefill_swa_block = prefill_swa_block.cumsum(dim=0)
        prefill_swa_block_ids = torch.arange(1, cumsum_prefill_swa_block[-1] + 1,
                                dtype=self.block_table.dtype,
                                device=self.block_table.device)
        num_prefill = prefill_seq_lens.shape[0]
        prefill_swa_block_table_shape = (num_prefill,
                                         self.vllm_config.model_config.max_model_len // 128)
        prefill_swa_block_table = torch.zeros(prefill_swa_block_table_shape,
                                         dtype=self.block_table.dtype,
                                         device=self.block_table.device)

        for i in range(num_prefill):
            start_idx = cumsum_prefill_swa_block[i] - prefill_swa_block[i]
            end_idx = cumsum_prefill_swa_block[i]
            prefill_swa_block_table[i, :prefill_swa_block[i]] = prefill_swa_block_ids[start_idx:end_idx]

        prefill_swa_slot_mapping = common_attn_metadata.swa_slot_mapping[tokens_start:]

        # TODO: zyl refactor this for asc scheduling.
        decode_input_positions = input_positions[:tokens_start]
        def _get_compressed_decode_token_start_and_end(decode_input_positions, compress_ratio):
            # TODO(cmq): decode_input_positions is a device tensor, 
            # this will introduce sync operation. Refactor me to torch.where instead
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            compressed_decode_num = mask.sum()

            end = min(self.num_prefill_tokens, self.num_prefill_tokens // compress_ratio + self.num_prefills)
            return compressed_decode_num, end

        compressed_tokens_start, compressed_tokens_end = _get_compressed_decode_token_start_and_end(decode_input_positions, self.compressor_ratio)

        prefill_slot_mapping = self.slot_mapping[compressed_tokens_start:compressed_tokens_end]

        AscendDSAMetadataBuilder.start_pos_prefill.fill_(0)
        seq_lens_q = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
        AscendDSAMetadataBuilder.start_pos_prefill[:num_prefill] = self.seq_lens[reqs_start:] - seq_lens_q

        # AscendDSAMetadataBuilder.start_pos[reqs_start:self.seq_lens[reqs_start:].shape[0] + 1] = self.seq_lens[reqs_start:] - seq_lens_q
        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.n_heads // tp_size
        index_topk = 512
        
        sas_c1_metadata = torch_npu.npu_kv_quant_sparse_attn_sharedkv_metadata(
            kv_quant_mode = 1,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=prefill_query_start_loc,
            seqused_kv=self.seq_lens[reqs_start:],
            max_seqlen_q=seq_lens_q.max(),
            max_seqlen_kv=self.seq_lens[reqs_start:].max(),
            batch_size=len(self.seq_lens[reqs_start:]),
            ori_mask_mode=4, # 4:sliding window
            ori_win_left=self.model_config.hf_config.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False
        )
        
        sas_c4_metadata = torch_npu.npu_kv_quant_sparse_attn_sharedkv_metadata(
            kv_quant_mode = 1,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=prefill_query_start_loc,
            seqused_kv=self.seq_lens[reqs_start:],
            max_seqlen_q=seq_lens_q.max(),
            max_seqlen_kv=self.seq_lens[reqs_start:].max(),
            batch_size=len(self.seq_lens[reqs_start:]),
            cmp_topk=index_topk, #
            cmp_ratio=4, #
            ori_mask_mode=4, # 4:sliding window
            cmp_mask_mode=3, # 3:causal
            ori_win_left=self.model_config.hf_config.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=True
        )
        
        sas_c128_metadata = torch_npu.npu_kv_quant_sparse_attn_sharedkv_metadata(
            kv_quant_mode = 1,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=prefill_query_start_loc,
            seqused_kv=self.seq_lens[reqs_start:],
            max_seqlen_q=seq_lens_q.max(),
            max_seqlen_kv=self.seq_lens[reqs_start:].max(),
            batch_size=len(self.seq_lens[reqs_start:]),
            cmp_ratio=128, #
            ori_mask_mode=4, # 4:sliding window
            cmp_mask_mode=3, # 3:causal
            ori_win_left=self.model_config.hf_config.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=True
        )
    
        return AscendDSAPrefillMetadata(
            attn_mask=self.attn_mask_builder.get_final_mla_mask(
                self.model_config),
            query_lens=self.query_lens[reqs_start:].to(torch.int32),
            seq_lens=self.seq_lens[reqs_start:],
            context_lens=self.seq_lens[reqs_start:],
            input_positions=prefill_input_positions,
            block_table=self.block_table[reqs_start:, ...],
            prefill_swa_block_table=prefill_swa_block_table,
            slot_mapping = prefill_slot_mapping,
            swa_slot_mapping=prefill_swa_slot_mapping,
            swa_block_table=common_attn_metadata.swa_block_table[reqs_start:, ...],
            state_block_table=common_attn_metadata.state_block_table[reqs_start:, ...],
            max_query_len=max_query_len,
            max_seq_lens=max_seq_lens,
            query_start_loc=prefill_query_start_loc,
            chunked_context=chunked_context_metadata,
            sin=sin,
            cos=cos,
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            start_pos=AscendDSAMetadataBuilder.start_pos_prefill[:num_prefill],
            sas_c1_metadata=sas_c1_metadata,
            sas_c4_metadata=sas_c4_metadata,
            sas_c128_metadata=sas_c128_metadata
        )

    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendDSADecodeMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[:self.num_decodes + 1]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:self.num_decodes + 1]

        input_positions = common_attn_metadata.positions[:self.
                                                         num_actual_tokens].long(
                                                         )
        input_positions = input_positions[:self.num_decode_tokens]

        input_positions_cpu = common_attn_metadata.positions_cpu[:self.
                                                         num_actual_tokens].long(
                                                         )
        input_positions_cpu = input_positions_cpu[:self.num_decode_tokens]

        # Notice that num_decodes != num_decode_tokens in SpecDecoding Scenario
        # actual_seq_lengths_q = query_start_loc_cpu[1:self.num_decodes +
        #                                            1].tolist()
        max_seq_lens = common_attn_metadata.seq_lens_cpu[:self.num_decodes].max().item()

        block_table_size = self.get_block_table_size(
            common_attn_metadata, BUILD_METADATA_STEP_DECODE)

        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        # NOTE: Maybe this block_table change can be removed when graph_pad_size > 1.
        # if self.graph_pad_size > self.num_decodes and \
        #         self.speculative_config.disable_padded_drafter_batch:
        #     self.block_table = self.block_table[:self.graph_pad_size, ...]
        seq_lens_list = common_attn_metadata.seq_lens_cpu[:self.num_decodes].tolist()

        cp_seq_len, batch_seq_mask = None, None

        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True)

        decode_input_positions = input_positions_cpu

        def _get_padded_compressed_position(decode_input_positions, compress_ratio):
            # TODO(lxs): refactor me to get_compressed_pos_and_indices
            if compress_ratio == 1:
                return decode_input_positions
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            input_positions = decode_input_positions[mask]
            # # why not - compress_ratio here?
            # input_positions = (input_positions + 1) - compress_ratio
            target_shape = (min(self.num_decode_tokens,
                                self.num_decode_tokens // compress_ratio + self.num_decodes),)
            pad_right = target_shape[0] - input_positions.shape[0]
            pad_positions = F.pad(input_positions, (0, pad_right), value=0.0)
            return pad_positions

        layer_name = f"c{self.compressor_ratio}"
        compress_cos, compress_sin = get_cos_and_sin_dsa(
            {layer_name: _get_padded_compressed_position(input_positions, self.compressor_ratio)},
            use_cache=True)


        def _get_compressed_decode_token_start(decode_input_positions, compress_ratio):
            # TODO(cmq): decode_input_positions is a device tensor, 
            # this will introduce sync operation. Refactor me to torch.where instead
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            compressed_decode_num = mask.sum()
            return compressed_decode_num

        compressed_tokens_start = _get_compressed_decode_token_start(decode_input_positions, self.compressor_ratio)

        slot_mapping = self.slot_mapping[:compressed_tokens_start]

        decode_swa_slot_mapping = common_attn_metadata.swa_slot_mapping[:self.num_decode_tokens]

        max_seqlen_kv = torch.max(common_attn_metadata.seq_lens_cpu[:self.num_decodes]).item()
        max_seqlen_q = torch.max(query_start_loc_cpu).item()
        AscendDSAMetadataBuilder.start_pos_decode.fill_(0)
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        AscendDSAMetadataBuilder.start_pos_decode[:self.num_decodes] = self.seq_lens[:self.num_decodes] - seq_lens_q
        
        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.n_heads // tp_size
        index_topk = 512
        AscendDSAMetadataBuilder.decode_sas_c1_metadata[:2048] = torch_npu.npu_kv_quant_sparse_attn_sharedkv_metadata(
            kv_quant_mode = 1,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=query_start_loc,
            seqused_kv=self.seq_lens[:self.num_decodes],
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=len(self.seq_lens[:self.num_decodes]),
            ori_mask_mode=4, # 4:sliding window
            cmp_mask_mode=3, # 3:causal
            ori_win_left=self.model_config.hf_config.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False
        )
        
        AscendDSAMetadataBuilder.decode_sas_c4_metadata[:2048] = torch_npu.npu_kv_quant_sparse_attn_sharedkv_metadata(
            kv_quant_mode = 1,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=query_start_loc,
            seqused_kv=self.seq_lens[:self.num_decodes],
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=len(self.seq_lens[:self.num_decodes]),
            cmp_topk=index_topk, #
            cmp_ratio=4, #
            ori_mask_mode=4, # 4:sliding window
            cmp_mask_mode=3, # 3:causal
            ori_win_left=self.model_config.hf_config.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=True
        )
        
        AscendDSAMetadataBuilder.decode_sas_c128_metadata[:2048] = torch_npu.npu_kv_quant_sparse_attn_sharedkv_metadata(
            kv_quant_mode = 1,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=query_start_loc,
            seqused_kv=self.seq_lens[:self.num_decodes],
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=len(self.seq_lens[:self.num_decodes]),
            cmp_ratio=128, #
            ori_mask_mode=4, # 4:sliding window
            cmp_mask_mode=3, # 3:causal
            ori_win_left=self.model_config.hf_config.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=True
        )
        
        decode_metadata = AscendDSADecodeMetadata(
            input_positions=input_positions,
            block_table=self.block_table[:block_table_size, ...],
            swa_block_table=common_attn_metadata.swa_block_table[:block_table_size, ...],
            slot_mapping=slot_mapping,
            swa_slot_mapping=decode_swa_slot_mapping,
            seq_lens=self.seq_lens[:self.num_decodes],
            seq_lens_list=seq_lens_list,
            max_seq_lens=max_seq_lens,
            max_seqlen_kv=max_seqlen_kv,
            max_seqlen_q=max_seqlen_q,
            attn_mask=self.attn_mask_builder.get_splitfuse_attn_mask(),
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            state_block_table=common_attn_metadata.state_block_table[:self.num_decodes],
            sin=sin[:self.num_decode_tokens, ...],
            cos=cos[:self.num_decode_tokens, ...],
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            cp_seq_len=cp_seq_len,
            batch_seq_mask=batch_seq_mask,
            start_pos=AscendDSAMetadataBuilder.start_pos_decode[:self.num_decodes],
            sas_c1_metadata=AscendDSAMetadataBuilder.decode_sas_c1_metadata,
            sas_c4_metadata=AscendDSAMetadataBuilder.decode_sas_c4_metadata,
            sas_c128_metadata=AscendDSAMetadataBuilder.decode_sas_c128_metadata)
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
        **kwargs
    ):
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
        self.softmax_scale = self.head_dim ** -0.5


        # MLA Args
        self.wq_a = kwargs['wq_a']
        self.wq_b = kwargs['wq_b']
        self.wkv = kwargs['wkv']
        self.q_norm = kwargs['q_norm']
        self.kv_norm = kwargs['kv_norm']

        self.indexer = kwargs.get('indexer', None)
        self.compressor =  kwargs.get('compressor', None)

        self.wo_a = kwargs['wo_a']
        self.wo_b = kwargs['wo_b']

        self.eps = 1e-6 # zyl

        self.attn_sink = kwargs['attn_sink']

        ascend_config = get_ascend_config()
        self.multistream_dsa_preprocess = ascend_config.multistream_dsa_preprocess
        # self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        # self.enable_prefetch = ascend_config.weight_prefetch_config.enabled
        # self.enable_mlapo = envs.VLLM_ASCEND_ENABLE_MLAPO

        self.vllm_config = get_current_vllm_config()

        # indexer param
        if self.indexer is not None:
            self.indexer_heads: int = self.indexer.n_heads  # 32
            self.inderxer_dim: int = self.indexer.head_dim  # 128
            self.inderxer_wq_b = self.indexer.wq_b    # (1024, 32*128)
            self.weights_proj = self.indexer.weights_proj   # (4096, 32)
            self.indexer_softmax_scale = self.inderxer_dim ** -0.5

            self.indexer_compress = self.indexer.compressor

            # indexer_compressor
            self.indexcom_ape = self.indexer.compressor.ape
            self.indexcom_wkv = self.indexer.compressor.wkv
            self.indexcom_wgate = self.indexer.compressor.wgate
            self.indexcom_norm = self.indexer.compressor.norm

            self.indexcom_head_dim = self.indexer.compressor.head_dim
            self.indexcom_rotate = self.indexer.compressor.rotate
            self.index_topk=512

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

    def compress_forward(self,
            x: torch.Tensor,
            kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            wkx,
            wgate,
            norm_weight,
            sin,
            cos,
            attentionmeatdata,
            rope_head_dim,
            compress_ratio,
            rotary_mode
    ):
        kv = self.kernel_compress(x)# self.wkv, self.wgate, self.kv_state, self.score_state
        return kv

    def kernel_compreess(self,
                         x):
        return None

    # TODO: cast to bfloat16 to speed up
    def rope_single(self, x,cos,sin,inverse=False):
        # dtype= x.dtype
        if inverse:
            # sin = sin * -1
            sin = -sin
        tnd_layout = 1
        if len(x.shape)==3:
            num_tokens,num_heads,rotary_dim = x.shape
        else:
            tnd_layout=0
            _,num_tokens,num_heads,rotary_dim = x.shape
        # x_rot = torch_npu.npu_rotary_mul(x.reshape(num_tokens, num_heads, 1, rotary_dim).to(torch.float32), cos, sin, rotary_mode="interleave")
        x_rot = torch_npu.npu_rotary_mul(x.reshape(num_tokens, num_heads, 1, rotary_dim), cos, sin, rotary_mode="interleave")
        if tnd_layout:
            x = x_rot.reshape(num_tokens, -1, rotary_dim)
        else:
            x = x_rot.reshape(1,num_tokens, -1, rotary_dim)
        return x

    def get_compress_topk_idxs(
        self,
        x
    ):
        return None

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: Optional[torch.Tensor] = None,
        kv_state: Tuple[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        output_padded = output
        # if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
        #     output[...]  =  self._forward_single_op_prefill(
        #         hidden_states,
        #         kv_cache,
        #         attn_metadata,
        #         kv_state,
        #         layer_name
        #     )
        # elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
        #     output[...]  =  self._forward_single_op_decode(
        #         hidden_states,
        #         kv_cache,
        #         attn_metadata,
        #         kv_state,
        #         layer_name
        #     )
        # return output_padded


        has_prefill = attn_metadata.num_prefills > 0
        has_decode = attn_metadata.num_decodes > 0
        decode_tokens = attn_metadata.num_decode_tokens
        actual_tokens = attn_metadata.num_actual_tokens
        prefill_hidden_states = hidden_states[decode_tokens:actual_tokens]
        decode_hidden_states = hidden_states[:decode_tokens]

        forward_context = get_forward_context()
        output_padded = output
        o_proj_input_shape = (forward_context.num_tokens,
                              self.n_local_heads, self.head_dim)
        o_proj_input = torch.empty(o_proj_input_shape,
                                   dtype=hidden_states.dtype,
                                   device=hidden_states.device)

        if has_prefill:
            output_prefill = self._forward_prefill(
                layer_name,
                prefill_hidden_states,
                kv_cache,
                attn_metadata,
                kv_state)
            o_proj_input[decode_tokens:actual_tokens] = output_prefill
            cos = attn_metadata.prefill.cos[layer_name]
            sin = attn_metadata.prefill.sin[layer_name]

        if has_decode:
            output_decode = self._forward_decode(
                layer_name,
                decode_hidden_states,
                kv_cache,
                attn_metadata,
                kv_state)
            o_proj_input[:decode_tokens] = output_decode
            cos = attn_metadata.decode.cos[layer_name]
            sin = attn_metadata.decode.sin[layer_name]

        cos = attn_metadata.cos[layer_name]
        sin = attn_metadata.sin[layer_name]
        num_tokens = o_proj_input.shape[0]
        o_nope, o_pe = o_proj_input.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        o_pe = self.rope_single(o_pe, cos, sin, True)
        o = torch.cat([o_nope, o_pe], dim=-1)

        # o
        o = o.view(num_tokens, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("tgd,grd->tgr", o, wo_a)
        o = o.reshape(num_tokens, -1)
        output[...] = self.wo_b(o)

        return output_padded

    def _forward_prefill(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: Tuple,
        attn_metadata: AscendDSAMetadata,
        kv_state: Tuple,
    ):
        if self.compress_ratio==1:
            (sliding_window_state) = kv_state
        elif self.compress_ratio==4:
            (sliding_window_state, compressor_kv_state, compressor_score_state,
             c4_indexer_kv_state, c4_indexer_score_state) = kv_state
        elif self.compress_ratio==128:
            (sliding_window_state, compressor_kv_state, compressor_score_state) = kv_state

        cos = attn_metadata.prefill.cos[layer_name]
        sin = attn_metadata.prefill.sin[layer_name]
        actual_seq_lengths_query = attn_metadata.prefill.query_start_loc
        actual_seq_lengths_key = attn_metadata.prefill.seq_lens
        num_decode_tokens = attn_metadata.num_decode_tokens
        max_seqlen_kv = attn_metadata.prefill.max_seq_lens
        seq_lens_q = actual_seq_lengths_query[1:] - actual_seq_lengths_query[:-1]
        max_seqlen_q = attn_metadata.prefill.max_query_len
        compressed_kv_block_table = attn_metadata.prefill.block_table
        compressed_kv_slot_mapping = attn_metadata.prefill.slot_mapping

        # mlaprolog
        # q
        qr = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        # q = triton_q_rms(q, self.eps)
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = self.rope_single(q_pe, cos, sin)
        q = torch.cat([q_nope, q_pe], dim=-1)
        # win kv & tok_dis
        kv = self.wkv(hidden_states)
        kv = self.kv_norm(kv)
        kv = kv.view(-1, 1, self.nope_head_dim+self.rope_head_dim)
        kv_nope, kv_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        kv_pe = self.rope_single(kv_pe, cos, sin)
        kv = torch.cat([kv_nope, kv_pe], dim=-1)

        # swa exec kv
        torch.ops.custom.kv_compress_epilog(
            kv_state[0].view(-1, kv_state[0].shape[-1]),
            kv.view(-1, kv.shape[-1]),
            attn_metadata.prefill.swa_slot_mapping,
            quant_group_size=64,
            quant_mode=2
        )
        compress_cos = attn_metadata.prefill.compress_cos[layer_name]
        compress_sin = attn_metadata.prefill.compress_sin[layer_name]
        if self.compress_ratio > 1:
            compress_topk_idxs = None
            if self.compress_ratio == 4:
                compress_topk_idxs = self.indexer_select_qli(x=hidden_states,
                                                    qr=qr,
                                                    kv_cache=kv_cache,
                                                    kv_state=kv_state,
                                                    attn_metadata=attn_metadata,
                                                    cos=cos,
                                                    sin=sin,
                                                    compressed_cos=compress_cos,
                                                    compressed_sin=compress_sin,
                                                    actual_seq_lengths_query=actual_seq_lengths_query,
                                                    actual_seq_lengths_key=actual_seq_lengths_key,
                                                    with_prefill=True)

            coff = 2 if self.compressor_overlap else 1
            # start_pos = actual_seq_lengths_key - seq_lens_q
            # compressor
            compressed_kv = torch.ops.custom.compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                compressor_kv_state,
                compressor_score_state,
                self.compressor_ape,
                self.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                kv_block_table = attn_metadata.prefill.state_block_table,
                score_block_table = attn_metadata.prefill.state_block_table,
                cu_seqlens = actual_seq_lengths_query,
                seqused = None, #actual_seq_lengths_key,
                start_pos = attn_metadata.prefill.start_pos,
                rope_head_dim = self.rope_head_dim,
                cmp_ratio = self.compress_ratio,
                coff = coff,
                norm_eps = self.compressor_norm_eps,
                rotary_mode = 2
            )

            if compressed_kv.numel() == 0:
                compressed_kv = None
            # elif self.compressor.rotate:
            #     compressed_kv = rotate_activation(compressed_kv, attn_metadata)

            torch.ops.custom.kv_compress_epilog(
                kv_cache[0].view(-1, kv_cache[0].shape[-1]),
                compressed_kv.reshape(-1, compressed_kv.shape[-1]),
                compressed_kv_slot_mapping[:compressed_kv.numel() // compressed_kv.shape[-1]],
                quant_group_size=64,
                quant_mode=2
            )

        cache, block_table, slot_mapping = generate_cache_and_block_table_and_slot_mapping(kv, actual_seq_lengths_key, block_size=128)
        torch.ops.custom.kv_compress_epilog(
            cache,
            kv.reshape(-1, kv.shape[-1]),
            slot_mapping,
            quant_group_size=64,
            quant_mode=2
        )

        if self.compress_ratio == 1:
            attn_output, _ = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv(
                q,
                ori_kv=cache,
                ori_block_table=block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=attn_metadata.prefill.sas_c1_metadata,
                kv_quant_mode=1,
                tile_size=64,
                rope_head_dim=64,
                softmax_scale=self.softmax_scale,
                ori_mask_mode=4, # 4:sliding window
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND"
            )
        elif self.compress_ratio == 4:
            attn_output, _ = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv(
                q,
                ori_kv=cache,
                cmp_kv=kv_cache[0],
                cmp_sparse_indices=compress_topk_idxs, # B T D 
                ori_block_table=block_table,
                cmp_block_table=compressed_kv_block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=attn_metadata.prefill.sas_c4_metadata,
                kv_quant_mode=1,
                tile_size=64,
                rope_head_dim=64,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio, #
                ori_mask_mode=4, # 4:sliding window
                cmp_mask_mode=3, # 3:causal
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND"
            )
        else:
            attn_output, _ = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv(
                q,
                ori_kv=cache,
                cmp_kv=kv_cache[0],
                ori_block_table=block_table,
                cmp_block_table=compressed_kv_block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=attn_metadata.prefill.sas_c128_metadata,
                kv_quant_mode=1,
                tile_size=64,
                rope_head_dim=64,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio, #
                ori_mask_mode=4, # 4:sliding window
                cmp_mask_mode=3, # 3:causal
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND"
            )
        return attn_output

    def _partial_rope(self, x, cos, sin):
        x_nope, x_pe = x.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        x_pe = self.rope_single(x_pe, cos, sin)
        return torch.cat([x_nope, x_pe], dim=-1)

    def _forward_decode(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: Tuple,
        attn_metadata: AscendDSAMetadata,
        kv_state: Tuple,
    ):
        if self.compress_ratio==1:
            (sliding_window_state) = kv_state
        elif self.compress_ratio==4:
            (sliding_window_state, compressor_kv_state, compressor_score_state, c4_indexer_kv_state, c4_indexer_score_state) = kv_state
        elif self.compress_ratio==128:
            (sliding_window_state, compressor_kv_state, compressor_score_state) = kv_state

        cos = attn_metadata.decode.cos[layer_name]
        sin = attn_metadata.decode.sin[layer_name]
        actual_seq_lengths_query = attn_metadata.decode.query_start_loc
        actual_seq_lengths_key = attn_metadata.decode.seq_lens
        compressed_kv_block_table = attn_metadata.decode.block_table
        compressed_kv_slot_mapping = attn_metadata.decode.slot_mapping

        # win kv & tok_dis
        kv = self.wkv(hidden_states)
        before_norm_event = torch.npu.current_stream().record_event() \
            if self.multistream_dsa_preprocess else None
        kv = self.kv_norm(kv)
        kv = kv.view(-1, 1, self.nope_head_dim+self.rope_head_dim)
        kv = self._partial_rope(kv, cos, sin)

        # swa exec kv
        torch.ops.custom.kv_compress_epilog(
            kv_state[0].view(-1, kv_state[0].shape[-1]),
            kv.view(-1, kv.shape[-1]),
            attn_metadata.decode.swa_slot_mapping,
            quant_group_size=64,
            quant_mode=2
        )

        # q
        with npu_stream_switch(attention_calculation_stream(), enabled=self.multistream_dsa_preprocess):
            if before_norm_event:
                torch.npu.current_stream().wait_event(before_norm_event)

            qr = q = self.wq_a(hidden_states) # bs
        if self.multistream_dsa_preprocess:
            torch.npu.current_stream().wait_stream(
                attention_calculation_stream())
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim)) # tp
        # q = triton_q_rms(q, self.eps)
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q = self._partial_rope(q, cos, sin)

        if self.compress_ratio > 1:
            compress_cos = attn_metadata.decode.compress_cos[layer_name]
            compress_sin = attn_metadata.decode.compress_sin[layer_name]
            compress_topk_idxs = None
            if self.compress_ratio == 4:
                compress_topk_idxs = self.indexer_select_qli(x=hidden_states,
                                                    qr=qr,
                                                    kv_cache=kv_cache,
                                                    kv_state=kv_state,
                                                    attn_metadata=attn_metadata,
                                                    cos=cos,
                                                    sin=sin,
                                                    compressed_cos=compress_cos,
                                                    compressed_sin=compress_sin,
                                                    actual_seq_lengths_query=actual_seq_lengths_query,
                                                    actual_seq_lengths_key=actual_seq_lengths_key,
                                                    with_prefill=False)

            coff = 2 if self.compressor_overlap else 1

            # compressor
            compressed_kv = torch.ops.custom.compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                compressor_kv_state,
                compressor_score_state,
                self.compressor_ape,
                self.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                kv_block_table = attn_metadata.decode.state_block_table,
                score_block_table = attn_metadata.decode.state_block_table,
                cu_seqlens = actual_seq_lengths_query,
                seqused = None, #actual_seq_lengths_key,
                start_pos = attn_metadata.decode.start_pos,
                rope_head_dim = self.rope_head_dim,
                cmp_ratio = self.compress_ratio,
                coff = coff,
                norm_eps = self.compressor_norm_eps,
                rotary_mode = 2
            )

            if len(compressed_kv_slot_mapping.tolist()):
                torch.ops.custom.kv_compress_epilog(
                    kv_cache[0].view(-1, kv_cache[0].shape[-1]),
                    compressed_kv.view(-1, compressed_kv.shape[-1]),
                    compressed_kv_slot_mapping,
                    quant_group_size=64,
                    quant_mode=2
                )

        if self.compress_ratio == 1:
            attn_output, _ = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv(
                q,
                ori_kv=kv_state[0].unsqueeze(2),
                ori_block_table=attn_metadata.decode.state_block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=attn_metadata.decode.sas_c1_metadata,
                kv_quant_mode=1,
                tile_size=64,
                rope_head_dim=64,
                softmax_scale=self.softmax_scale,
                ori_mask_mode=4, # 4:sliding window
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND"
            )
        elif self.compress_ratio == 4:
            attn_output, _ = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv(
                q,
                ori_kv=kv_state[0].unsqueeze(2),
                cmp_kv=kv_cache[0],
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=attn_metadata.decode.state_block_table,
                cmp_block_table=compressed_kv_block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=attn_metadata.decode.sas_c4_metadata,
                kv_quant_mode=1,
                tile_size=64,
                rope_head_dim=64,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio, #
                ori_mask_mode=4, # 4:sliding window
                cmp_mask_mode=3, # 3:causal
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND"
            )
        else:
            attn_output, _ = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv(
                q,
                ori_kv=kv_state[0].unsqueeze(2),
                cmp_kv=kv_cache[0],
                ori_block_table=attn_metadata.decode.state_block_table,
                cmp_block_table=compressed_kv_block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=attn_metadata.decode.sas_c128_metadata,
                kv_quant_mode=1,
                tile_size=64,
                rope_head_dim=64,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio, #
                ori_mask_mode=4, # 4:sliding window
                cmp_mask_mode=3, # 3:causal
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND"
            )
        return attn_output 

    def indexer_select_qli(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        kv_state: Tuple,
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        with_prefill: bool = False,
    ):
        seqlen, _ = x.size()  # [T, H]
        bsz = 1
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        (_, _, _, c4_indexer_kv_state, c4_indexer_score_state) = kv_state
        seq_lens_q = actual_seq_lengths_query[1:] - actual_seq_lengths_query[:-1]
        q = self.inderxer_wq_b(qr)
        q = q.view(-1, self.indexer_heads, self.indexcom_head_dim)  # [T, N, D]
        ## rope
        q_nope, q_pe = q.split([self.indexcom_head_dim-self.rope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = self.rope_single(q_pe, cos, sin)
        q = torch.cat([q_nope, q_pe], dim=-1)

        q = rotate_activation(q, attn_metadata)

        coff = 2 if self.compressor_overlap else 1
        # start_pos = actual_seq_lengths_key - seq_lens_q

        if with_prefill:
            kv_block_table = attn_metadata.prefill.state_block_table
            score_block_table = attn_metadata.prefill.state_block_table
        else:
            kv_block_table = attn_metadata.decode.state_block_table
            score_block_table = attn_metadata.decode.state_block_table

        kv = torch.ops.custom.compressor(
            x,
            self.indexcom_wkv.weight,
            self.indexcom_wgate.weight,
            c4_indexer_kv_state,
            c4_indexer_score_state,
            self.indexcom_ape,
            self.indexcom_norm.weight,
            compressed_sin.view(-1, compressed_sin.shape[-1]),
            compressed_cos.view(-1, compressed_cos.shape[-1]),
            kv_block_table = kv_block_table,
            score_block_table = score_block_table,
            cu_seqlens = actual_seq_lengths_query,
            seqused = None, #actual_seq_lengths_key,
            start_pos = attn_metadata.prefill.start_pos if with_prefill else attn_metadata.decode.start_pos,
            rope_head_dim = self.rope_head_dim,
            cmp_ratio = self.compress_ratio,
            coff = coff,
            norm_eps = self.compressor_norm_eps,
            rotary_mode = 2
        )

        if kv.numel() == 0:
            kv = None
        elif self.indexer.compressor.rotate:
            kv = rotate_activation(kv, attn_metadata)

        weights = self.weights_proj(x) * (self.indexer_softmax_scale * self.indexcom_head_dim ** -0.5)

        soc_version = get_ascend_device_type()
        dst_type = torch.float8_e4m3fn if soc_version in {AscendDeviceType.A5} else torch.int8

        q, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=dst_type)
        # if kv is not None:
        #     kv, kv_scale = torch_npu.npu_dynamic_quant(kv, dst_type=dst_type)

        #     kv_scale = kv_scale.unsqueeze(-1)

        if soc_version not in {AscendDeviceType.A5}:
            q_scale = q_scale.to(torch.float16)
            if kv is not None:
                kv_scale = kv_scale.to(torch.float16)
                kv_scale = kv_scale.unsqueeze(-1)

        if with_prefill:
            if kv is not None:
                torch.ops.custom.indexer_compress_epilog(
                    kv_cache[1].view(-1, kv.shape[-1]),
                    kv_cache[2].view(-1, kv_cache[2].shape[-1]),
                    kv, 
                    attn_metadata.prefill.slot_mapping, 
                    quant_mode=1, 
                    round_scale=True
                )
        else:
            if kv is not None and len(attn_metadata.decode.slot_mapping.tolist()):
                torch.ops.custom.indexer_compress_epilog(
                    kv_cache[1].view(-1, kv.shape[-1]),
                    kv_cache[2].view(-1, kv_cache[2].shape[-1]),
                    kv, 
                    attn_metadata.decode.slot_mapping, 
                    quant_mode=1, 
                    round_scale=True
                )

        if with_prefill:
            qlens = attn_metadata.prefill.query_start_loc[1:]
            kvlens = attn_metadata.prefill.seq_lens
            block_table = attn_metadata.prefill.block_table
            max_seqlen_q = attn_metadata.prefill.max_query_len
            max_seqlen_kv = attn_metadata.prefill.max_seq_lens
        else:
            qlens = attn_metadata.decode.query_start_loc[1:]
            kvlens = attn_metadata.decode.seq_lens
            block_table = attn_metadata.decode.block_table
            max_seqlen_q = attn_metadata.decode.max_seqlen_q
            max_seqlen_kv = attn_metadata.decode.max_seqlen_kv

        topk_idxs, _ = torch.ops.custom.npu_quant_lightning_indexer(
            query=q,
            key=kv_cache[1],
            weights=weights.to(torch.float),
            query_dequant_scale=q_scale,
            key_dequant_scale=kv_cache[2],
            actual_seq_lengths_query=qlens,
            actual_seq_lengths_key=kvlens,
            block_table=block_table,
            metadata=metadata,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=512,
            sparse_mode=3,
            pre_tokens = (1<<63)-1,
            next_tokens = (1<<63)-1,
            cmp_ratio = 4,
            return_value = False
        )
        return topk_idxs
