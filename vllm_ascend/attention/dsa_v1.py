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
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.ops.pypto import AttentionPostV4
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

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


def hadamard_transform_ref(x: torch.Tensor, scale=1.0):
    from scipy.linalg import hadamard
    if hadamard is None:
        raise ImportError("Please install scipy")
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * scale
    return out[..., :dim].reshape(*x_shape)

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, scale=hidden_size ** -0.5)
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

def sparse_attn_torch(
    q: torch.Tensor, 
    kv: torch.Tensor, 
    attn_sink: torch.Tensor, 
    topk_idxs: torch.Tensor, 
    softmax_scale: float
) -> torch.Tensor:
    q= q.unsqueeze(0)
    kv=kv.unsqueeze(0).squeeze(2)
    topk_idxs=topk_idxs.to(q.device)
    # print(f'q.shape: {q.shape}, kv.shape: {kv.shape}, topk_ids.shape: {topk_idxs.shape}')
    b, m, h, d = q.shape
    
    # Prepare indices: clamp -1 to 0 for gathering, but keep mask
    mask = (topk_idxs == -1)
    safe_idxs = topk_idxs.clone()
    safe_idxs[mask] = 0
    
    # Gather KV: (b, m, topk, d)
    batch_indices = torch.arange(b, device=kv.device).view(b, 1, 1)
    kv_gathered = kv[batch_indices, safe_idxs, :]
    
    # Compute Scores (FP32)
    q_f32 = q.float()
    kv_f32 = kv_gathered.float()
    
    # (b, m, h, 1, d) @ (b, m, 1, topk, d)^T -> (b, m, h, topk)
    scores = torch.matmul(q_f32.unsqueeze(3), kv_f32.unsqueeze(2).transpose(-1, -2)).squeeze(3)
    scores = scores * softmax_scale
    scores = scores.masked_fill(mask.unsqueeze(2), float("-inf"))
    
    # Softmax logic with Sink
    scores_max = torch.max(scores, dim=-1).values # (b, m, h)
    exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))
    exp_scores = exp_scores.masked_fill(mask.unsqueeze(2), 0.0)
    
    sum_exp = exp_scores.sum(dim=-1)
    sink_term = torch.exp(attn_sink.float().view(1, 1, h) - scores_max)
    total_denominator = sum_exp + sink_term
    
    # Weighted Sum
    numerator = torch.matmul(exp_scores.unsqueeze(3), kv_f32.unsqueeze(2)).squeeze(3)
    output = numerator / total_denominator.unsqueeze(-1)
    output=output.squeeze(0)
    return output.to(q.dtype)

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
    total_blocks = blocks_per_req.sum().item() + 1

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
        length = int(length_list[i].item())
        num_blocks = int(blocks_per_req[i].item())

        if length > 0:
            # Slice the valid data for this request from the packed input
            # Shape: [length, n, d]
            req_data = x[input_offset : input_offset + length]

            # Select the assigned blocks in the output
            # Shape: [num_blocks, block_size, n, d]
            target_blocks = out[block_offset : block_offset + num_blocks]

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
    c4_sin: torch.Tensor = None
    c4_cos: torch.Tensor = None
    c128_sin: torch.Tensor = None
    c128_cos: torch.Tensor = None

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
    attn_mask: Optional[torch.Tensor] = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    c4_sin: torch.Tensor = None
    c4_cos: torch.Tensor = None
    c128_sin: torch.Tensor = None
    c128_cos: torch.Tensor = None
    cp_seq_len: torch.Tensor = None
    batch_seq_mask: torch.Tensor = None



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


    def __post_init__(self):
        pass
        # supported_head_sizes = AscendMLABackend.get_supported_head_sizes()
        # if self.head_dim is not None and self.head_dim \
        #         not in supported_head_sizes:
        #     raise ValueError(
        #         f"Only {supported_head_sizes} are supported for head_dim,",
        #         f"received {self.head_dim}.")


M = TypeVar("M", bound=AscendDSAMetadata)


class AscendDSAMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH
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
        self.chunked_prefill_enabled = False
        # self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill #zyl 

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
        reqs_start = self.num_decodes  # prefill_start

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
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc

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
        )
    
    def build_prefill_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendDSAPrefillMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc

        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        input_positions = common_attn_metadata.positions[:self.
                                                         num_actual_tokens].long(
                                                         )

        chunked_context_metadata = self.build_chunked_metadata(
            common_prefix_len, common_attn_metadata)
        reqs_start = self.num_decodes  # prefill_start
        tokens_start = self.num_decode_tokens

        
        max_query_len = self.query_lens[reqs_start:].max().item()
        max_seq_lens = common_attn_metadata.seq_lens_cpu[reqs_start:num_reqs].max().item()
        prefill_query_start_loc = query_start_loc[
            reqs_start:] - query_start_loc[reqs_start]

        prefill_input_positions = input_positions[tokens_start:]
        cos, sin = get_cos_and_sin_dsa(prefill_input_positions)

        # c4 rope
        c4_mask = ((prefill_input_positions+1) % 4) == 0
        c4_input_positions = prefill_input_positions[c4_mask]
        c4_input_positions = (c4_input_positions + 1) - 4
        c4_target_shape = (min(self.num_prefill_tokens, self.num_prefill_tokens // 4 + self.num_prefills),)
        pad_right = c4_target_shape[0] - c4_input_positions.shape[0]
        c4_pad_positions = F.pad(c4_input_positions, (0, pad_right), value=0.0)
        c4_cos, c4_sin = get_cos_and_sin_dsa(c4_pad_positions)


        # c128 rope
        c128_mask = ((prefill_input_positions+1) % 128) == 0
        c128_input_positions = prefill_input_positions[c128_mask]
        c128_input_positions = (c128_input_positions + 1) - 128
        c128_target_shape = (min(self.num_prefill_tokens, self.num_prefill_tokens // 128 + self.num_prefills),)
        pad_right = c128_target_shape[0] - c128_input_positions.shape[0]
        c128_pad_positions = F.pad(c128_input_positions, (0, pad_right), value=0.0)
        c128_cos, c128_sin = get_cos_and_sin_dsa(c128_pad_positions)

        # tmp swa_block
        # [8,129,257]
        prefill_seq_len = self.seq_lens[reqs_start:]
        # [1,2,3]
        prefill_block = (prefill_seq_len + 128 - 1) // 128
        # [1,3,6]
        block_cumsum = prefill_block.cumsum(dim=0)
        end = block_cumsum[-1]
        block_id = torch.arange(1, end + 1,
                                dtype=self.block_table.dtype,
                                device=self.block_table.device)
        num_prefill = self.seq_lens[reqs_start:].shape[0]
        # [num_req, max_model_len // block_size]
        prefill_block_table_shape = (num_prefill, 65536//128)

        prefill_swa_block_table = torch.zeros(prefill_block_table_shape,
                                         dtype=self.block_table.dtype,
                                         device=self.block_table.device)
        
        for i in range(num_prefill):
            start_idx = block_cumsum[i] - prefill_block[i]
            end_idx = block_cumsum[i]
            prefill_swa_block_table[i, :prefill_block[i]] = block_id[start_idx:end_idx]

        prefill_swa_slot_mapping = common_attn_metadata.swa_slot_mapping[tokens_start:]
        prefill_slot_mapping = self.slot_mapping[tokens_start:]

        block_table = self.block_table[reqs_start:, ...]

        return AscendDSAPrefillMetadata(
            attn_mask=self.attn_mask_builder.get_final_mla_mask(
                self.model_config),
            query_lens=self.query_lens[reqs_start:].to(torch.int32),
            seq_lens=self.seq_lens[reqs_start:],
            context_lens=self.seq_lens[reqs_start:],
            input_positions=prefill_input_positions,
            block_table=block_table,
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
            c4_sin=c4_sin,
            c4_cos=c4_cos,
            c128_sin=c128_sin,
            c128_cos=c128_cos,
        )

    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendDSADecodeMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[:self.num_decodes + 1]

        input_positions = common_attn_metadata.positions[:self.
                                                         num_actual_tokens].long(
                                                         )
        input_positions = input_positions[:self.num_decode_tokens]

        # Notice that num_decodes != num_decode_tokens in SpecDecoding Scenario
        # actual_seq_lengths_q = query_start_loc_cpu[1:self.num_decodes +
        #                                            1].tolist()
        max_seq_lens = common_attn_metadata.seq_lens_cpu[:self.num_decodes].max().item()

        block_table_size = self.get_block_table_size(
            common_attn_metadata, BUILD_METADATA_STEP_DECODE)

        block_table = self.block_table[:block_table_size, ...]
        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        # NOTE: Maybe this block_table change can be removed when graph_pad_size > 1.
        # if self.graph_pad_size > self.num_decodes and \
        #         self.speculative_config.disable_padded_drafter_batch:
        #     self.block_table = self.block_table[:self.graph_pad_size, ...]
        seq_lens_list = common_attn_metadata.seq_lens_cpu[:self.num_decodes].tolist()

        cp_seq_len, batch_seq_mask = None, None

        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True)

        decode_swa_slot_mapping = common_attn_metadata.swa_slot_mapping[:self.num_decode_tokens]
        slot_mapping = self.slot_mapping[:self.num_decode_tokens]


        decode_input_positions = input_positions
        # c4 rope
        c4_mask = ((decode_input_positions+1) % 4) == 0
        c4_input_positions = decode_input_positions[c4_mask]
        c4_target_shape = (min(self.num_decode_tokens, self.num_decode_tokens // 4 + self.num_decodes),)
        pad_right = c4_target_shape[0] - c4_input_positions.shape[0]
        c4_pad_positions = F.pad(c4_input_positions, (0, pad_right), value=0.0)
        # c4_cos, c4_sin = get_cos_and_sin_dsa(c4_pad_positions)
        c4_cos, c4_sin = get_cos_and_sin_dsa({"c4": c4_pad_positions},use_cache=True)

        # c128 rope
        c128_mask = ((decode_input_positions+1) % 128) == 0
        c128_input_positions = decode_input_positions[c128_mask]
        c128_target_shape = (min(self.num_decode_tokens, self.num_decode_tokens // 128 + self.num_decodes),)
        pad_right = c128_target_shape[0] - c128_input_positions.shape[0]
        c128_pad_positions = F.pad(c128_input_positions, (0, pad_right), value=0.0)
        # c128_cos, c128_sin = get_cos_and_sin_dsa(c128_pad_positions)
        c128_cos, c128_sin = get_cos_and_sin_dsa({"c128": c128_pad_positions},use_cache=True)

        max_seqlen_kv = torch.max(query_start_loc).item()
        max_seqlen_q = torch.max(self.seq_lens[:self.num_decodes]).item()

        decode_metadata = AscendDSADecodeMetadata(
            input_positions=input_positions,
            block_table=block_table,
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
            state_block_table=common_attn_metadata.state_block_table[:self.num_decodes],
            sin=sin[:self.num_decode_tokens, ...],
            cos=cos[:self.num_decode_tokens, ...],
            c4_sin=c4_sin,
            c4_cos=c4_cos,
            c128_sin=c128_sin,
            c128_cos=c128_cos,
            cp_seq_len=cp_seq_len,
            batch_seq_mask=batch_seq_mask)
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
    ):
        if attn_state in {
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding
        }:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
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

        # ascend_config = get_ascend_config()
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
        # self.npu_attention_post_func = AttentionPostV4()


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
        dtype= x.dtype
        if inverse:
            sin = sin * -1
        tnd_layout = 1
        if len(x.shape)==3:
            num_tokens,num_heads,rotary_dim = x.shape
        else:
            tnd_layout=0
            _,num_tokens,num_heads,rotary_dim = x.shape
        # print(f'cos.shape: {cos.shape}, x.shape: {x.shape}')
        x_rot = torch_npu.npu_rotary_mul(x.reshape(num_tokens, num_heads, 1, rotary_dim).to(torch.float32), cos, sin, rotary_mode="interleave")
        if tnd_layout:
            x = x_rot.reshape(num_tokens, -1, rotary_dim)
        else:
            x = x_rot.reshape(1,num_tokens, -1, rotary_dim)
        return x.to(dtype)
    
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
                              self.num_heads * self.head_dim)
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
            o_proj_input[decode_tokens:actual_tokens] = output_prefill.view(-1, output_prefill.shape[1] * output_prefill.shape[2])
            cos = attn_metadata.prefill.cos[layer_name]
            sin = attn_metadata.prefill.sin[layer_name]

        if has_decode:
            output_decode = self._forward_decode(
                layer_name,
                decode_hidden_states,
                kv_cache,
                attn_metadata,
                kv_state)
            o_proj_input[:decode_tokens] = output_decode.view(-1, output_decode.shape[1] * output_decode.shape[2])
            cos = attn_metadata.decode.cos[layer_name]
            sin = attn_metadata.decode.sin[layer_name]

        # zyl remove
        # o_proj_input = o_proj_input.view(-1, 64, 512)
        # cos = cos.view(-1,cos.shape[-1])
        # sin = sin.view(-1,sin.shape[-1])
        # wo_a = self.wo_a.weight.reshape(64,self.o_lora_rank,-1)
        # wo_b = self.wo_b.weight.reshape(8192,4096)
        # attn post
        # print(f'****************************cos = {cos.shape}')
        # output[...] = self.npu_attention_post_func(o_proj_input, cos, sin, wo_a, wo_b)
        cos = attn_metadata.cos[layer_name]
        sin = attn_metadata.sin[layer_name]
        o = o_proj_input.view(-1, 64, 512)
        num_tokens = o.shape[0]
        o_nope, o_pe = o.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        o_pe = self.rope_single(o_pe, cos, sin, True)
        o = torch.cat([o_nope, o_pe], dim=-1)

        # o
        o = o.view(num_tokens, self.n_local_groups, -1)
        # print(f"================={self.n_local_groups=}")
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("tgd,grd->tgr", o, wo_a)
        # print(f"attn Inner einsum {o=}")
        o = o.reshape(num_tokens, -1)
        output[...] = self.wo_b(o)
        # print(f"attn Inner einsum {output[:o.shape[0]]=}")

        return output_padded
    
    def _forward_prefill(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: Tuple,
        attn_metadata: AscendDSAMetadata,
        kv_state: Tuple,
    ):
        # if True:
        #     return torch.rand(hidden_states.shape[0], 32768,
        #                       dtype=hidden_states.dtype,
        #                       device=hidden_states.device)
        if self.compress_ratio==1:
            (sliding_window_state) = kv_state
        elif self.compress_ratio==4:
            (sliding_window_state, compressor_kv_state, compressor_score_state,
             c4_indexer_kv_state, c4_indexer_score_state) = kv_state
        elif self.compress_ratio==128:
            (sliding_window_state, compressor_kv_state, compressor_score_state) = kv_state

        # if torch.distributed.get_rank() == 0 and '.0' in layer_name:
        #     logger.info(f'>>>>> mla fwd, layer_name={layer_name}, hidden_states={hidden_states.shape}, state_ids={state_ids}, kv_state={kv_state.shape}')
        # forward_context = get_forward_context()
        # cos = attn_metadata.prefill.cos
        # sin = attn_metadata.prefill.sin
        cos = attn_metadata.prefill.cos[layer_name]
        sin = attn_metadata.prefill.sin[layer_name]
        actual_seq_lengths_query = attn_metadata.prefill.query_start_loc
        actual_seq_lengths_key = attn_metadata.prefill.seq_lens
        num_decode_tokens = attn_metadata.num_decode_tokens
        max_seqlen_kv =  torch.max(actual_seq_lengths_key).item()
        seq_lens_q = actual_seq_lengths_query[1:] - actual_seq_lengths_query[:-1]
        max_seqlen_q = torch.max(seq_lens_q).item()
        compressed_kv_block_table = attn_metadata.prefill.block_table
        compressed_kv_slot_mapping = attn_metadata.prefill.slot_mapping

        # mlaprolog
        # q
        qr = self.q_norm(self.wq_a(hidden_states)) # bs
        # print(f"attn Inner {qr=}, {qr.mean()=}")
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim)) # tp
        # print(f"attn Inner {q=}, {q.mean()=}")
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        # print(f"attn Inner rsqrt {q=}, {q.mean()=}")
        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = self.rope_single(q_pe, cos, sin)
        q = torch.cat([q_nope, q_pe], dim=-1)
        # print(f"attn Inner {q=}, {q.mean()=}")
        # win kv & tok_dis
        kv = self.wkv(hidden_states)
        kv = self.kv_norm(kv)
        kv = kv.view(-1, 1, self.nope_head_dim+self.rope_head_dim)
        kv_nope, kv_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        kv_pe = self.rope_single(kv_pe, cos, sin)
        kv = torch.cat([kv_nope, kv_pe], dim=-1)
        # print(f"attn Inner {kv=}")

        # swa exec kv
        torch_npu.npu_scatter_nd_update_(
            kv_state[0].view(-1, kv.shape[-1]), 
            attn_metadata.prefill.swa_slot_mapping.unsqueeze(-1),
            kv
        )

        # topk_idxs = self.get_window_topk_idxs(self.win, bsz, seqlen, start_pos) # ignorn
        if self.compress_ratio > 1:
            if self.compress_ratio == 4:
                # slot_mapping = attn_metadata.slot_mapping_list[0][num_decode_tokens:]
                compress_cos = attn_metadata.prefill.c4_cos[layer_name]
                compress_sin = attn_metadata.prefill.c4_sin[layer_name]
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
            elif self.compress_ratio == 128:
                # slot_mapping = attn_metadata.slot_mapping_list[1][num_decode_tokens:]
                compress_topk_idxs = None
                compress_cos = attn_metadata.prefill.c128_cos[layer_name]
                compress_sin = attn_metadata.prefill.c128_sin[layer_name]

            coff = 2 if self.compressor_overlap else 1
            start_pos = actual_seq_lengths_key - seq_lens_q
            # compressor
            compressed_kv = torch.ops.custom.npu_compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                compressor_kv_state,
                compressor_score_state,
                self.compressor_ape,
                self.compressor_norm.weight, 
                compress_sin.view(-1, compress_sin.shape[-1]).to(torch.bfloat16),
                compress_cos.view(-1, compress_cos.shape[-1]).to(torch.bfloat16),
                kv_block_table = attn_metadata.prefill.state_block_table,
                score_block_table = attn_metadata.prefill.state_block_table,
                cu_seqlens = actual_seq_lengths_query,
                seqused = None, #actual_seq_lengths_key,
                start_pos = start_pos,
                rope_head_dim = self.rope_head_dim,
                cmp_ratio = self.compress_ratio,
                coff = coff,
                norm_eps = self.compressor_norm_eps,
                rotary_mode = 2
            )

            if compressed_kv.numel() == 0:
                compressed_kv = None
            # elif self.compressor.rotate:
            #     compressed_kv = rotate_activation(compressed_kv)
            # print(f"prefill {compressed_kv=}, {self.compress_ratio}, {coff=}, {start_pos=}")

            # kv_compress_epilog
            torch_npu.npu_scatter_nd_update_(
                            kv_cache[0].view(-1, compressed_kv.shape[-1]), 
                            compressed_kv_slot_mapping.unsqueeze(-1),
                            compressed_kv.view(-1, compressed_kv.shape[-1]))
            # kv_compress_epilog(
            #     compressed_kv, 
            #     slot_mapping, 
            #     kv_compress_cache,
            #     quant_group_size
            #     )
        
            # compress_out_shape = (kv.shape[0], self.head_dim)
            # compress_out = torch.empty(compress_out_shape,
            #                        dtype=kv.dtype,
            #                        device=kv.device)
            # compress_kernal(kv, self.wkv, self.wgate, kv_state[1], kv_state[2], self.ape, 
            #                 self.kv_norm, self.compress_sin, self.compress_cos, state_ids, 
            #                 state_ids, actual_seq_lengths_query, actual_seq_lengths_key,
            #                 start_pos, self.rope_head_dim, self.compress_ratio, self.overlap+1,
            #                 self.eps, compress_out)
        
            # kv_compress_epilog_kernal(compress_out, slot_mapping, kv_cache[0])

        # attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(
        #     query=ql_nope,
        #     key=kv_cache[0],
        #     value=kv_cache[0],
        #     sparse_indices=topk_indices,
        #     scale_value=self.scale,
        #     sparse_block_size=1,
        #     block_table=attn_metadata.block_tables,
        #     actual_seq_lengths_query=actual_seq_lengths_query,
        #     actual_seq_lengths_kv=actual_seq_lengths_key,
        #     query_rope=q_pe,
        #     key_rope=kv_cache[1],
        #     layout_query="TND",
        #     layout_kv="PA_BSND",
        #     sparse_mode=3,
        # )

        sliding_window_kv_padded = pad_to_blocks(kv, actual_seq_lengths_key, block_size=128)
        # sliding_window_kv_padded = torch.cat([sliding_window_kv_padded, sliding_window_kv_padded], dim=0)
        # print(f"prefill {q=}, {q.shape=} {sliding_window_kv_padded=} {sliding_window_kv_padded.shape=}, {attn_metadata.prefill.block_table=}, {self.attn_sink=}")
        # print(f"prefill {max_seqlen_q=} {max_seqlen_kv=}, {actual_seq_lengths_key=}, {actual_seq_lengths_query=}, {self.num_heads=}, {self.head_dim=}, {sliding_window_state=}")
        if self.compress_ratio == 1:
            metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                head_dim=self.head_dim,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(actual_seq_lengths_key),
                ori_mask_mode=4, # 4:sliding window
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=(sliding_window_state != None),
                has_cmp_kv=(self.compress_ratio != 1)
            )
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=sliding_window_kv_padded,
                ori_block_table=attn_metadata.prefill.prefill_swa_block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=metadata,
                softmax_scale=self.softmax_scale,
                ori_mask_mode=4, # 4:sliding window
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND"
            )
            # print(f"metadata {metadata=}, {metadata.mean()=}")
        elif self.compress_ratio == 4:
            metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                head_dim=self.head_dim,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(actual_seq_lengths_key),
                topk=self.index_topk, #
                cmp_ratio=self.compress_ratio, #
                ori_mask_mode=4, # 4:sliding window
                cmp_mask_mode=3, # 3:causal
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=(sliding_window_state != None),
                has_cmp_kv=(self.compress_ratio != 1)
            )
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=sliding_window_kv_padded,
                cmp_kv=kv_cache[0],
                cmp_sparse_indices=compress_topk_idxs, # B T D 
                ori_block_table=attn_metadata.prefill.prefill_swa_block_table,
                cmp_block_table=compressed_kv_block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=metadata,
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
            metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                head_dim=self.head_dim,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(actual_seq_lengths_key),
                cmp_ratio=self.compress_ratio, #
                ori_mask_mode=4, # 4:sliding window
                cmp_mask_mode=3, # 3:causal
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=(sliding_window_state != None),
                has_cmp_kv=(self.compress_ratio != 1)
            )
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=sliding_window_kv_padded,
                cmp_kv=kv_cache[0],
                ori_block_table=attn_metadata.prefill.prefill_swa_block_table,
                cmp_block_table=compressed_kv_block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio, #
                ori_mask_mode=4, # 4:sliding window
                cmp_mask_mode=3, # 3:causal
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND"
            )

        # torch.save(q, "/home/w00879359/q_test.pt")
        # torch.save(sliding_window_kv_padded, "/home/w00879359/sliding_window_kv_padded_test.pt")
        # exit()
        # attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
        #     q,
        #     ori_kv=sliding_window_kv_padded,
        #     cmp_kv=None if self.compress_ratio == 1 else kv_cache[0],
        #     cmp_sparse_indices= None if self.compress_ratio == 1 else compress_topk_idxs,
        #     ori_block_table=attn_metadata.prefill.block_table,
        #     cmp_block_table=None if self.compress_ratio == 1 else compressed_kv_block_table,
        #     cu_seqlens_q=actual_seq_lengths_query,
        #     seqused_kv=actual_seq_lengths_key,
        #     sinks=self.attn_sink,
        #     metadata=metadata,
        #     softmax_scale=self.softmax_scale,
        #     cmp_ratio=self.compress_ratio, #
        #     ori_mask_mode=4, # 4:sliding window
        #     cmp_mask_mode=3, # 3:causal
        #     ori_win_left=self.window_size - 1,
        #     ori_win_right=0,
        #     layout_q="TND",
        #     layout_kv="PA_ND"
        # )

        # print(f"attn_output {attn_output=}, {attn_output.shape=}, {attn_output.mean()=}")
        return attn_output

    def _forward_decode(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: Tuple,
        attn_metadata: AscendDSAMetadata,
        kv_state: Tuple,
    ):
        # if True:
        #     return torch.rand(hidden_states.shape[0], 32768,
        #                       dtype=hidden_states.dtype,
        #                       device=hidden_states.device)
        if self.compress_ratio==1:
            (sliding_window_state) = kv_state
        elif self.compress_ratio==4:
            (sliding_window_state, compressor_kv_state, compressor_score_state, c4_indexer_kv_state, c4_indexer_score_state) = kv_state
        elif self.compress_ratio==128:
            (sliding_window_state, compressor_kv_state, compressor_score_state) = kv_state

        # if torch.distributed.get_rank() == 0 and '.0' in layer_name:
        #     logger.info(f'>>>>> mla fwd, layer_name={layer_name}, hidden_states={hidden_states.shape}, state_ids={state_ids}, kv_state={kv_state.shape}')
        # forward_context = get_forward_context()
        cos = attn_metadata.decode.cos[layer_name]
        sin = attn_metadata.decode.sin[layer_name]
        actual_seq_lengths_query = attn_metadata.decode.query_start_loc
        actual_seq_lengths_key = attn_metadata.decode.seq_lens
        num_decode_tokens = attn_metadata.num_decode_tokens
        max_seqlen_kv = attn_metadata.decode.max_seqlen_kv
        seq_lens_q = actual_seq_lengths_query[1:] - actual_seq_lengths_query[:-1]
        max_seqlen_q = attn_metadata.decode.max_seqlen_q
        compressed_kv_block_table = attn_metadata.decode.block_table
        compressed_kv_slot_mapping = attn_metadata.decode.slot_mapping

        # q
        qr = q = self.wq_a(hidden_states) # bs
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim)) # tp
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
        torch_npu.npu_scatter_nd_update_(
            kv_state[0].view(-1, kv.shape[-1]), 
            attn_metadata.decode.swa_slot_mapping.unsqueeze(-1),
            kv
        )

        if self.compress_ratio > 1:
            if self.compress_ratio == 4:
                # slot_mapping = attn_metadata.decode.slot_mapping_list[0][:num_decode_tokens]
                compress_cos = attn_metadata.decode.c4_cos[layer_name]
                compress_sin = attn_metadata.decode.c4_sin[layer_name]
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
                
            elif self.compress_ratio == 128:
                # slot_mapping = attn_metadata.decode.slot_mapping_list[1][:num_decode_tokens]
                compress_topk_idxs = None
                compress_cos = attn_metadata.decode.c128_cos[layer_name]
                compress_sin = attn_metadata.decode.c128_sin[layer_name]

            coff = 2 if self.compressor_overlap else 1
            start_pos = actual_seq_lengths_key - seq_lens_q
            # compressor
            compressed_kv = torch.ops.custom.npu_compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                compressor_kv_state,
                compressor_score_state,
                self.compressor_ape,
                self.compressor_norm.weight, 
                compress_sin.view(-1, compress_sin.shape[-1]).to(torch.bfloat16),
                compress_cos.view(-1, compress_cos.shape[-1]).to(torch.bfloat16),
                kv_block_table = attn_metadata.decode.state_block_table,
                score_block_table = attn_metadata.decode.state_block_table,
                cu_seqlens = actual_seq_lengths_query,
                seqused = None, #actual_seq_lengths_key,
                start_pos = start_pos,
                rope_head_dim = self.rope_head_dim,
                cmp_ratio = self.compress_ratio,
                coff = coff,
                norm_eps = self.compressor_norm_eps,
                rotary_mode = 2
            )
            # kv_compress_epilog
            torch_npu.npu_scatter_nd_update_(
                            kv_cache[0].view(-1, compressed_kv.shape[-1]), 
                            compressed_kv_slot_mapping.unsqueeze(-1),
                            compressed_kv.view(-1, compressed_kv.shape[-1]))
        
            # compress_kv = compress_kernal(kv, self.wkv, self.wgate, kv_state[1], kv_state[2], self.ape, 
            #                 self.kv_norm, self.compress_sin, self.compress_cos, state_ids, 
            #                 state_ids, actual_seq_lengths_query, actual_seq_lengths_key,
            #                 start_pos, self.rope_head_dim, self.compress_ratio, self.overlap+1,
            #                 self.eps)
            # kv_compress_epilog_kernal(compress_kv, slot_mapping, kv_cache[0])

        # attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(
        #     query=ql_nope,
        #     key=kv_cache[0],
        #     value=kv_cache[0],
        #     sparse_indices=topk_indices,
        #     scale_value=self.scale,
        #     sparse_block_size=1,
        #     block_table=attn_metadata.block_tables,
        #     actual_seq_lengths_query=actual_seq_lengths_query,
        #     actual_seq_lengths_kv=actual_seq_lengths_key,
        #     query_rope=q_pe,
        #     key_rope=kv_cache[1],
        #     layout_query="TND",
        #     layout_kv="PA_BSND",
        #     sparse_mode=3,
        # )

        # print(f"=====================> {actual_seq_lengths_query=}, {actual_seq_lengths_key=}, {max_seqlen_q=}, {max_seqlen_kv=}")
        if self.compress_ratio == 1:
            metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                head_dim=self.head_dim,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(actual_seq_lengths_key),
                ori_mask_mode=4, # 4:sliding window
                cmp_mask_mode=3, # 3:causal
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=(sliding_window_state != None),
                has_cmp_kv=(self.compress_ratio != 1)
            )
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=kv_state[0].unsqueeze(2),
                ori_block_table=attn_metadata.decode.state_block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=metadata,
                softmax_scale=self.softmax_scale,
                ori_mask_mode=4, # 4:sliding window
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND"
            )
        elif self.compress_ratio == 4:
            metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                head_dim=self.head_dim,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(actual_seq_lengths_key),
                topk=self.index_topk, #
                cmp_ratio=self.compress_ratio, #
                ori_mask_mode=4, # 4:sliding window
                cmp_mask_mode=3, # 3:causal
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=(sliding_window_state != None),
                has_cmp_kv=(self.compress_ratio != 1)
            )
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=kv_state[0].unsqueeze(2),
                cmp_kv=kv_cache[0],
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=attn_metadata.decode.state_block_table,
                cmp_block_table=compressed_kv_block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=metadata,
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
            metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                head_dim=self.head_dim,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(actual_seq_lengths_key),
                cmp_ratio=self.compress_ratio, #
                ori_mask_mode=4, # 4:sliding window
                cmp_mask_mode=3, # 3:causal
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=(sliding_window_state != None),
                has_cmp_kv=(self.compress_ratio != 1)
            )
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=kv_state[0].unsqueeze(2),
                cmp_kv=kv_cache[0],
                ori_block_table=attn_metadata.decode.state_block_table,
                cmp_block_table=compressed_kv_block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=metadata,
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

    def _forward_single_op_prefill(
        self,
        hidden_states,
        kv_cache,
        attn_metadata,
        kv_state,
        layer_name
    ):     
        cos = attn_metadata.prefill.cos[layer_name]
        sin = attn_metadata.prefill.sin[layer_name]
        
        seqlen, _ = hidden_states.size()
        bsz = 1
        start_pos = 0
        win = self.window_size
        ratio = self.compress_ratio
        # q
        x = hidden_states
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q_dtype = q.dtype
        q *= torch.rsqrt(q.to(torch.float32).square().mean(-1, keepdim=True) + self.eps)
        q = q.to(q_dtype)
        
        
        # qq = q.clone()
        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = self.rope_single(q_pe, cos, sin)
        q = torch.cat([q_nope, q_pe], dim=-1)
        

        # win kv & topk_idxs
        kv = self.wkv(x)
        # print(f'======================kv.shape: {kv.shape} weights.shape: {self.wkv.weight.shape}')
        kv = self.kv_norm(kv)
        kv = kv.view(-1, 1, self.nope_head_dim+self.rope_head_dim) # 5 512
        
        
        
        kv_nope, kv_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        kv_pe = self.rope_single(kv_pe, cos, sin)
        kv = torch.cat([kv_nope, kv_pe], dim=-1)
        

        # print(f'kv max_abs:{max_abs}, rmse:{rmse}, rel_l2:{rel_l2}, cossim:{cossim}')
        torch_npu.npu_scatter_nd_update_(
                        kv_state[0].view(-1, kv.shape[-1]), attn_metadata.prefill.swa_slot_mapping.unsqueeze(-1),
                        kv)
        
        # print(f"=====================================in attention kv rank : {torch.distributed.get_rank()}, kv is {kv}, mean is {kv.mean()}")
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(kv.device)
        # print(f"=====================================in attention topkidx rank : {torch.distributed.get_rank()}, topkidx is {topk_idxs}")
        if self.compress_ratio > 1:
            offset = kv.size(0) if start_pos == 0 else win # TODO zyl
            if self.indexer is not None:
                compress_topk_idxs = self.indexer_select_single_op(x, qr, cos, sin,kv_cache, kv_state,attn_metadata,offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset).to(kv.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()
        
        # compress kv & attn
        if start_pos == 0:
            # self.kv_cache[:bsz, :min(win, seqlen)] = kv[:, -win:]
            if self.compress_ratio > 1:

                if (kv_compress := self.compressor(x, start_pos, cos, sin, kv_state)) is not None:
                    # print(f'kv.shape: {kv.shape}, kv_compress:{kv_compress.shape}')                    if kv_compress.shape[1]:  # bsnd
                        if self.compress_ratio ==4 and kv_compress.numel()!=0:
                            torch_npu.npu_scatter_nd_update_(
                            kv_cache[0].view(-1, kv_compress.shape[-1]), attn_metadata.prefill.slot_mapping_list[0].unsqueeze(-1),
                            kv_compress.squeeze(0))
                        elif self.compress_ratio ==128 and kv_compress.numel()!=0:
                            torch_npu.npu_scatter_nd_update_(
                            kv_cache[0].view(-1, kv_compress.shape[-1]), attn_metadata.prefill.slot_mapping_list[1].unsqueeze(-1),
                            kv_compress.squeeze(0))

                        kv = torch.cat([kv, kv_compress.squeeze(0)], dim=0)
                    
            # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
            o = sparse_attn_torch(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)

        o_nope, o_pe = o.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        o_pe = self.rope_single(o_pe, cos, sin, True)
        o = torch.cat([o_nope, o_pe], dim=-1)

        # o
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        o=o.flatten(2).squeeze(0)
        x = self.wo_b(o)
        
        return x

    def indexer_select(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        kv_state: Tuple,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        need_gather_q_kv: bool = False,
    ):
        start_pos = 0 # (wy): start_pos=0
        seqlen, _ = x.size()
        bsz = 1
        rd = self.rope_head_dim
        (_, _, _,
             c4_indexer_kv_state, c4_indexer_score_state) = kv_state
        # q process in new stream
        q, _ = self.wq_b(qr)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
        q = q.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]

        q_nope, q_pe = q.split([self.indexcom_head_dim-self.rope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = self.rope_single(q_pe, cos, sin)
        q = torch.cat([q_nope, q_pe], dim=-1)

        seq_lens_q = actual_seq_lengths_query[1:] - actual_seq_lengths_query[:-1]

        c4_indexer_kv_state = torch.zeros(2,128,c4_indexer_kv_state.shape[-1]).to(torch.float32).to("npu")
        c4_indexer_score_state = torch.zeros(2,128,256).to(torch.float32).to("npu")
        state_ids = attn_metadata.state_ids.view(1, 1)
        coff = 2 if self.compressor_overlap else 1
        start_pos = actual_seq_lengths_key - seq_lens_q
        # print(f"================> indexer_select_qli:  {x.shape=},\n {start_pos=},\n {c4_indexer_kv_state.shape=},\n {state_ids=},\n {actual_seq_lengths_query=},\n {actual_seq_lengths_key=}")
        # print(f"================> indexer_select_qli:  {self.indexcom_ape.shape=},\n {self.rope_head_dim=},\n {self.compress_ratio=},\n {coff=},\n {self.compressor_norm_eps=},\n {self.indexcom_wgate.weight.shape=}")
        # k = self.compress_forward(x, kv_cache)
        k = torch.ops.custom.npu_compressor(
            x,
            self.indexcom_wkv.weight,
            self.indexcom_wgate.weight,
            c4_indexer_kv_state,
            c4_indexer_score_state,
            self.indexcom_ape,
            self.indexcom_norm.weight, 
            sin.view(-1, sin.shape[-1]),
            cos.view(-1, cos.shape[-1]),
            kv_block_table = attn_metadata.state_ids,
            score_block_table = attn_metadata.state_ids,
            cu_seqlens = actual_seq_lengths_query,
            seqused = actual_seq_lengths_key,
            start_pos = actual_seq_lengths_key - seq_lens_q,
            rope_head_dim = self.rope_head_dim,
            cmp_ratio = self.compress_ratio,
            coff = 0 if self.compressor_overlap else 1,
            norm_eps = self.compressor_norm_eps,
            rotary_mode = 2
        )

        weights, _ = self.weights_proj(x)

        block_table = attn_metadata.block_tables

        torch.ops._C_ascend.indexer_compress_epilog(k, attn_metadata.slot_mapping, kv_cache[1], kv_cache[2])

        soc_version = get_ascend_device_type()
        dst_type = torch.float8_e4m3fn if soc_version in {AscendDeviceType.A5} else torch.int8

        q_shape = q.shape

        q, q_scale = torch_npu.npu_dynamic_quant(q.view(-1, self.head_dim), dst_type=dst_type)

        if soc_version not in {AscendDeviceType.A5}:
            q_scale = q_scale.to(torch.float16)

        sparse_indices, _ = torch.ops._C_ascend.npu_lightning_indexer(
            query=q.view(q_shape),
            key=kv_cache[1],
            weights=weights,
            query_dequant_scale=q_scale.view(q_shape[:-1]),
            key_dequant_scale=kv_cache[2],
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            query_quant_mode=0,
            key_quant_mode=0,
            sparse_count=512,
            sparse_mode=3,
            cmp_ratio=4,
            layout_query="TND",
            layout_key="PA_BSND",
        )
        return sparse_indices

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

        q = rotate_activation(q)
        
        # c4_indexer_kv_state = torch.zeros(2,128,256).to(torch.float32).to("npu")
        # c4_indexer_score_state = torch.zeros(2,128,256).to(torch.float32).to("npu")
        # state_ids = attn_metadata.state_ids.view(1, 1)
        coff = 2 if self.compressor_overlap else 1
        start_pos = actual_seq_lengths_key - seq_lens_q
        
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            kv_block_table = attn_metadata.prefill.state_block_table
            score_block_table = attn_metadata.prefill.state_block_table
        else:
            kv_block_table = attn_metadata.decode.state_block_table
            score_block_table = attn_metadata.decode.state_block_table

        kv = torch.ops.custom.npu_compressor(
            x,
            self.indexcom_wkv.weight,
            self.indexcom_wgate.weight,
            c4_indexer_kv_state,
            c4_indexer_score_state,
            self.indexcom_ape,
            self.indexcom_norm.weight,
            compressed_sin.view(-1, compressed_sin.shape[-1]).to(torch.bfloat16),
            compressed_cos.view(-1, compressed_cos.shape[-1]).to(torch.bfloat16),
            kv_block_table = kv_block_table,
            score_block_table = score_block_table,
            cu_seqlens = actual_seq_lengths_query.to(c4_indexer_kv_state.device),
            seqused = None, #actual_seq_lengths_key,
            start_pos = start_pos,
            rope_head_dim = self.rope_head_dim,
            cmp_ratio = self.compress_ratio,
            coff = coff,
            norm_eps = self.compressor_norm_eps,
            rotary_mode = 2
        )

        if kv.numel() == 0:
            kv = None
        elif self.indexer.compressor.rotate:
            kv = rotate_activation(kv)

        # print(f"indexer kvcache info:\n{kv_cache[1].shape=}\n{kv_cache[1].dtype=}\n{kv_cache[2].shape=}\n{kv_cache[2].dtype=}\n{attn_metadata.prefill.slot_mapping_list=}\n{attn_metadata.prefill.block_table_list=}")

        # if kv is not None:
        #     torch_npu.npu_scatter_nd_update_(
        #                     kv_cache[1].view(-1, kv.shape[-1]), attn_metadata.prefill.slot_mapping_list[0].unsqueeze(-1),
        #                     kv.view(kv.shape[1], kv.shape[-1]))
        weights = self.weights_proj(x) * (self.indexer_softmax_scale * self.indexcom_head_dim ** -0.5)

        soc_version = get_ascend_device_type()
        dst_type = torch.float8_e4m3fn if soc_version in {AscendDeviceType.A5} else torch.int8

        # if kv is not None:
        #     print(f'q kv in indexer: {q.shape=}, {kv.shape=}, {weights.shape=}')

        q, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=dst_type)
        if kv is not None:
            kv, kv_scale = torch_npu.npu_dynamic_quant(kv, dst_type=dst_type)

            kv_scale = kv_scale.unsqueeze(-1)

        if soc_version not in {AscendDeviceType.A5}:
            q_scale = q_scale.to(torch.float16)
            if kv is not None:
                kv_scale = kv_scale.to(torch.float16)
                kv_scale = kv_scale.unsqueeze(-1)

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            # print(f"indexer prefill kvcache info:\n{kv_cache[1].shape=}\n{kv_cache[1].dtype=}\n{kv_cache[2].shape=}\n{kv_cache[2].dtype=}\n{attn_metadata.prefill.slot_mapping_list=}\n{attn_metadata.prefill.block_table_list=}")
            # print(f"input info:\n{attn_metadata.prefill.seq_lens=}\n{attn_metadata.prefill.query_lens=}\n{attn_metadata.prefill.query_start_loc=}")
            if kv is not None:
                torch_npu.npu_scatter_nd_update_(
                            kv_cache[1].view(-1, kv.shape[-1]), attn_metadata.prefill.slot_mapping.unsqueeze(-1),
                            kv.view(-1, kv.shape[-1]))
                torch_npu.npu_scatter_nd_update_(
                            kv_cache[2].view(-1, kv_scale.shape[-1]), attn_metadata.prefill.slot_mapping.unsqueeze(-1),
                            kv_scale.view(-1, kv_scale.shape[-1]))
        else:
            # print(f"indexer decode kvcache info:\n{kv_cache[1].shape=}\n{kv_cache[1].dtype=}\n{kv_cache[2].shape=}\n{kv_cache[2].dtype=}\n{attn_metadata.decode.slot_mapping_list=}\n{attn_metadata.decode.block_table_list=}")
            # print(f"input info:\n{attn_metadata.decode.seq_lens=}\n{attn_metadata.decode.query_start_loc=}")
            if kv is not None:
                torch_npu.npu_scatter_nd_update_(
                            kv_cache[1].view(-1, kv.shape[-1]), attn_metadata.decode.slot_mapping.unsqueeze(-1),
                            kv.view(-1, kv.shape[-1]))
                torch_npu.npu_scatter_nd_update_(
                            kv_cache[2].view(-1, kv_scale.shape[-1]), attn_metadata.decode.slot_mapping.unsqueeze(-1),
                            kv_scale.view(-1, kv_scale.shape[-1]))

        metadata = torch.zeros((2048), dtype = torch.int32)
        usedCoreNum = 24 
        mBaseSize = 256
        s2BaseSize = 2048
        metadata[:3] = torch.tensor([usedCoreNum, mBaseSize, s2BaseSize], dtype=torch.int32)
        metadata[3:27] = torch.zeros((usedCoreNum), dtype = torch.int32) #bN2End 每个核处理数据的BN2结束点
        metadata[27:51] = torch.zeros((usedCoreNum), dtype = torch.int32) #mEnd 每个核处理数据的M结束点
        metadata[51:75] = torch.zeros((usedCoreNum), dtype = torch.int32) #s2End 每个核处理数据的S2结束点
        metadata = metadata.to(q.device)

        # if kv is not None:
        #     print(f"input info:\n{kv.shape=}\n{kv_scale.shape=}")
        #     print(f"input info:\n{kv.dtype=}\n{kv_scale.dtype=}")
        # print(f"input info:\n{q.shape=}\n{weights.shape=}\n{q_scale.shape=}\n{torch.Tensor([q.shape[1]])=}")
        # print(f"input info:\n{q.dtype=}\n{weights.dtype=}\n{q_scale.dtype=}\n{torch.Tensor([q.shape[1]])=}")
        # print(f"input info:\n{attn_metadata.prefill.seq_lens=}\n{attn_metadata.prefill.query_lens=}\n{attn_metadata.prefill.query_start_loc=}")
        # print(f"input info:\n{attn_metadata.decode.seq_lens=}\n{attn_metadata.decode.query_lens=}\n{attn_metadata.decode.query_start_loc=}")

        if with_prefill:
            qlens = attn_metadata.prefill.query_start_loc[1:].to(q.device)
            kvlens = attn_metadata.prefill.seq_lens
            block_table = attn_metadata.prefill.block_table
        else:
            qlens = attn_metadata.decode.query_start_loc[1:].to(q.device)
            kvlens = attn_metadata.decode.seq_lens
            block_table = attn_metadata.decode.block_table

        topk_idxs, _ = torch.ops.custom.npu_quant_lightning_indexer(
            query=q,
            key=kv_cache[1],
            weights=weights.to(torch.float16),
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
        # return None
        # topk_idxs [b, s, N, k]
        # topk_idxs = topk_idxs[..., :math.floor(kvlens.sum() / 4)].unsqueeze(0).squeeze(2)  # TODO 多序列需要修改
        # mask = topk_idxs == -1
        # topk_idxs = torch.where(mask, -1, topk_idxs)
        # if start_pos == 0:
        #     mask = topk_idxs >= torch.arange(1, seqlen + 1,device=q.device).unsqueeze(1) // ratio
        return topk_idxs
    
    def indexer_select_single_op(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache,
        kv_state,
        attn_metadata,
        offset=0, # (wy): start_pos=0
    ):
        start_pos = 0 # (wy): start_pos=0
        seqlen, _ = x.size()
        bsz = 1
        ratio = self.compress_ratio
        end_pos = start_pos + seqlen
        q = self.inderxer_wq_b(qr)
        # print(f'qr.shape in indexer: {qr.shape}, q.shape : {q.shape}, self.wq_b.shape: {self.wq_b.weight.shape}')
        q = q.view(bsz, seqlen, self.indexer_heads, self.indexcom_head_dim)
        ## rope

        q_nope, q_pe = q.split([self.indexcom_head_dim-self.rope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = self.rope_single(q_pe, cos, sin)
        q = torch.cat([q_nope, q_pe], dim=-1)
        

        q = rotate_activation(q)
        kv = self.indexer.compressor(x, start_pos, cos, sin, kv_state)
        if kv is not None:
            torch_npu.npu_scatter_nd_update_(
                            kv_cache[1].view(-1, kv.shape[-1]), attn_metadata.prefill.slot_mapping_list[0].unsqueeze(-1),
                            kv.view(kv.shape[1], kv.shape[-1]))
        weights = self.weights_proj(x) * (self.indexer_softmax_scale * self.indexcom_head_dim ** -0.5)
        # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
        # print(f'q.shape: {q.shape}, kv.shape: {kv.shape}')
        index_score = torch.einsum("bshd,btd->bsht", q, kv.squeeze(2))
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio,device=q.device).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1,device=q.device).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1,device=q.device).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs
    
    def _forward_single_op_decode(
        self,        
        hidden_states,
        kv_cache,
        attn_metadata,
        kv_state,
        layer_name):
        cos = attn_metadata.decode.cos[layer_name]
        sin = attn_metadata.decode.sin[layer_name]
        
        seqlen, _ = hidden_states.size()
        bsz = 1
        start_pos = int(attn_metadata.decode.seq_lens) - 1 
        end_pos = int(attn_metadata.decode.seq_lens)
        win = self.window_size
        ratio = self.compress_ratio
        x = hidden_states
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q_dtype = q.dtype
        q *= torch.rsqrt(q.to(torch.float32).square().mean(-1, keepdim=True) + self.eps)
        q = q.to(q_dtype)

        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = self.rope_single(q_pe, cos, sin)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv = kv.view(-1, 1, self.nope_head_dim+self.rope_head_dim) # 5 512

        kv_nope, kv_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        kv_pe = self.rope_single(kv_pe, cos, sin)
        kv = torch.cat([kv_nope, kv_pe], dim=-1)
        
        torch_npu.npu_scatter_nd_update_(
                        kv_state[0].view(-1, kv.shape[-1]), attn_metadata.decode.swa_slot_mapping.unsqueeze(-1),
                        kv)
        
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(kv.device)
        if self.compress_ratio > 1:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer_select_single_op_decode(x, qr, cos, sin,kv_cache, kv_state,attn_metadata,start_pos,offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset).to(kv.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()
        
        # compress kv & attn
        if start_pos == 0:
            # self.kv_cache[:bsz, :min(win, seqlen)] = kv[:, -win:]
            if self.compress_ratio > 1:

                if (kv_compress := self.compressor(x, start_pos, cos, sin, freqs_cis, kv_state)) is not None:
                    # print(f'kv.shape: {kv.shape}, kv_compress:{kv_compress.shape}')                    if kv_compress.shape[1]:  # bsnd
                        if self.compress_ratio ==4:
                            torch_npu.npu_scatter_nd_update_(
                            kv_cache[0].view(-1, kv_compress.shape[-1]), attn_metadata.prefill.slot_mapping_list[0].unsqueeze(-1),
                            kv_compress.squeeze(0))
                        elif self.compress_ratio ==128:
                            torch_npu.npu_scatter_nd_update_(
                            kv_cache[0].view(-1, kv_compress.shape[-1]), attn_metadata.prefill.slot_mapping_list[1].unsqueeze(-1),
                            kv_compress.squeeze(0))

                        kv = torch.cat([kv, kv_compress.squeeze(0)], dim=0)
                    
            # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
            o = sparse_attn_torch(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            swa_kv = kv_state[0][2][:128].unsqueeze(1)  # TODO
            if self.compress_ratio >1:
                if (kv_compress := self.compressor(x, start_pos, cos, sin, kv_state)) is not None:
                    if self.compress_ratio ==4 and kv_compress.numel()!=0:
                        torch_npu.npu_scatter_nd_update_(
                            kv_cache[0].view(-1, kv_compress.shape[-1]), attn_metadata.decode.slot_mapping_list[0][0].unsqueeze(-1),
                            kv_compress)
                    elif self.compress_ratio ==128 and kv_compress.numel()!=0:
                            torch_npu.npu_scatter_nd_update_(
                            kv_cache[0].view(-1, kv_compress.shape[-1]), attn_metadata.decode.slot_mapping_list[1][0].unsqueeze(-1),
                            kv_compress)

                if self.compress_ratio ==4:
                    compress_kv = kv_cache[0][1][:end_pos//self.compress_ratio]
                    swa_kv = torch.cat([swa_kv, compress_kv], dim=0)


            
            o = sparse_attn_torch(q, swa_kv, self.attn_sink, topk_idxs, self.softmax_scale)
            
            


        o_nope, o_pe = o.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        o_pe = self.rope_single(o_pe, cos, sin, True)
        o = torch.cat([o_nope, o_pe], dim=-1)

        # o
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        # print(f"=====================================in attention o2 rank : {torch.distributed.get_rank()}, o2 is {o}")
        o=o.flatten(2).squeeze(0)
        x = self.wo_b(o)
        # print(f"=====================================in attention attn out rank : {torch.distributed.get_rank()}, attn out is {x}")
        
        return x

    def indexer_select_single_op_decode(
            self,
            x: torch.Tensor,
            qr: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            kv_cache,
            kv_state,
            attn_metadata,
            start_pos,
            offset=0, # (wy): start_pos=0
        ):
            start_pos = start_pos # (wy): start_pos=0
            seqlen, _ = x.size()
            bsz = 1
            ratio = self.compress_ratio
            end_pos = start_pos + seqlen
            # if self.indexer.compressor.kv_cache is None:
            #     self.indexer.compressor.kv_cache = self.indexer.kv_cache
            q = self.inderxer_wq_b(qr)
            # print(f'qr.shape in indexer: {qr.shape}, q.shape : {q.shape}, self.wq_b.shape: {self.wq_b.weight.shape}')
            q = q.view(bsz, seqlen, self.indexer_heads, self.indexcom_head_dim)
            ## rope
            cos_q, sin_q = cos, sin
            cos_q = cos_q.view(1, 1, -1, self.rope_head_dim)
            sin_q = sin_q.view(1, 1, -1, self.rope_head_dim)

            q_nope, q_pe = q.split([self.indexcom_head_dim-self.rope_head_dim, self.rope_head_dim], dim=-1)
            q_pe = self.rope_single(q_pe, cos, sin)
            q = torch.cat([q_nope, q_pe], dim=-1)

            q = rotate_activation(q)
            kv = self.indexer.compressor(x, start_pos, cos_q, sin_q, kv_state)
            if kv is not None:
                torch_npu.npu_scatter_nd_update_(
                                kv_cache[1].view(-1, kv.shape[-1]), attn_metadata.decode.slot_mapping_list[0][0].unsqueeze(-1),
                                kv.view(kv.shape[1], kv.shape[-1]))
            weights = self.weights_proj(x) * (self.indexer_softmax_scale * self.indexcom_head_dim ** -0.5)
            # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
            # print(f'q.shape: {q.shape}, kv.shape: {kv.shape}')
            kv = kv_cache[1][1][:end_pos//4].unsqueeze(0).squeeze(2)  ## TODO
            index_score = torch.einsum("bshd,btd->bsht", q, kv)
            index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
            if start_pos == 0:
                mask = torch.arange(seqlen // ratio,device=q.device).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1,device=q.device).unsqueeze(1) // ratio
                index_score += torch.where(mask, float("-inf"), 0)
            topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
            if start_pos == 0:
                mask = topk_idxs >= torch.arange(1, seqlen + 1,device=q.device).unsqueeze(1) // ratio
                topk_idxs = torch.where(mask, -1, topk_idxs + offset)
            else:
                topk_idxs += offset
            return topk_idxs