from dataclasses import dataclass
from typing import (TYPE_CHECKING, ClassVar, NamedTuple, Optional, Tuple, Type,
                    TypeVar)

import numpy as np
import torch
import torch_npu
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
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.common_cp import (AscendPCPMetadata,
                                             CPChunkedContextMetadata)
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         enable_cp,
                                         maybe_save_kv_layer_to_connector,
                                         split_decodes_and_prefills,
                                         trans_rope_weight, transdata,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params, get_graph_params,
    update_draft_graph_params_workspaces, update_graph_params_workspaces)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.shared_weight_layer import (
    is_hidden_layer, post_process_after_loading_for_shared_weight_series,
    reach_layer_for_shared_weight_series,
    register_layer_to_shared_weight_series)
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND,
                               flashcomm2_o_shared_enabled, maybe_trans_nz,
                               weak_ref_tensors)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024


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
    def get_impl_cls() -> Type["DSAAttentionImpl"]:
        return AscendDSAImpl

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

    num_actual_tokens_pcp_padded: int
    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor

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
        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
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

    def pad_actual_seq_len_q_mtp_enable_pad(self, num_reqs_pad_size, num_reqs,
                                            actual_seq_lengths_q,
                                            common_attn_metadata):
        """
        Pads actual_seq_lengths_q evenly to not exceed 16 tokens per request
        in order to meet the requirement of npu_fused_infer_attention_score.

        In Torchair scenario, the lengths of the queries must be padded to the same length.
        And npu_fused_infer_attention_score constraint requires the last element must equal to batch_size(num_tokens).

        For example:
        batch_size=36, num_reqs_pad_size=2, num_reqs=16
        By default, each request should have inference 2 token, which means actual_seq_lengths_q should be
        [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36].

        However, mtp torchair + PD scenario, the actual_seq_lengths_q may be
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] before padding, since the first decode request only has 1 token.
        In order to meet the requirement of npu_fused_infer_attention_score, we need to pad actual_seq_lengths_q evenly to not exceed 16 tokens per request.
        after padding actual_seq_lengths_q should be similar to [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,36]
        """
        FIA_SEQ_LEN_LIMIT = 16
        need_padding = num_reqs_pad_size != 0 and \
                       len(common_attn_metadata.actual_seq_lengths_q) > num_reqs and \
                       common_attn_metadata.actual_seq_lengths_q[num_reqs] - actual_seq_lengths_q[
                           -1] > FIA_SEQ_LEN_LIMIT
        if need_padding:
            padding_seq_len_q = common_attn_metadata.actual_seq_lengths_q[
                num_reqs:num_reqs + num_reqs_pad_size]
            start_val = actual_seq_lengths_q[-1]
            end_val = padding_seq_len_q[-1]

            num_step = len(padding_seq_len_q)
            interpolated = np.round(
                np.linspace(start_val, end_val,
                            num_step + 1)[1:]).astype(int).tolist()
            assert interpolated[-1] == end_val
            assert len(interpolated) == len(padding_seq_len_q)
            actual_seq_lengths_q = actual_seq_lengths_q + interpolated
        else:
            actual_seq_lengths_q = actual_seq_lengths_q + common_attn_metadata.actual_seq_lengths_q[
                num_reqs:num_reqs + num_reqs_pad_size]

        return actual_seq_lengths_q

    def pad_actual_seq_len_q_mtp_disable_pad(self, num_reqs_pad_size, num_reqs,
                                             actual_seq_lengths_q):
        """
        Only use for acl full graph mode.
        Pad the last element of the actual_seq_lengths_q equal to the TND(T) and
        the num of dimensions equal to the batch_size of main model.

        For example:
        batch_size = 8, num_reqs = 4, num_speculative_tokens = 1
        input actual_seq_lengths_q = [1, 2, 4, 5]  (the 3rd req was accept a token)
        After padding the actual_seq_lengths_q will be similar to [1, 2, 4, 5, 6, 6, 7, 8]
        """
        need_padding = num_reqs_pad_size > 0
        if need_padding:
            start_val = actual_seq_lengths_q[-1]
            end_val = num_reqs + num_reqs_pad_size
            num_step = num_reqs_pad_size
            interpolated = np.round(
                np.linspace(start_val, end_val,
                            num_step + 1)[1:]).astype(int).tolist()
            assert interpolated[-1] == end_val
            assert len(interpolated) == num_reqs_pad_size
            actual_seq_lengths_q = actual_seq_lengths_q + interpolated
        return actual_seq_lengths_q

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
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
        self.set_num_actual_tokens(common_attn_metadata)
        assert self.num_decodes + self.num_prefills == num_reqs
        assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens

        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        self.slot_mapping = common_attn_metadata.slot_mapping[:self.
                                                              num_actual_tokens]

        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        self.query_lens = query_seq_lens_cpu[:num_reqs]
        self.seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        self.set_prefill_block_table(common_attn_metadata)

        prefill_metadata = None
        # if self.num_prefills > 0:
        #     prefill_metadata = self.build_prefill_metadata(
        #         common_prefix_len, common_attn_metadata)

        decode_metadata = None
        # if self.num_decodes > 0:
        #     decode_metadata = self.build_decode_metadata(
        #         common_prefix_len, common_attn_metadata)

        return self.metadata_cls(  # type: ignore
            num_actual_tokens_pcp_padded=self.num_actual_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            query_lens=self.query_lens.tolist(),
            slot_mapping=self.slot_mapping,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_mask=common_attn_metadata.attn_mask,
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            block_tables=self.block_table,
            seq_lens=self.seq_lens,
        )

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

    def set_prefill_block_table(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        # If graph_pad_size > -1, mean is running in fullgraph mode.
        self.graph_pad_size = common_attn_metadata.graph_pad_size
        # NOTE: Maybe this block_table change can be removed when graph_pad_size > 1.
        if self.graph_pad_size > common_attn_metadata.num_reqs and self.speculative_config.disable_padded_drafter_batch:
            self.block_table = (
                common_attn_metadata.block_table_tensor[:self.graph_pad_size])
        else:
            self.block_table = (
                common_attn_metadata.block_table_tensor[:common_attn_metadata.
                                                        num_reqs])

    def set_decode_block_table(self):
        self.block_table = self.block_table[:self.num_decodes, ...]


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
            self.softmax_scale = self.inderxer_dim ** -0.5

            self.indexer_compress = self.indexer.compressor

        # compress param
        if self.compressor is not None:
            self.compressor_head_dim = self.compressor.head_dim
            self.overlap = self.compressor.overlap
            self.rotate = self.compressor.rotate

            self.ape = self.compressor.ape
            self.wkv = self.compressor.wkv
            self.wgate = self.compressor.wgate
            self.compress_norm = self.compressor.norm

        
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

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass

    def _v_up_proj(self, x, has_prefill: bool):
        if x.dtype in [torch.float16, torch.bfloat16] \
                and hasattr(torch.ops._C_ascend, "batch_matmul_transpose") \
                and not self.enable_sfa_cp \
                and not has_prefill:
            x = x.view(-1, self.num_heads, self.kv_lora_rank)
            b, _, _ = x.shape
            res = torch.empty((b, self.num_heads, self.v_head_dim),
                              dtype=x.dtype,
                              device=x.device)
            torch.ops._C_ascend.batch_matmul_transpose(x, self.W_UV, res)
            x = res.reshape(-1, self.num_heads * self.v_head_dim)
        else:
            # Convert from (B, N, L) to (N, B, L)
            x = x.view(-1, self.local_num_heads,
                       self.kv_lora_rank).transpose(0, 1)
            # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            x = torch.bmm(x, self.W_UV)
            # # Convert from (N, B, V) to (B, N * V)
            x = x.transpose(0,
                            1).reshape(-1,
                                       self.local_num_heads * self.v_head_dim)
        return x


    # def exec_kv(
    #     self,
    #     kv_no_split: torch.Tensor,
    #     cos: torch.Tensor,
    #     sin: torch.Tensor,
    #     kv_cache: Tuple,
    #     slots: torch.Tensor,
    #     slots_cp: Optional[torch.Tensor],
    # ):
    #     B = kv_no_split.shape[0]
    #     N = self.num_kv_heads
    #     S = 1
    #     # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
    #     kv_no_split = kv_no_split.view(
    #         B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
    #     cache_mode = "PA"

    #     if self.enable_sfa_cp:
    #         assert slots_cp is not None
    #         _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
    #             kv_no_split,
    #             self.kv_a_layernorm.weight,
    #             cos,
    #             sin,
    #             slots_cp.to(torch.int64),
    #             kv_cache[1],
    #             kv_cache[0],
    #             epsilon=self.kv_a_layernorm.variance_epsilon,
    #             cache_mode=cache_mode,
    #             is_output_kv=True,
    #         )
    #         #TODO: Temporarily adapt SFA-CP and replace it later with PCP. --clrs97
    #         k_pe = get_tp_group().all_gather(k_pe, 0)
    #         k_nope = get_tp_group().all_gather(k_nope, 0)

    #         if kv_cache is not None:
    #             torch_npu.npu_scatter_nd_update_(
    #                 kv_cache[0].view(-1, k_nope.shape[-1]), slots.view(-1, 1),
    #                 k_nope.view(-1, k_nope.shape[-1]))
    #             torch_npu.npu_scatter_nd_update_(
    #                 kv_cache[1].view(-1, k_pe.shape[-1]), slots.view(-1, 1),
    #                 k_pe.view(-1, k_pe.shape[-1]))
    #     else:
    #         torch_npu.npu_kv_rmsnorm_rope_cache(
    #             kv_no_split,
    #             self.kv_a_layernorm.weight,
    #             cos,
    #             sin,
    #             slots.to(torch.int64),
    #             kv_cache[1],
    #             kv_cache[0],
    #             epsilon=self.kv_a_layernorm.variance_epsilon,
    #             cache_mode=cache_mode,
    #         )

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        if inverse:
            sin = sin * -1
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch_npu.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        # forward_context = get_forward_context()
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        has_prefill = attn_metadata.has_prefill
        cos = attn_metadata.cos
        sin = attn_metadata.sin
        actual_seq_lengths_query = attn_metadata.cum_query_lens
        actual_seq_lengths_key = attn_metadata.seq_lens
        output_padded = output

        # q
        qr = q = self.wq_a(hidden_states) # bs
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim)) # tp
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self.rope_single(q_pe, cos, sin)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # win kv & tok_dis
        kv = self.wkv(hidden_states)
        kv = self.kv_norm(kv)
        kv_nope, kv_pe = kv.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_pe = self.rope_single(kv_pe, cos, sin)
        kv = torch.cat([kv_nope, kv_pe], dim=-1)

        # topk_idxs = self.get_window_topk_idxs(self.win, bsz, seqlen, start_pos) # ignorn

        if self.compress_ration > 1:
            if self.compress_ratio == 4:
                compress_topk_idxs = self.indexer_select(x=hidden_states,
                                                    qr=qr,
                                                    kv_cache=kv_cache,
                                                    attn_metadata=attn_metadata,
                                                    cos=cos,
                                                    sin=sin,
                                                    actual_seq_lengths_query=actual_seq_lengths_query,
                                                    actual_seq_lengths_key=actual_seq_lengths_key)
            elif self.compress_ration == 128:
                compress_topk_idxs = self.get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv_cache[0],
            value=kv_cache[0],
            sparse_indices=topk_indices,
            scale_value=self.scale,
            sparse_block_size=1,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=kv_cache[1],
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
        attn_output = self._v_up_proj(attn_output, has_prefill)
        maybe_npu_prefetch(inputs=self.o_proj.weight,
                           dependency=attn_output,
                           max_size=MAX_O_PROJ_PREFETCH_SIZE,
                           enabled=self.enable_prefetch)
        output[...] = self.o_proj(attn_output)[0]
        return output_padded

    def indexer_select(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        need_gather_q_kv: bool = False,
    ):
        # q process in new stream
        q, _ = self.wq_b(qr)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
        q = q.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]

        k_proj, _ = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k_proj = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            k_proj, need_gather_q_kv)
        k = self.k_norm(k_proj).unsqueeze(1)
        k = k.view(-1, 1, self.head_dim)

        # if HAS_TRITON:
        #     cos = cos.view(-1, self.qk_rope_head_dim)
        #     sin = sin.view(-1, self.qk_rope_head_dim)
        #     q, k = rope_forward_triton(q,
        #                                k,
        #                                cos,
        #                                sin,
        #                                rope_dim=self.qk_rope_head_dim,
        #                                is_neox_style=True)

        ## rope
        cos_q, sin_q = cos, sin
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

        q_pe, q_nope = torch.split(
            q,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1)  # [b,s,64,64+64]

        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos_q, sin_q)
        q_pe = q_pe.squeeze(2)
        q = torch.cat([q_pe, q_nope], dim=-1)  # [b*s,64,128]
        ## rope

        self.compress_forward(x, kv_cache)

        weights, _ = self.weights_proj(x)

        block_table = attn_metadata.block_tables

        topk_indices = torch.ops._C_ascend.npu_lightning_indexer(
            query=q,
            key=kv_cache[2],          # kv cache ?
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=512,
            sparse_mode=3)
        return topk_indices
    