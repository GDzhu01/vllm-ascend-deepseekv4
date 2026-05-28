import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Tuple, Type, TypeVar

import torch
import torch.distributed as dist
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
from vllm_ascend.distributed.parallel_state import get_otp_group
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    AscendDeviceType,
    attention_calculation_stream,
    extract_dsv4_layer_index,
    get_ascend_device_type,
    get_dsv4_compress_ratio,
    log_oproj_tp_debug,
    npu_stream_switch,
    olora_tp_enable,
    oproj_tp_enable,
    prefetch_stream,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
import os
import custom_ops

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_ascend.ops.triton.rms_norm import triton_q_rms

if HAS_TRITON:
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms  # noqa: F811
else:
    triton_q_rms = None  # type: ignore

BUILD_METADATA_STEP_PREFILL = 0
BUILD_METADATA_STEP_DECODE = 1

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm.distributed.parallel_state import get_tp_group
import torch
from vllm.distributed.communication_op import tensor_model_parallel_all_gather

# 选择 device 模块（CUDA 或 NPU）
try:
    import torch_npu  # noqa
    _dev = torch.npu
except ImportError:
    _dev = torch.cuda

# 全局通信流（每个 device 一条就够，按需懒初始化）
_comm_stream = None
def _get_comm_stream():
    global _comm_stream
    if _comm_stream is None:
        _comm_stream = _dev.Stream()
    return _comm_stream

def async_tp_all_gather(x: torch.Tensor, pad_size: int, do_gather: bool = True):
    """
    异步版 tensor_model_parallel_all_gather.
    返回 (out_tensor, event, pad_size). 用 wait_and_unpad 拿结果.
    """
    if not do_gather:
        return x, None, 0

    comm_stream = _get_comm_stream()
    cur_stream = _dev.current_stream()

    # ① 让 comm_stream 等待 x 在 cur_stream 上的生产完成
    comm_stream.wait_stream(cur_stream)

    with _dev.stream(comm_stream):
        # ② 防止 x 的内存被 cur_stream 提前释放
        x.record_stream(comm_stream)
        # ③ 直接复用 vllm 原封装, 保留所有内部逻辑
        out = tensor_model_parallel_all_gather(x, 0)

    # ④ 记录完成事件
    event = comm_stream.record_event()
    return out, event, pad_size

def wait_and_unpad(out: torch.Tensor, event, pad_size: int) -> torch.Tensor:
    if event is None:
        return out
    # ⑤ 当前计算流等通信完成
    _dev.current_stream().wait_event(event)
    # ⑥ 防止 out 被 comm_stream 在还没被 cur_stream 用完时回收
    out.record_stream(_dev.current_stream())
    if pad_size > 0:
        out = out[:-pad_size]
    return out

def _scatter_nd_update_asc(var: torch.Tensor, indices: torch.Tensor,
                           update: torch.Tensor) -> None:
    if var.numel() == 0 or indices.numel() == 0 or update.numel() == 0:
        return

    indices = indices.view(-1, 1)
    update_2d = update.view(-1, update.shape[-1])
    if update_2d.shape[0] > indices.shape[0]:
        update_2d = update_2d[:indices.shape[0]]
    elif update_2d.shape[0] < indices.shape[0]:
        indices = indices[:update_2d.shape[0]]

    import custom_ops

    torch.ops.custom.scatter_nd_update_asc(var, indices, update_2d)


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
        self.spec_slot_mapping = None
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.spec_slot_mapping = [torch.zeros((vllm_config.scheduler_config.max_num_batched_tokens, ), dtype=torch.int32, device=self.device) for _ in range(spec_token_num)]
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

        self.reorder_batch_threshold = self.decode_threshold
        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
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
        
        self.compressor_ratio = getattr(kv_cache_spec, 'compress_ratio', 0)
        hf_config = self.model_config.hf_config
        layer_idx = extract_dsv4_layer_index(hf_config, layer_names[0])
        self.layer_compressor_ratio = get_dsv4_compress_ratio(hf_config, layer_idx)

        if AscendDSAMetadataBuilder.hadamard is None:
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

            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            self.query_lens = query_seq_lens_cpu[:num_reqs]
            self.common_ratio_to_sas_metadata["query_lens"] = self.query_lens
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
            self.query_lens = self.common_ratio_to_sas_metadata["query_lens"]

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
            attn_mask=None,
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
        # reqs_start: the start request position of prefill request
        reqs_start = self.num_decodes
        # reqs_start: the start token position of prefill request
        tokens_start = self.num_decode_tokens
        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        if self.prefill_ratio_to_sas_metadata.get("prefill_input_positions", None) is None:
            input_positions = common_attn_metadata.positions[:self.
                                                         num_actual_tokens].long(
                                                         )
            max_query_len = self.query_lens[reqs_start:].max().item()
            max_seq_lens = common_attn_metadata.seq_lens_cpu[reqs_start:].max(
            ).item()
            self.prefill_ratio_to_sas_metadata["input_positions"] = input_positions
            self.prefill_ratio_to_sas_metadata["max_query_len"] = max_query_len
            self.prefill_ratio_to_sas_metadata["max_seq_lens"] = max_seq_lens

            prefill_query_start_loc = query_start_loc[
                reqs_start:] - query_start_loc[reqs_start]
            prefill_input_positions = input_positions[tokens_start:]
            self.prefill_ratio_to_sas_metadata["prefill_input_positions"] = prefill_input_positions
            self.prefill_ratio_to_sas_metadata["prefill_query_start_loc"] = prefill_query_start_loc

            cos, sin = get_cos_and_sin_dsa(prefill_input_positions)
            self.prefill_ratio_to_sas_metadata["cos"] = cos
            self.prefill_ratio_to_sas_metadata["sin"] = sin

            prefill_seq_lens = self.seq_lens[reqs_start:]
            num_prefill = prefill_seq_lens.shape[0]
            self.prefill_ratio_to_sas_metadata["prefill_seq_lens"] = prefill_seq_lens
            self.prefill_ratio_to_sas_metadata["num_prefill"] = num_prefill
        else:
            input_positions = self.prefill_ratio_to_sas_metadata["input_positions"]
            max_query_len = self.prefill_ratio_to_sas_metadata["max_query_len"]
            max_seq_lens = self.prefill_ratio_to_sas_metadata["max_seq_lens"]
            prefill_input_positions = self.prefill_ratio_to_sas_metadata["prefill_input_positions"]
            prefill_query_start_loc = self.prefill_ratio_to_sas_metadata["prefill_query_start_loc"]
            cos = self.prefill_ratio_to_sas_metadata["cos"]
            sin = self.prefill_ratio_to_sas_metadata["sin"]
            prefill_seq_lens = self.prefill_ratio_to_sas_metadata["prefill_seq_lens"]
            num_prefill = self.prefill_ratio_to_sas_metadata["num_prefill"]

        def _get_padded_compressed_position(prefill_input_positions,
                                            compress_ratio):
            if compress_ratio <= 1:
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
            # Note(qcs): some models use compress_ratio=0 as non-compression tag.
            _cmp_seq_lens = (
                prefill_seq_lens // compress_ratio if compress_ratio >= 1 
                else prefill_seq_lens
            )
            return torch.concat(
                (torch.tensor([0], device=_cmp_seq_lens.device),
                 torch.cumsum(_cmp_seq_lens, -1)),
                dim=-1)

        def _get_compressed_decode_token_start_and_end(decode_input_positions,
                                                       compress_ratio):
            # Note(qcs): some models use compress_ratio=0 as non-compression tag.
            if compress_ratio == 0:
                compress_ratio = 1
            # TODO(yilin): decode_input_positions is a device tensor,
            # this will introduce sync operation. Refactor me to torch.where instead
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            compressed_decode_num = mask.sum()

            end = min(
                self.num_prefill_tokens,
                self.num_prefill_tokens // compress_ratio + self.num_prefills)
            return compressed_decode_num, end

        if self.prefill_ratio_to_sas_metadata.get(f"c{self.layer_compressor_ratio}_cos", None) is None:
            compress_cos, compress_sin = get_cos_and_sin_dsa(
                _get_padded_compressed_position(prefill_input_positions,
                                                self.layer_compressor_ratio))
            self.prefill_ratio_to_sas_metadata[f"c{self.layer_compressor_ratio}_cos"] = compress_cos
            self.prefill_ratio_to_sas_metadata[f"c{self.layer_compressor_ratio}_sin"] = compress_sin
        else:
            compress_cos = self.prefill_ratio_to_sas_metadata[f"c{self.layer_compressor_ratio}_cos"]
            compress_sin = self.prefill_ratio_to_sas_metadata[f"c{self.layer_compressor_ratio}_sin"]

        if self.prefill_ratio_to_sas_metadata.get(f"compressed_c{self.compressor_ratio}_tokens_start", None) is None:
            decode_input_positions = input_positions[:tokens_start]
            compressed_tokens_start, compressed_tokens_end = _get_compressed_decode_token_start_and_end(
                decode_input_positions, self.compressor_ratio)
            self.prefill_ratio_to_sas_metadata[f"compressed_c{self.compressor_ratio}_tokens_start"] = compressed_tokens_start
            self.prefill_ratio_to_sas_metadata[f"compressed_c{self.compressor_ratio}_tokens_ebd"] = compressed_tokens_end
        else:
            compressed_tokens_start = self.prefill_ratio_to_sas_metadata[f"compressed_c{self.compressor_ratio}_tokens_start"]
            compressed_tokens_end = self.prefill_ratio_to_sas_metadata[f"compressed_c{self.compressor_ratio}_tokens_ebd"]   

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

        cu_c4_cmp_seqlen_list = None
        cu_c128_cmp_seqlen_list = None

        layer_name = f"c{self.compressor_ratio}"
        if self.compressor_ratio <= 1:
            if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
                self.prefill_ratio_to_sas_metadata[layer_name] = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
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
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=False,
                    device=str(self.seqused_q.device))
            sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
        elif self.compressor_ratio == 4:
            if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
                self.prefill_ratio_to_sas_metadata[layer_name] = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
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
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                    device=str(self.seqused_q.device))
            sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
        else:
            if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
                self.prefill_ratio_to_sas_metadata[layer_name] = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
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
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                    device=str(self.seqused_q.device))
            sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
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
            attn_mask=None,
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
            decode_input_positions = input_positions_cpu
            seq_lens_list = common_attn_metadata.seq_lens_cpu[:self.
                                                          num_decodes].tolist(
                                                          )
            self.decode_ratio_to_sas_metadata["query_start_loc_cpu"] = query_start_loc_cpu
            self.decode_ratio_to_sas_metadata["decode_input_positions"] = decode_input_positions
            self.decode_ratio_to_sas_metadata["max_seq_lens"] = max_seq_lens
            self.decode_ratio_to_sas_metadata["seq_lens_list"] = seq_lens_list

            max_seqlen_kv = torch.max(
            common_attn_metadata.seq_lens_cpu[:self.num_decodes]).item()
            max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()
            self.decode_ratio_to_sas_metadata["max_seqlen_kv"] = max_seqlen_kv
            self.decode_ratio_to_sas_metadata["max_seqlen_q"] = max_seqlen_q

            seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
            start_pos_decode = self.seq_lens[:self.num_decodes] - seq_lens_q
            self.decode_ratio_to_sas_metadata["start_pos_decode"] = start_pos_decode
        else:
            query_start_loc = self.decode_ratio_to_sas_metadata["query_start_loc"]
            input_positions = self.decode_ratio_to_sas_metadata["input_positions"]
            cos = self.decode_ratio_to_sas_metadata["cos"]
            sin = self.decode_ratio_to_sas_metadata["sin"]
            query_start_loc_cpu = self.decode_ratio_to_sas_metadata["query_start_loc_cpu"]
            decode_input_positions = self.decode_ratio_to_sas_metadata["decode_input_positions"]
            max_seq_lens = self.decode_ratio_to_sas_metadata["max_seq_lens"]
            seq_lens_list = self.decode_ratio_to_sas_metadata["seq_lens_list"]
            max_seqlen_kv = self.decode_ratio_to_sas_metadata["max_seqlen_kv"]
            max_seqlen_q = self.decode_ratio_to_sas_metadata["max_seqlen_q"]
            start_pos_decode = self.decode_ratio_to_sas_metadata["start_pos_decode"]

        block_table_size = self.get_block_table_size(
            common_attn_metadata, BUILD_METADATA_STEP_DECODE)

        cp_seq_len, batch_seq_mask = None, None

        def _get_padded_compressed_position(decode_input_positions,
                                            compress_ratio, device):
            if compress_ratio <= 1:
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
        if self.decode_ratio_to_sas_metadata.get(f"c{self.layer_compressor_ratio}" + "_cos", None) is None:
            compress_cos, compress_sin = get_cos_and_sin_dsa(
                {
                    f"c{self.layer_compressor_ratio}":
                    _get_padded_compressed_position(decode_input_positions,
                                                    self.layer_compressor_ratio,
                                                    input_positions.device)
                },
                use_cache=True)
            self.decode_ratio_to_sas_metadata[f"c{self.layer_compressor_ratio}" + "_cos"] = compress_cos
            self.decode_ratio_to_sas_metadata[f"c{self.layer_compressor_ratio}" + "_sin"] = compress_sin
        else:
            compress_cos = self.decode_ratio_to_sas_metadata[f"c{self.layer_compressor_ratio}" + "_cos"]
            compress_sin = self.decode_ratio_to_sas_metadata[f"c{self.layer_compressor_ratio}" + "_sin"]

        def _get_compressed_decode_token_start(decode_input_positions,
                                               compress_ratio):
            # Note(qcs): some models use compress_ratio=0 as non-compression tag.
            if compress_ratio == 0:
                compress_ratio = 1
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            compressed_decode_num = mask.sum().item()
            return compressed_decode_num

        if self.decode_ratio_to_sas_metadata.get("compressed_tokens_start_" + str(self.compressor_ratio), None) is None:
            compressed_tokens_start = _get_compressed_decode_token_start(
                decode_input_positions, self.compressor_ratio)
            self.decode_ratio_to_sas_metadata["compressed_tokens_start_" + str(self.compressor_ratio)] = compressed_tokens_start
        else:
            compressed_tokens_start = self.decode_ratio_to_sas_metadata["compressed_tokens_start_" + str(self.compressor_ratio)]

        slot_mapping = self.slot_mapping[:compressed_tokens_start]

        assert self.start_pos_decode is not None
        self.start_pos_decode.fill_(0)
        self.start_pos_decode[:self.num_decodes] = start_pos_decode

        if num_reqs_actual is not None and num_reqs_actual < self.num_decodes:
            self.start_pos_decode[num_reqs_actual:].fill_(0)
            self.block_table[num_reqs_actual:self.num_decodes, ...].fill_(0)

        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        index_topk = self.model_config.hf_config.index_topk

        assert self.decode_sas_metadata is not None
        if self.compressor_ratio <= 1:
            if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
                self.decode_ratio_to_sas_metadata[layer_name] = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
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
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=False,
                    device=str(self.seqused_q.device))
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
        elif self.compressor_ratio == 4:
            if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
                self.decode_ratio_to_sas_metadata[layer_name] = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
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
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                    device=str(self.seqused_q.device))
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
        else:
            if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
                self.decode_ratio_to_sas_metadata[layer_name] = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
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
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                    device=str(self.seqused_q.device))
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
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
            attn_mask=None,
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

    def build_for_drafting(self,
        draft_step: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSADecodeMetadata:
        assert self.compressor_ratio <= 1, "vLLM-Ascend only support SWA-layer for Deepseek-V4 now."
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
        num_reqs_actual = kwargs.get("num_reqs_actual", None)
        num_input_tokens = common_attn_metadata.num_input_tokens
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        if num_prefills:
            cos, sin = get_cos_and_sin_dsa(input_positions)
        else:
            cos, sin = get_cos_and_sin_dsa(input_positions, True)

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        self.spec_slot_mapping[draft_step - 1][:num_input_tokens] = slot_mapping

        prefill_metadata = None
        if num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata_for_drafting(
                draft_step=draft_step,
                common_attn_metadata=common_attn_metadata,
                reqs_start=num_decodes,
                tokens_start=num_decode_tokens,
                num_prefill_tokens=num_prefill_tokens)

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = self.build_decode_metadata_for_drafting(
                draft_step=draft_step,
                common_attn_metadata=common_attn_metadata,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens)

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_lens=None,
            slot_mapping=None,
            head_dim=self.model_config.get_head_size(),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            attn_mask=None,
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=None,
            block_tables=None,
            seq_lens=None,
            cos=cos,
            sin=sin,
            hadamard=None,
        )

    def build_prefill_metadata_for_drafting(self,
        draft_step: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs,
    ) -> AscendDSAPrefillMetadata:
        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size

        reqs_start = kwargs.get("reqs_start")
        tokens_start = kwargs.get("tokens_start")
        num_prefill_tokens = kwargs.get("num_prefill_tokens")
        query_start_loc = common_attn_metadata.query_start_loc
        prefill_query_start_loc = query_start_loc[reqs_start:] - query_start_loc[reqs_start]
        seq_lens_q = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
        seq_lens = common_attn_metadata.seq_lens[reqs_start:]

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        input_positions = common_attn_metadata.positions[:num_actual_tokens].long()
        prefill_input_positions = input_positions[tokens_start:]
        cos, sin = get_cos_and_sin_dsa(prefill_input_positions)

        prefill_slot_mapping = self.spec_slot_mapping[draft_step - 1][tokens_start:num_prefill_tokens]
        block_table = common_attn_metadata.block_table_tensor[:common_attn_metadata.num_reqs]

        sas_metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=prefill_query_start_loc,
            cu_seqlens_ori_kv=prefill_query_start_loc,
            cu_seqlens_cmp_kv=None,
            seqused_q=self.seqused_q,
            seqused_kv=seq_lens,
            max_seqlen_q=seq_lens_q.max(),
            max_seqlen_kv=seq_lens.max(),
            batch_size=len(seq_lens),
            cmp_ratio=1,
            ori_mask_mode=4,  # 4:sliding window
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
            device=str(self.seqused_q.device))

        return AscendDSAPrefillMetadata(
            attn_mask=None,
            query_lens=None,
            seq_lens=seq_lens,
            context_lens=None,
            input_positions=None,
            block_table=block_table[reqs_start:, ...],
            slot_mapping=prefill_slot_mapping,
            max_query_len=None,
            max_seq_lens=None,
            query_start_loc=prefill_query_start_loc,
            sin=sin,
            cos=cos,
            compress_sin=None,
            compress_cos=None,
            start_pos=None,
            sas_metadata=sas_metadata,
            qli_metadata=None,
            cu_c4_cmp_seqlen_list=None,
            cu_c128_cmp_seqlen_list=None)

    def build_decode_metadata_for_drafting(self,
        draft_step: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs,
    ) -> AscendDSADecodeMetadata:
        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size

        num_decodes = kwargs.get("num_decodes")
        num_decode_tokens = kwargs.get("num_decode_tokens")
        query_start_loc = common_attn_metadata.query_start_loc[:num_decodes + 1]
        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:num_decodes+ 1]
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()

        if common_attn_metadata._seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        elif common_attn_metadata.seq_lens_cpu is not None:
            _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        else:
            _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        max_seqlen_kv = torch.max(_seq_lens_cpu[:num_decodes]).item()

        input_positions = common_attn_metadata.positions[:num_decode_tokens].long()
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True)

        slot_mapping = self.spec_slot_mapping[draft_step - 1][:num_decode_tokens]
        block_table = common_attn_metadata.block_table_tensor

        decode_sas_metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=query_start_loc,  # cached
            cu_seqlens_ori_kv=self.cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=self.cu_seqlens_cmp_kv,
            seqused_q=self.seqused_q,
            seqused_kv=seq_lens[:num_decodes],  # cached
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=len(seq_lens[:num_decodes]),  # cached
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
            device=str(self.seqused_q.device))

        decode_metadata = AscendDSADecodeMetadata(
            input_positions=None,
            block_table=block_table[:num_decodes, ...],
            slot_mapping=slot_mapping,
            seq_lens=seq_lens[:num_decodes],  # cached
            seq_lens_list=None,
            max_seq_lens=None,
            max_seqlen_kv=None,
            max_seqlen_q=None,
            attn_mask=None,
            query_start_loc=query_start_loc,  # cached
            query_start_loc_cpu=None,
            sin=sin[:num_decode_tokens, ...],
            cos=cos[:num_decode_tokens, ...],
            compress_sin=None,
            compress_cos=None,
            cp_seq_len=None,
            batch_seq_mask=None,
            start_pos=None,  # cached
            sas_metadata=decode_sas_metadata,
            qli_metadata=None)
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
            self.index_topk = self.indexer.index_topk

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

        
        if os.environ.get("USE_COMPRESSOR_PREFETCH", "0").lower() in ("1", "true", "yes", "on"):
            self.compressor_prefetch = True
            self.compressor_stream = prefetch_stream()
        else:
            self.compressor_prefetch = False
        if os.environ.get("USE_ATTN_SP", "0").lower() in ("1", "true", "yes", "on"):
            self.use_attn_sp = True
        else:
            self.use_attn_sp = False

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass

    def dsa_warmup_with_multistream(self, hidden_states: torch.Tensor) -> None:
        """Warm up the aux (attention_calculation_stream) so ACL graph capture
        sees a fully-formed stream graph with **legal** addresses on the aux
        stream.

        Why this is needed (背景):
          - When ``multistream_dsa_preprocess=True``, the real DSA forward
            schedules ``wkv`` / ``kv_norm`` / ``inplace_partial_rotary_mul`` /
            ``scatter_nd_update_v2`` / ``npu_dynamic_quant`` / ``weights_proj``
            on the aux stream returned by ``attention_calculation_stream()``.
          - During ``execute_dummy_batch``/``profile_run`` we do not have a
            real ``attn_metadata`` — the persistent ``input_batch.block_table``
            / ``slot_mapping`` GPU buffers may still hold stale entries left
            over from prior real requests (or KV-transfer-written data on D
            ranks). If we let the real DSA forward run with these residuals
            inside an ACL-graph replay window, the aux-stream
            ``npu_dynamic_quant`` / ``scatter_nd_update_v2`` will dereference
            those stale slot ids and trip an MTE DDR-out-of-range fault.
          - We instead drive each aux-stream op once with locally-allocated
            ``torch.zeros(...)`` buffers, so capture records *legal*
            addresses, and downstream replays observe a well-formed graph.

        Mirrors ``dsa_warmup_with_multistream`` in vllm-ascend main
        (`vllm_ascend/attention/dsa_v1.py`), adapted to this fork's option
        ``multistream_dsa_preprocess`` and stream ``attention_calculation_stream``.
        """
        if not self.multistream_dsa_preprocess:
            return

        aux_stream = attention_calculation_stream()
        dtype = hidden_states.dtype
        device = hidden_states.device
        hidden_size = hidden_states.shape[-1]
        hidden_states_dummy = torch.zeros(
            1, hidden_size, dtype=dtype, device=device)

        e_warmup = torch.npu.current_stream().record_event()
        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_warmup)

            # ---- Part 1: wkv (+ optional dynamic quant for w8a8 int8) ----
            if hasattr(self.wkv, "weight_scale") and \
                    self.wkv.weight.dtype == torch.int8:
                kv_q_dummy, kv_s_dummy = torch_npu.npu_dynamic_quant(
                    hidden_states_dummy)
                _ = torch_npu.npu_quant_matmul(
                    kv_q_dummy,
                    self.wkv.weight,
                    self.wkv.weight_scale,
                    pertoken_scale=kv_s_dummy,
                    output_dtype=dtype,
                )
            else:
                _ = self.wkv(hidden_states_dummy)

            assert self.rope_head_dim is not None
            kv_dummy = torch.zeros(
                1,
                self.nope_head_dim + self.rope_head_dim,
                dtype=dtype,
                device=device,
            )
            _ = self.kv_norm(kv_dummy)

            # ---- Part 2: indexer aux-stream ops (dynamic_quant +
            #              scatter_nd_update + weights_proj). Only needed for
            #              compress_ratio==4 paths that actually run them on
            #              the aux stream.
            if self.indexer is not None and self.compress_ratio == 4:
                soc_version = get_ascend_device_type()
                dst_type = (torch.float8_e4m3fn
                            if soc_version in {AscendDeviceType.A5}
                            else torch.int8)

                kv_quant_dummy, kv_scale_dummy = torch_npu.npu_dynamic_quant(
                    hidden_states_dummy, dst_type=dst_type)
                if soc_version not in {AscendDeviceType.A5}:
                    kv_scale_dummy = kv_scale_dummy.to(torch.float16)
                kv_scale_dummy = kv_scale_dummy.unsqueeze(-1)

                slot_mapping_dummy = torch.zeros(
                    1, 1, dtype=torch.int64, device=device)
                dummy_shape = (1, 1, 1, kv_quant_dummy.shape[-1])
                indexer_k_cache_dummy = torch.zeros(
                    dummy_shape, dtype=kv_quant_dummy.dtype, device=device)
                indexer_scale_cache_dummy = torch.zeros(
                    dummy_shape, dtype=torch.float16, device=device)
                _scatter_nd_update_asc(
                    indexer_k_cache_dummy, slot_mapping_dummy, kv_quant_dummy)
                _scatter_nd_update_asc(
                    indexer_scale_cache_dummy,
                    slot_mapping_dummy,
                    kv_scale_dummy,
                )
                _ = self.weights_proj(hidden_states_dummy)

        torch.npu.current_stream().wait_stream(aux_stream)

    def _assert_deepseek_v4_oproj_tp(self, layer_name: str) -> None:
        hf_config = self.vllm_config.model_config.hf_text_config
        model_type = getattr(hf_config, "model_type", None)
        assert model_type == "deepseek_v4", (
            "oproj tensor parallel DSA path is only supported for "
            f"deepseek_v4, got model_type={model_type}, "
            f"layer_name={layer_name}"
        )

    def _forward_deepseek_v4_oproj_tp(
        self,
        o_proj_input: torch.Tensor,
        output: torch.Tensor,
        num_tokens: int,
        layer_name: str,
        wo_a_prefix: str,
    ) -> None:
        self._assert_deepseek_v4_oproj_tp(layer_name)
        otp_group = get_otp_group()
        otp_size = otp_group.world_size
        otp_rank = otp_group.rank_in_group

        local_batch_size = o_proj_input.size(0)
        total_groups = o_proj_input.size(1)
        input_size_per_group = o_proj_input.size(2)
        assert local_batch_size == num_tokens, (
            f"o_proj local_batch_size={local_batch_size} must match "
            f"num_tokens={num_tokens}, layer_name={layer_name}"
        )
        assert self.wo_a.weight.dim() == 3, (
            "DeepSeek V4 wo_a weight must be reshaped to "
            f"[groups_per_rank, input, rank], got "
            f"{tuple(self.wo_a.weight.shape)}"
        )

        groups_per_rank = self.wo_a.weight.shape[0]
        assert groups_per_rank > 0, (
            f"wo_a groups_per_rank must be positive, prefix={wo_a_prefix}"
        )
        assert total_groups == groups_per_rank * otp_size, (
            f"wo_a input groups={total_groups} must equal "
            f"groups_per_rank={groups_per_rank} * otp_size={otp_size}, "
            f"prefix={wo_a_prefix}"
        )
        assert self.wo_a.weight.shape[1] == input_size_per_group, (
            f"wo_a input dim mismatch: input={input_size_per_group}, "
            f"weight={tuple(self.wo_a.weight.shape)}, prefix={wo_a_prefix}"
        )
        assert self.wo_a.weight.shape[2] == self.o_lora_rank, (
            f"wo_a rank mismatch: expected={self.o_lora_rank}, "
            f"weight={tuple(self.wo_a.weight.shape)}, prefix={wo_a_prefix}"
        )

        wo_b_input_size = groups_per_rank * self.o_lora_rank
        assert self.wo_b.input_size_per_partition == wo_b_input_size, (
            f"wo_b local input mismatch: expected={wo_b_input_size}, "
            f"actual={self.wo_b.input_size_per_partition}, "
            f"prefix={getattr(self.wo_b, 'prefix', None)}"
        )
        assert self.wo_b.weight.shape[-1] == wo_b_input_size, (
            f"wo_b weight dim mismatch: expected last dim={wo_b_input_size}, "
            f"weight={tuple(self.wo_b.weight.shape)}, "
            f"prefix={getattr(self.wo_b, 'prefix', None)}"
        )

        send_buf = (
            o_proj_input.reshape(
                local_batch_size,
                otp_size,
                groups_per_rank,
                input_size_per_group,
            )
            .transpose(0, 1)
            .contiguous()
            .view(-1)
        )
        recv_buf = torch.empty(
            local_batch_size * total_groups * input_size_per_group,
            dtype=o_proj_input.dtype,
            device=o_proj_input.device,
        )
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            group=otp_group.device_group,
        )
        wo_a_input = recv_buf.view(
            local_batch_size * otp_size,
            groups_per_rank,
            input_size_per_group,
        )
        log_oproj_tp_debug(
            f"compute prefix={wo_a_prefix} "
            f"oproj_tp_wo_a_all2all_input_shape={tuple(o_proj_input.shape)} "
            f"oproj_tp_wo_a_local_input_shape={tuple(wo_a_input.shape)} "
            f"wo_a_weight_shape={tuple(self.wo_a.weight.shape)} "
            f"otp_rank={otp_rank} otp_size={otp_size}"
        )

        wo_a_output = torch_npu.npu_transpose_batchmatmul(
            wo_a_input,
            self.wo_a.weight,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
            batch_split_factor=1)
        wo_b_input = wo_a_output.reshape(
            local_batch_size * otp_size,
            wo_b_input_size,
        )
        wo_b_prefix = getattr(self.wo_b, "prefix", f"{layer_name}.wo_b")
        log_oproj_tp_debug(
            f"compute prefix={wo_b_prefix} "
            f"wo_a_output_shape={tuple(wo_a_output.shape)} "
            f"wo_b_input_shape={tuple(wo_b_input.shape)} "
            f"wo_b_weight_shape={tuple(self.wo_b.weight.shape)} "
            f"otp_rank={otp_rank} otp_size={otp_size}"
        )

        bias_ = None if (otp_rank > 0 or self.wo_b.skip_bias_add) else self.wo_b.bias
        assert self.wo_b.quant_method is not None
        output_parallel = self.wo_b.quant_method.apply(
            self.wo_b,
            wo_b_input,
            bias=bias_,
        )
        output_tp = otp_group.reduce_scatter(output_parallel, dim=0)
        output[...] = output_tp.view(local_batch_size, self.wo_b.output_size)
        log_oproj_tp_debug(
            f"compute prefix={wo_b_prefix} "
            f"wo_b_output_parallel_shape={tuple(output_parallel.shape)} "
            f"wo_b_output_shape={tuple(output.shape)} "
            f"otp_rank={otp_rank} otp_size={otp_size}"
        )

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

        has_prefill = attn_metadata[0].num_prefills > 0
        has_decode = attn_metadata[0].num_decodes > 0
        decode_tokens = attn_metadata[0].num_decode_tokens
        actual_tokens = attn_metadata[0].num_actual_tokens

        # ------------------------------------------------------------------
        # SP 优化:
        # 仅在「纯 prefill」(no decode) 且开启了 SP gather 的情况下,
        # 推迟入口处的 all_gather。这样 wq_a / wq_b / wkv / q_norm / kv_norm
        # 等逐 token 的算子能直接跑在 SP 分片上,每张卡只处理自己那一段,
        # 然后在 _forward_prefill 内部、第一个 _scatter_nd_update_asc 之前
        # 统一 all_gather,再做 RoPE / scatter / compressor / indexer / attn。
        #
        # 混合 prefill+decode 或纯 decode 时,prefill_hidden_states /
        # decode_hidden_states 的切片必须基于完整 token 顺序,因此仍然在
        # 入口处先 gather。
        # ------------------------------------------------------------------
        defer_gather_in_prefill = (has_prefill and not has_decode
                                and need_gather_q_kv and self.use_attn_sp)

        if defer_gather_in_prefill:
            # 纯 prefill: hidden_states 仍是 SP 分片,
            # decode_tokens == 0,所以分片整体就是 prefill 段。
            prefill_hidden_states = hidden_states
            decode_hidden_states = None
        else:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                hidden_states, need_gather_q_kv)
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
            output_prefill = self._forward_prefill(
                layer_name,
                prefill_hidden_states,
                kv_cache,
                attn_metadata,
                need_gather_q_kv=need_gather_q_kv,
                defer_gather=defer_gather_in_prefill,
            )
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
        wo_a_prefix = getattr(self.wo_a, "prefix", f"{layer_name}.wo_a")
        log_oproj_tp_debug(
            f"compute prefix={wo_a_prefix} "
            f"wo_a_input_shape={tuple(o_proj_input.shape)} "
            f"wo_a_weight_shape={tuple(self.wo_a.weight.shape)} "
            f"n_local_groups={self.n_local_groups} "
            f"o_lora_rank={self.o_lora_rank}"
        )
        if (oproj_tp_enable()
                and not getattr(forward_context, "is_draft_model", False)):
            self._forward_deepseek_v4_oproj_tp(
                o_proj_input,
                output,
                num_tokens,
                layer_name,
                wo_a_prefix,
            )
            return output_padded
        elif olora_tp_enable():
            o_proj_input = self.wo_a(o_proj_input)
        else:
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
        log_oproj_tp_debug(
            f"compute prefix={wo_a_prefix} "
            f"wo_a_output_shape={tuple(o_proj_input.shape)} "
            f"wo_b_input_shape={tuple(o_proj_input.shape)}"
        )
        output[...] = self.wo_b(o_proj_input)

        return output_padded

    def _forward_prefill(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: Tuple,
        attn_metadata: AscendDSAMetadata,
        need_gather_q_kv: bool = False,
        defer_gather: bool = False,
    ):
        compress_common_attn_metadata = None
        if self.compress_ratio == 4:
            (compressor_attn_cache, swa_cache, _, _, compressor_kv_state,
            compressor_score_state, _, _) = kv_cache
            (compressor_attn_metadata, swa_metadata, _,
            compressor_kv_state_metadata, compressor_score_state_metadata, _,
            _) = attn_metadata
            compress_common_attn_metadata = compressor_attn_metadata
        elif self.compress_ratio == 128:
            (swa_cache, compressor_attn_cache, compressor_kv_state,
            compressor_score_state) = kv_cache
            (swa_metadata, compressor_attn_metadata,
            compressor_kv_state_metadata,
            compressor_score_state_metadata) = attn_metadata
            compress_common_attn_metadata = compressor_attn_metadata
        else:
            (swa_cache, ) = kv_cache
            (swa_metadata, ) = attn_metadata
            compress_common_attn_metadata = swa_metadata

        assert compress_common_attn_metadata.prefill is not None
        cos = compress_common_attn_metadata.prefill.cos[layer_name]
        sin = compress_common_attn_metadata.prefill.sin[layer_name]
        actual_seq_lengths_query = compress_common_attn_metadata.prefill.query_start_loc
        actual_seq_lengths_key = compress_common_attn_metadata.prefill.seq_lens

        num_prefill_tokens = cos.shape[0]   # 真实 (unpadded) prefill token 数
        def gather_with_overlap():
            nonlocal hidden_states, q, kv, qr
            # ------------------------------------------------------------------
            # 阶段 1: 在 sharded hidden_states 上做 wq_a / q_norm / wkv / kv_norm
            # 它们本来就是逐 token 算子,不依赖 full sequence
            # ------------------------------------------------------------------
            do_gather = _EXTRA_CTX.flash_comm_v1_enabled and (get_tp_group().world_size > 1)
            pad_size  = _EXTRA_CTX.pad_size

            # (a) 先算 qr 的 sharded 版本 —— 它是关键路径起点
            qr_sharded = self.q_norm(self.wq_a(hidden_states))

            # (b) 立刻派发 qr 的 all_gather  ★最先入队,体量小,马上就要用★
            qr_buf, qr_handle, _ = async_tp_all_gather(qr_sharded, pad_size,
                                                    do_gather=do_gather)

            # (c) 算 kv 的 sharded 版本（与 qr.all_gather overlap）
            kv_sharded = self.kv_norm(self.wkv(hidden_states))
            kv_inner   = self.nope_head_dim + self.rope_head_dim
            kv_2d      = kv_sharded.reshape(kv_sharded.shape[0], kv_inner)

            # (d) 派发 kv 的 all_gather  ★小,排第二★
            kv_buf, kv_handle, _ = async_tp_all_gather(kv_2d, pad_size,
                                                    do_gather=do_gather)

            # (e) 最后派发 hidden_states 的 all_gather  ★最大,但下面有 wq_b 撑着★
            hs_buf, hs_handle, _ = async_tp_all_gather(hidden_states, pad_size,
                                                    do_gather=do_gather)

            # ------------------------------------------------------------------
            # 阶段 2: 关键路径 —— wq_b 大 GEMM,与 kv / hs 的通信 overlap
            # ------------------------------------------------------------------
            qr_full = wait_and_unpad(qr_buf, qr_handle, pad_size)
            if qr_full.shape[0] != num_prefill_tokens:
                qr_full = qr_full[:num_prefill_tokens]
            qr = qr_full

            # 此刻通信流上还在跑 kv.all_gather + hs.all_gather
            # 计算流跑 wq_b（最大 GEMM）+ q_rms,真正形成 overlap
            q = self.wq_b(qr_full).unflatten(-1, (self.n_local_heads, self.head_dim))
            q = triton_q_rms(q, self.eps)

            # ------------------------------------------------------------------
            # 阶段 3: 用到 kv 时才 wait
            # ------------------------------------------------------------------
            kv_full_2d = wait_and_unpad(kv_buf, kv_handle, pad_size)
            if kv_full_2d.shape[0] != num_prefill_tokens:
                kv_full_2d = kv_full_2d[:num_prefill_tokens]
            kv = kv_full_2d.view(-1, 1, kv_inner)

            # ------------------------------------------------------------------
            # 阶段 4: 最晚 wait hidden_states (compressor / indexer 才用)
            # ------------------------------------------------------------------
            hidden_states = wait_and_unpad(hs_buf, hs_handle, pad_size)
            if hidden_states.shape[0] != num_prefill_tokens:
                hidden_states = hidden_states[:num_prefill_tokens]

        def gather_without_overlap():
            nonlocal hidden_states, q, kv, qr
            # ---- 优化1: 先派发 hidden_states 的 gather (它后续才被用) ----
            # 让它在通信流上和下面的 wq_a / q_norm / wkv / kv_norm overlap
            hidden_states_full = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                hidden_states, need_gather_q_kv)
            # ---- 阶段1: SP 分片上的本地计算 (用 sharded hidden_states) ----
            qr = self.q_norm(self.wq_a(hidden_states))   # [T_local, q_lora_rank]
            # ---- 优化1续: 派发 qr 的 gather, 与下面 kv 的本地计算 overlap ----
            qr = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                qr, need_gather_q_kv)
            kv = self.wkv(hidden_states)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
            # ---- 阶段2: kv gather (统一 2D, 避免 unpad 走错分支) ----
            kv_inner = kv.shape[-1]
            kv_2d = kv.reshape(kv.shape[0], kv_inner)
            kv_2d = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                kv_2d, need_gather_q_kv)
            kv = kv_2d.view(-1, 1, kv_inner)
            # ---- unpad 兜底: 注意是对 hidden_states_full 切, 不是 hidden_states ----
            if qr.shape[0] != num_prefill_tokens:
                qr                 = qr[:num_prefill_tokens]
                kv                 = kv[:num_prefill_tokens]
                hidden_states_full = hidden_states_full[:num_prefill_tokens]
            # 用 full 替换 sharded, 后续阶段3全部基于 full sequence
            hidden_states = hidden_states_full
            # ---- 优化2: 用 full-seq qr 本地算 q (省掉 q 的 all_gather) ----
            q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
            q = triton_q_rms(q, self.eps)

            # 调试期断言
            assert q.shape[0] == cos.shape[0], (
                f"size mismatch: q={q.shape}, cos={cos.shape}, "
                f"kv={kv.shape}, hidden={hidden_states.shape}"
            )
        
        if defer_gather:
            if self.compress_ratio == 4 and need_gather_q_kv:
                gather_with_overlap()
            else:
                gather_without_overlap()
        else:
            # 非 SP 路径: 与原始实现等价, 不做任何 gather
            qr = self.q_norm(self.wq_a(hidden_states))
            q  = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
            q  = triton_q_rms(q, self.eps)
            kv = self.wkv(hidden_states)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

        # ------------------------------------------------------------------
        # RoPE / scatter / compressor / indexer / attention
        # 全部基于 full sequence
        # ------------------------------------------------------------------
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # 在 RoPE / wkv / scatter 之前记录事件，供 compressor_stream 使用
        if self.compressor_prefetch:
            main_stream_done_event = torch.npu.current_stream().record_event()
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # swa exec kv  ——  第一个 _scatter_nd_update_asc,gather 必须在它之前完成
        _scatter_nd_update_asc(
            swa_cache.view(-1, kv.shape[-1]),
            swa_metadata.prefill.slot_mapping.unsqueeze(-1), kv)

        compress_cos = compress_common_attn_metadata.prefill.compress_cos[layer_name]
        compress_sin = compress_common_attn_metadata.prefill.compress_sin[layer_name]

        if self.compressor_prefetch and self.compress_ratio > 1 and self.compress_ratio != 4:
            # 用 qr（或 q）作为 dependency,prefetch 与后续 RoPE / wkv / scatter 并行
            with torch.npu.stream(self.compressor_stream):
                self.compressor_stream.wait_event(main_stream_done_event)
                torch_npu.npu_prefetch(
                    self.compressor_wkv.weight,
                    qr,                                      # 锚点:在 qr 产出后开始预取
                    self.compressor_wkv.weight.numel() *
                    self.compressor_wkv.weight.element_size(),
                    0,                                       # offset
                )
                torch_npu.npu_prefetch(self.compressor_wgate.weight, qr,
                                self.compressor_wgate.weight.numel() *
                                self.compressor_wgate.weight.element_size())

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

            # 等待 prefetch 完成（如果启用了 prefetch）
            if self.compressor_prefetch:
                torch.npu.current_stream().wait_stream(self.compressor_stream)
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
            _scatter_nd_update_asc(
                compressor_attn_cache.view(-1, compressed_kv.shape[-1]),
                compressor_attn_metadata.prefill.slot_mapping.unsqueeze(-1),
                compressed_kv.view(-1, compressed_kv.shape[-1]))

        if self.compress_ratio <= 1:
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_cache,
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
                layout_kv="PA_ND")[0]
        elif self.compress_ratio == 4:
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_cache,
                cmp_kv=compressor_attn_cache,
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
                layout_kv="PA_ND")[0]
        else:
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_cache,
                cmp_kv=compressor_attn_cache,
                ori_block_table=swa_metadata.prefill.block_table,
                cmp_block_table=compressor_attn_metadata.prefill.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                cu_seqlens_ori_kv=actual_seq_lengths_query,
                cu_seqlens_cmp_kv=compress_common_attn_metadata.prefill
                .cu_c128_cmp_seqlen_list,
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
                layout_kv="PA_ND")[0]
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
            # (swa_cache, compressor_attn_cache, _, _, compressor_kv_state, compressor_score_state, _, _) = kv_cache
            # (swa_metadata, compressor_attn_metadata, _, compressor_kv_state_metadata, compressor_score_state_metadata, _, _) = attn_metadata
            (compressor_attn_cache, swa_cache, _, _, compressor_kv_state, compressor_score_state, _, _) = kv_cache
            (compressor_attn_metadata, swa_metadata, _, compressor_kv_state_metadata, compressor_score_state_metadata, _, _) = attn_metadata
            compress_common_attn_metadata = compressor_attn_metadata
        elif self.compress_ratio == 128:
            (swa_cache, compressor_attn_cache, compressor_kv_state, compressor_score_state) = kv_cache
            (swa_metadata, compressor_attn_metadata, compressor_kv_state_metadata, compressor_score_state_metadata) = attn_metadata
            # (compressor_attn_cache, swa_cache, compressor_kv_state, compressor_score_state) = kv_cache
            # (compressor_attn_metadata, swa_metadata, compressor_kv_state_metadata, compressor_score_state_metadata) = attn_metadata
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
        if self.compressor_prefetch:
            main_stream_done_event = torch.npu.current_stream().record_event()
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
            _scatter_nd_update_asc(
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
                # 注意:indexer_select_qli 内部已经对 self.compressor_wkv/wgate/norm
                # 发过 prefetch(在其内部 indexcom_* compressor 之后、返回之前),
                # 这里不再重复 prefetch,避免浪费 prefetch 槽位。
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

        
            if self.compressor_prefetch and self.compress_ratio > 1 and self.compress_ratio != 4:
                # 用 qr（或 q）作为 dependency,prefetch 与后续 RoPE / wkv / scatter 并行
                with torch.npu.stream(self.compressor_stream):
                    self.compressor_stream.wait_event(main_stream_done_event)
                    torch_npu.npu_prefetch(
                        self.compressor_wkv.weight,
                        qr,                                      # 锚点:在 qr 产出后开始预取
                        self.compressor_wkv.weight.numel() *
                        self.compressor_wkv.weight.element_size(),
                        0,                                       # offset
                    )
                    torch_npu.npu_prefetch(self.compressor_wgate.weight, qr,
                                    self.compressor_wgate.weight.numel() *
                                    self.compressor_wgate.weight.element_size())

            coff = 2 if self.compressor_overlap else 1

            # 等待 prefetch 完成（如果启用了 prefetch）
            if self.compressor_prefetch:
                torch.npu.current_stream().wait_stream(self.compressor_stream)
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
            _scatter_nd_update_asc(
                compressor_attn_cache.view(-1, compressed_kv.shape[-1]),
                compressor_attn_metadata.decode.slot_mapping.unsqueeze(-1),
                compressed_kv.view(-1, compressed_kv.shape[-1]))
        if self.compress_ratio <= 1:
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
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
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
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
            attn_output = torch.ops.custom.npu_sparse_attn_sharedkv(
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

        if self.compressor_prefetch:
            # ================================================================
            # Prefetch:把后面 compressor 要用的权重提前搬进 L2。
            # 位置选择理由:
            #   - wq_b 的 MatMul 已完成,它对 L2 的抢占已结束,不会立刻把我们挤掉;
            #   - 后面的 partial_rotary_mul + rotate_activation + block_table
            #     选择,作用在 q 上,不碰 indexcom_* 权重,形成干净的 overlap 窗口;
            #   - 搬完的时刻刚好接近 compressor 启动时刻,L2 命中率最高。
            # 依赖锚点用 q(已产出,确保合法);搬运起点由本行位置决定。
            # ap e/ norm 很小,放不放都行;这里一起搬,代价可忽略。
            # ================================================================
            # 记录 q 完成事件，供 compressor_stream 使用
            q_ready_event = torch.npu.current_stream().record_event()
            with torch.npu.stream(self.compressor_stream):
                self.compressor_stream.wait_event(q_ready_event)
                torch_npu.npu_prefetch(self.indexcom_wkv.weight,  q, self.indexcom_wkv.weight.numel()  * self.indexcom_wkv.weight.element_size())
                torch_npu.npu_prefetch(self.indexcom_wgate.weight, q, self.indexcom_wgate.weight.numel() * self.indexcom_wgate.weight.element_size())
                torch_npu.npu_prefetch(self.indexcom_norm.weight, q, self.indexcom_norm.weight.numel() * self.indexcom_norm.weight.element_size())
    
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

        # 等待 prefetch 完成（如果启用了 prefetch）
        if self.compressor_prefetch:
            torch.npu.current_stream().wait_stream(self.compressor_stream)
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

        if dst_type is torch.int8 and q.shape[-1] == 128:
            import custom_ops
            q, q_scale = torch.ops.custom.npu_dynamic_quant_v3(q)
        else:
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
                _scatter_nd_update_asc(
                    indexer_kv_cache.view(-1, kv.shape[-1]),
                    indexer_kv_scale_metadata.prefill.slot_mapping.unsqueeze(-1),
                    kv.view(-1, kv.shape[-1]))
                _scatter_nd_update_asc(
                    indexer_scale_cache.view(-1, kv_scale.shape[-1]),
                    indexer_kv_scale_metadata.prefill.slot_mapping.unsqueeze(-1),
                    kv_scale.view(-1, kv_scale.shape[-1]))
        else:
            assert indexer_kv_scale_metadata.decode is not None
            if kv is not None:
                _scatter_nd_update_asc(
                    indexer_kv_cache.view(-1, kv.shape[-1]),
                    indexer_kv_scale_metadata.decode.slot_mapping.unsqueeze(-1),
                    kv.view(-1, kv.shape[-1]))
                _scatter_nd_update_asc(
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
        if self.compressor_prefetch:
            main_stream_done_event = torch.npu.current_stream().record_event()
        
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
            sparse_count=self.index_topk,
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            return_value=False)
        if self.compressor_prefetch:
            # 用 qr（或 q）作为 dependency,prefetch 与后续 RoPE / wkv / scatter 并行
            with torch.npu.stream(self.compressor_stream):
                self.compressor_stream.wait_event(main_stream_done_event)
                torch_npu.npu_prefetch(
                    self.compressor_wkv.weight,
                    qr,                                      # 锚点:在 qr 产出后开始预取
                    self.compressor_wkv.weight.numel() *
                    self.compressor_wkv.weight.element_size(),
                    0,                                       # offset
                )
                torch_npu.npu_prefetch(self.compressor_wgate.weight, qr,
                                self.compressor_wgate.weight.numel() *
                                self.compressor_wgate.weight.element_size())
        return topk_idxs

