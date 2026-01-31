import torch
import torch_npu
import torch.nn.functional as F

from tests.ut.base import TestBase
from vllm_ascend.utils import enable_custom_op

import torch
import torch_npu
import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F
enable_custom_op()

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

class TestQLI(TestBase):
    def setUp(self):
        torch.manual_seed(42)

    def test_qli(self):
        b = 4
        s = 4
        hc = 4
        d = 4096
        eps = 1e-6

        np.random.seed(0)

        # start run custom ops
        print(f'======================== PTA eager BEGIN ========================')
        n1 = 64
        n2 = 1
        d = 128
        block_size = 128
        layout_key = "PA_BSND"
        layout_query = "BSND"
        query_quant_mode = 0
        key_quant_mode = 0
        np.random.seed(0)
        # -------------
        b = 24
        t = None
        s1 = 4
        s2 = 512
        act_seq_q = None
        act_seq_k = None
        sparse_mode = 0
        sparse_count = 512
        cmp_ratio = 1
        max_block_table_num = (s2 + block_size - 1) // block_size
        block_table = torch.tensor([range(b * max_block_table_num)], dtype = torch.int32).reshape(b, -1)
        key = torch.tensor(np.random.uniform(-128, 127, (b * max_block_table_num, block_size, n2, d))).to(torch.int8)
        key_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b * max_block_table_num, block_size, n2)))
        key_dequant_scale = key_dequant_scale.to(torch.float16)
        query = torch.tensor(np.random.uniform(-128, 127, (b, s1, n1, d))).to(torch.int8)
        query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b, s1, n1))).to(torch.float16)
        weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.float16)
        actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32) \
                                    if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32)
        actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32) \
                                    if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32)
        max_seqlen_q = actual_seq_lengths_query.max().item()
        max_seqlen_k = actual_seq_lengths_key.max().item()
        metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata (
                                        actual_seq_lengths_query = actual_seq_lengths_query.npu() if actual_seq_lengths_query is not None else torch.tensor([]).npu(),
                                        actual_seq_lengths_key = actual_seq_lengths_key.npu() if actual_seq_lengths_key is not None else torch.tensor([]).npu(),
                                        num_heads_q = n1,
                                        num_heads_k = n2,
                                        head_dim = d,
                                        query_quant_mode = query_quant_mode, 
                                        key_quant_mode = key_quant_mode,
                                        batch_size = b, 
                                        max_seqlen_q = max_seqlen_q,
                                        max_seqlen_k = max_seqlen_k,  
                                        layout_query = layout_query, 
                                        layout_key = layout_key,
                                        sparse_count = sparse_count, 
                                        sparse_mode = sparse_mode, 
                                        pre_tokens = (1<<63)-1, 
                                        next_tokens = (1<<63)-1, 
                                        cmp_ratio = cmp_ratio,
                                        device = 'npu:0')

        npu_out,_ = torch.ops._C_ascend.npu_quant_lightning_indexer(query.npu(), key.npu(), weights.npu(), query_dequant_scale.npu(),
                                                        key_dequant_scale.npu(),
                                                        actual_seq_lengths_query=actual_seq_lengths_query.npu(),
                                                        actual_seq_lengths_key=actual_seq_lengths_key.npu(),
                                                        block_table=block_table.npu(),
                                                        metadata = metadata,
                                                        query_quant_mode=query_quant_mode,
                                                        key_quant_mode=key_quant_mode,
                                                        layout_query=layout_query,
                                                        layout_key=layout_key, sparse_count=sparse_count,
                                                        sparse_mode=sparse_mode, pre_tokens=(1<<63)-1,
                                                        next_tokens=(1<<63)-1, cmp_ratio=cmp_ratio)
        print(f'======================== PTA eager FINISH ========================')


    def test_qli_tnd(self):
        np.random.seed(0)
        # start run custom ops
        print(f'======================== PTA eager BEGIN ========================')
        n1 = 64
        n2 = 1
        d = 128
        block_size = 128
        layout_key = "PA_BSND"
        layout_query = "TND"
        query_quant_mode = 0
        key_quant_mode = 0

        # -------------
        b = 1
        s1 = 117
        s2 = 19
        act_seq_q = [117]
        act_seq_k = [117]
        sparse_mode = 3
        sparse_count = 512
        cmp_ratio = 4
        block_count = 281344
        max_block_num_per_seq = 19
        block_table = torch.tensor([range(b * max_block_num_per_seq)], dtype = torch.int32).reshape(b, -1)
        key = torch.tensor(np.random.uniform(-128, 127, (block_count, block_size, n2, d))).to(torch.int8)
        key_dequant_scale = torch.tensor(np.random.uniform(0, 10, (block_count, block_size, n2)))
        key_dequant_scale = key_dequant_scale.to(torch.float16)
        query = torch.tensor(np.random.uniform(-128, 127, (b*s1, n1, d))).to(torch.int8)
        query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b*s1, n1))).to(torch.float16)
        weights = torch.tensor(np.random.uniform(0, 0.01, (b*s1, n1))).to(torch.float16)
        actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32) \
                                    if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32)
        actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32) \
                                    if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32)
        max_seqlen_q = actual_seq_lengths_query.max().item()
        max_seqlen_k = actual_seq_lengths_key.max().item()
        actual_seq_lengths_query=actual_seq_lengths_query.npu() if actual_seq_lengths_query is not None else torch.tensor([]).npu()
        actual_seq_lengths_key=actual_seq_lengths_key.npu() if actual_seq_lengths_key is not None else torch.tensor([]).npu()
        
        print("==== npu_quant_lightning_indexer_metadata args before====")

        def _print_tensor(name, t):
            if isinstance(t, torch.Tensor):
                print(f"{name}: "
                    f"dtype={t.dtype}, "
                    f"shape={tuple(t.shape)}, "
                    f"device={t.device}, "
                    f"value_sample={t.flatten()[:8]}")
            else:
                print(f"{name}: {t} (type={type(t)})")

        _print_tensor("actual_seq_lengths_query", actual_seq_lengths_query)
        _print_tensor("actual_seq_lengths_key", actual_seq_lengths_key)

        print(f"num_heads_q = {n1}")
        print(f"num_heads_k = {n2}")
        print(f"head_dim = {d}")
        print(f"query_quant_mode = {query_quant_mode}")
        print(f"key_quant_mode = {key_quant_mode}")
        print(f"batch_size = {b}")
        print(f"max_seqlen_q = {max_seqlen_q}")
        print(f"max_seqlen_k = {max_seqlen_k}")
        print(f"layout_query = {layout_query}")
        print(f"layout_key = {layout_key}")
        print(f"sparse_count = {sparse_count}")
        print(f"sparse_mode = {sparse_mode}")
        print(f"pre_tokens = {(1<<63)-1}")
        print(f"next_tokens = {(1<<63)-1}")
        print(f"cmp_ratio = {cmp_ratio}")
        print(f"device = {'npu:0'}")

        print("==============================================")
        metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata (
                                        actual_seq_lengths_query = actual_seq_lengths_query,
                                        actual_seq_lengths_key = actual_seq_lengths_key,
                                        num_heads_q = n1,
                                        num_heads_k = n2,
                                        head_dim = d,
                                        query_quant_mode = query_quant_mode, 
                                        key_quant_mode = key_quant_mode,
                                        batch_size = b, 
                                        max_seqlen_q = max_seqlen_q,
                                        max_seqlen_k = max_seqlen_k,  
                                        layout_query = layout_query, 
                                        layout_key = layout_key,
                                        sparse_count = sparse_count, 
                                        sparse_mode = sparse_mode, 
                                        pre_tokens = (1<<63)-1, 
                                        next_tokens = (1<<63)-1, 
                                        cmp_ratio = cmp_ratio,
                                        device = 'npu:0')
        print("==== npu_quant_lightning_indexer_metadata args ====")

        def _print_tensor(name, t):
            if isinstance(t, torch.Tensor):
                print(f"{name}: "
                    f"dtype={t.dtype}, "
                    f"shape={tuple(t.shape)}, "
                    f"device={t.device}, "
                    f"value_sample={t.flatten()[:8]}")
            else:
                print(f"{name}: {t} (type={type(t)})")

        _print_tensor("actual_seq_lengths_query", actual_seq_lengths_query)
        _print_tensor("actual_seq_lengths_key", actual_seq_lengths_key)

        print(f"num_heads_q = {n1}")
        print(f"num_heads_k = {n2}")
        print(f"head_dim = {d}")
        print(f"query_quant_mode = {query_quant_mode}")
        print(f"key_quant_mode = {key_quant_mode}")
        print(f"batch_size = {b}")
        print(f"max_seqlen_q = {max_seqlen_q}")
        print(f"max_seqlen_k = {max_seqlen_k}")
        print(f"layout_query = {layout_query}")
        print(f"layout_key = {layout_key}")
        print(f"sparse_count = {sparse_count}")
        print(f"sparse_mode = {sparse_mode}")
        print(f"pre_tokens = {(1<<63)-1}")
        print(f"next_tokens = {(1<<63)-1}")
        print(f"cmp_ratio = {cmp_ratio}")
        print(f"device = {'npu:0'}")

        print("==============================================")

        metadata = metadata.reshape(-1,8)
        for i in range(int(1024/8)):
            print(metadata[i, :])
        
        print("==============================================")

        npu_out,_ = torch.ops._C_ascend.npu_quant_lightning_indexer(query.npu(), key.npu(), weights.npu(), query_dequant_scale.npu(),
                                                        key_dequant_scale.npu(),
                                                        actual_seq_lengths_query=actual_seq_lengths_query.npu(),
                                                        actual_seq_lengths_key=actual_seq_lengths_key.npu(),
                                                        block_table=block_table.npu(),
                                                        metadata = metadata,
                                                        query_quant_mode=query_quant_mode,
                                                        key_quant_mode=key_quant_mode,
                                                        layout_query=layout_query,
                                                        layout_key=layout_key, sparse_count=sparse_count,
                                                        sparse_mode=sparse_mode, pre_tokens=(1<<63)-1,
                                                        next_tokens=(1<<63)-1, cmp_ratio=cmp_ratio)
        print(f'======================== PTA eager FINISH ========================')