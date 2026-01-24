#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import test_sas
import torch
import torch_npu
# import check_valid_param
import pytest
import random
import numpy as np
import math
import custom_ops as ops

DATA_RANGE_LEFT = -10
DATA_RANGE_RIGHT = 10

np.random.seed(42)
torch.manual_seed(42)

class GeneralizedSFA:
    def __init__(self, layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K,
                 block_num1, block_num2, block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio,
                 ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right):
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.q_type = q_type
        self.ori_kv_type = ori_kv_type
        self.cmp_kv_type = cmp_kv_type
        self.B = B
        self.S1 = S1
        self.T1 = T1
        self.N1 = N1
        self.N2 = N2
        self.D = D
        self.K = K
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.block_size1 = block_size1
        self.block_size2 = block_size2
        self.cu_seqlens_q = cu_seqlens_q
        self.seqused_kv = seqused_kv
        self.softmax_scale = softmax_scale
        self.cmp_ratio = cmp_ratio
        self.ori_mask_mode = ori_mask_mode
        self.cmp_mask_mode = cmp_mask_mode
        self.ori_win_left = ori_win_left
        self.ori_win_right = ori_win_right

    def calulate_by_bnsd(self, q_bnsd, ori_k_bnsd, cu_seqlens_q, seqused_kv, sinks, template_idx, cmp_k_bnsd=None,
                         cmp_sparse_indices_bnsd=None):
        attn_out = torch.zeros(q_bnsd.shape, dtype=q_bnsd.dtype)
        B = q_bnsd.shape[0]
        act_q = prefix_sum_to_original(cu_seqlens_q)
        G = int(self.N1 / self.N2)

        for i_B in range(B):
            print(f"i_B = {i_B}/{B}")
            cur_act_q = act_q[i_B]
            cur_ori_act_kv = seqused_kv[i_B]
            if template_idx == 1 or template_idx == 2:
                cur_cmp_act_kv = math.floor(cur_ori_act_kv / self.cmp_ratio)
            else:
                cur_cmp_act_kv = None
            for i_N2 in range(self.N2):
                print(f"    i_N2 = {i_N2}/{self.N2}")
                cur_sinks = sinks[i_N2 * G:(i_N2 + 1) * G]
                cur_sinks_expand = cur_sinks.unsqueeze(1)
                for i_S1 in range(cur_act_q):
                    milestones = [int(cur_act_q * pct / 100) for pct in range(10, 101, 10)]
                    milestones = list(dict.fromkeys(milestones))
                    if i_S1 in milestones:
                        current_pct = (i_S1 / cur_act_q) * 100
                        print(f"      进度：{current_pct:.1f}% | 步数：{i_S1:>{len(str(cur_act_q))}}/{cur_act_q}")
                    if i_S1 < cur_act_q - cur_ori_act_kv:  # 根据 ori_kv 判断行无效
                        attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = torch.zeros([G, self.D], dtype=torch.float)
                        continue

                    q_curr = q_bnsd[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :]

                    if template_idx == 2:
                        topk_id = cmp_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        empty_flag, cur_cmp_k = self.gather_cmp_kv(cmp_k_bnsd, topk_id, i_B, i_N2, i_S1, cur_ori_act_kv,
                                                                   cur_act_q)
                    elif template_idx == 1:
                        threshold = 0
                        if self.cmp_mask_mode == 3:
                            threshold = math.floor((cur_ori_act_kv - cur_act_q + i_S1 + 1) / (self.cmp_ratio))
                        if threshold == 0:
                            empty_flag = True
                        else:
                            empty_flag = False
                        cur_cmp_k = cmp_k_bnsd[i_B, i_N2, :threshold, :]
                    else:
                        empty_flag = True
                        cur_cmp_k = []

                    if self.ori_mask_mode == 4:
                        ori_threshold = cur_ori_act_kv - cur_act_q + i_S1 + 1
                        ori_win_end = ori_threshold + self.ori_win_right
                        ori_win_start = max(ori_threshold - self.ori_win_left - 1, 0)

                    cur_ori_k_bnsd = ori_k_bnsd[i_B, i_N2, ori_win_start:ori_win_end, :]
                    if empty_flag:
                        k_concat = cur_ori_k_bnsd
                    else:
                        k_concat = torch.concat([cur_ori_k_bnsd, cur_cmp_k], dim=0)

                    q_curr_fp32 = q_curr.to(dtype=torch.float32)
                    k_concat_fp32 = k_concat.to(dtype=torch.float32)
                    v_concat_fp32 = k_concat_fp32.clone()

                    mm1_res = torch.matmul(q_curr_fp32, k_concat_fp32.T)
                    scale_res = mm1_res * self.softmax_scale
                    softmax_res = self.sinks_softmax(scale_res, cur_sinks_expand)
                    mm2_res = torch.matmul(softmax_res, v_concat_fp32)
                    # mm1_res降精度引入误差，以输入全1、ori_s2=128、cmp_s2=32、s1=1、scale_value=0.01为例
                    # softmax之后  1/(160+math.exp(1-5.12)) = 0.006249365513072936
                    # mm2之后 0.006249365513072936*160 = 0.9998984820916699
                    # 实际softmax之后转bf16为 0.006256103515625
                    # 最终结果 0.006256103515625*160 = 1.0009765625
                    # import pdb; pdb.set_trace()
                    attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = mm2_res.to(dtype=q_bnsd.dtype)
        return attn_out

    def gather_cmp_kv(self, k_tensor, topk_id, i_B, i_N2, i_S1, cur_ori_act_kv, cur_act_q, sparse_block_size=1):
        s2_sparse = list()
        cur_cmp_act_kv = math.floor(cur_ori_act_kv / self.cmp_ratio)
        threshold = 0
        if self.cmp_mask_mode == 3:
            threshold = math.floor((cur_ori_act_kv - cur_act_q + i_S1 + 1) / self.cmp_ratio)
        valid_count = min(self.K, math.ceil(threshold / sparse_block_size))
        for i_valid in range(valid_count):
            cur_topk_id = topk_id[i_valid]

            if cur_topk_id == -1:
                break
            begin_idx = cur_topk_id * sparse_block_size
            end_idx = begin_idx + sparse_block_size if begin_idx + sparse_block_size <= cur_cmp_act_kv else cur_cmp_act_kv
            if begin_idx >= threshold:
                continue
            if end_idx <= threshold:
                s2_sparse.extend(np.arange(begin_idx, end_idx))
            else:
                s2_sparse.extend(np.arange(begin_idx, threshold))

        empty_flag = False
        if len(s2_sparse) == 0:
            cur_cmp_k = []
            empty_flag = True
        else:
            cur_cmp_k = k_tensor[i_B, i_N2, s2_sparse, :]
        return empty_flag, cur_cmp_k

    def sinks_softmax(self, x, sinks):  # [G, S2] [G, 1]
        x = x.to(dtype=torch.float)
        x_concat = torch.cat([x, sinks], dim=1)
        x_max = x_concat.max(dim=-1, keepdims=True)[0]
        x_sub = x - x_max
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True) + torch.exp(sinks - x_max)
        ans = y / x_sum
        return ans

    def trans_shape_to_bnsd(self, tensor, shape, layout, act_seq=None):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            N = shape[2]
            D = shape[3]
            tensor = tensor.permute(0, 2, 1, 3)
            return tensor, [B, N, S, D]
        elif layout in ["TND"]:
            T = shape[0]
            N = shape[1]
            D = shape[2]
            B = len(act_seq) - 1  # TND act_q is cumulative
            max_s1 = get_max_adjacent_diff(act_seq)
            act_seq_per_batch = prefix_sum_to_original(act_seq)
            new_tensor = torch.zeros((B, N, max_s1, D), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:cur_act_seq, :] = tensor[t_start:t_end, n_index, :]
                t_start += cur_act_seq
            return new_tensor, [B, N, max_s1, D]
        else:
            return tensor, shape

    def trans_bnsd_to_target_layout(self, tensor, layout, act_seq=None):
        if layout in ["BSND"]:
            output = tensor.permute(0, 2, 1, 3).contiguous()
            return output
        elif layout in ["TND"]:
            T = act_seq[-1]
            B = tensor.shape[0]
            N = tensor.shape[1]
            D = tensor.shape[3]
            output = torch.zeros((T, N, D), dtype=torch.float)
            t_start = 0
            act_seq_per_batch = prefix_sum_to_original(act_seq)
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N):
                    output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :cur_act_seq, :]
                t_start += cur_act_seq
            return output
        else:
            return tensor

    def forward(self, q, ori_k_bnsd, cu_seqlens_q, seqused_kv, sinks, template_idx, cmp_k_bnsd=None,
                cmp_sparse_indices=None):
        q_bnsd, q_bnsd_shape = self.trans_shape_to_bnsd(q, q.shape, self.layout_q, cu_seqlens_q)

        if template_idx == 2:
            cmp_sparse_indices_bnsd, cmp_sparse_indices_bnsd_shape = self.trans_shape_to_bnsd(cmp_sparse_indices,
                                                                     cmp_sparse_indices.shape, self.layout_q,
                                                                     cu_seqlens_q)
        else:
            cmp_sparse_indices_bnsd = None
        attn_out = self.calulate_by_bnsd(q_bnsd, ori_k_bnsd, cu_seqlens_q, seqused_kv, sinks, template_idx,
                   cmp_k_bnsd, cmp_sparse_indices_bnsd)

        attn_out = self.trans_bnsd_to_target_layout(attn_out, self.layout_q, cu_seqlens_q)
        return attn_out

def prefix_sum_to_original(cu_seqlens_q):
    """
    从前缀和张量反向计算出原始的非前缀和张量（替代原列表逻辑）

    Args:
        cu_seqlens_q (torch.Tensor): 形状为 [B+1] 的一维前缀和张量（元素为数字类型，如int/float）

    Returns:
        torch.Tensor: 原始的非前缀和张量，形状为 [B]（与原列表长度一致）

    Raises:
        TypeError: 输入非tensor/非一维tensor
        ValueError: tensor长度<2（无法计算差值）
    """
    # 1. 基础类型校验：必须是torch.Tensor
    if not isinstance(cu_seqlens_q, torch.Tensor):
        raise TypeError(f"输入必须是torch.Tensor，当前类型：{type(cu_seqlens_q)}")

    # 2. 维度校验：必须是一维tensor（原列表对应一维）
    if cu_seqlens_q.ndim != 1:
        raise TypeError(f"输入必须是一维tensor，当前维度：{cu_seqlens_q.ndim}，形状：{cu_seqlens_q.shape}")

    # 3. 长度校验（前缀和tensor至少需2个元素才能反向计算）
    if len(cu_seqlens_q) < 2:
        raise ValueError(f"前缀和tensor长度需≥2，当前长度：{len(cu_seqlens_q)}")

    # 4. 核心逻辑：计算相邻元素差值（用tensor向量化运算替代循环，效率更高）
    # 原理：original_val[i] = cu_seqlens_q[i+1] - cu_seqlens_q[i]
    # 切片实现：cu_seqlens_q[1:] 取第2个到最后一个元素，cu_seqlens_q[:-1] 取第1个到倒数第2个元素
    original_tensor = cu_seqlens_q[1:] - cu_seqlens_q[:-1]

    return original_tensor

def get_max_adjacent_diff(cu_seqlens_q):
    """
    计算前缀和列表中相邻元素（后-前）的最大差值

    Args:
        cu_seqlens_q (list): 长度为 B+1 的前缀和列表

    Returns:
        float/int: 相邻元素的最大差值；若列表长度<2，返回 None
    """
    # 边界检查：列表长度不足2时无相邻元素
    if len(cu_seqlens_q) < 2:
        return None

    # 初始化最大差值为第一个相邻对的差值
    max_diff = cu_seqlens_q[1] - cu_seqlens_q[0]

    # 遍历所有相邻元素对（从第2对开始）
    for i in range(1, len(cu_seqlens_q)-1):
        current_diff = cu_seqlens_q[i+1] - cu_seqlens_q[i]
        # 更新最大差值
        if current_diff > max_diff:
            max_diff = current_diff

    return max_diff

def gen_cmp_sparse_indices_bsnd(cmp_ratio, B, S1, N2, K, seqused_kv, cmp_mask_mode):
    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    cmp_sparse_indices = torch.full((B, S1, N2, K), fill_value=-1, dtype=torch.int32)
    for i_B in range(B):
        cur_act_kv = seqused_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(S1):
                if cmp_mask_mode == 3:
                    cur_valid_s2_max = math.floor((cur_act_kv - S1 + i_S1 + 1) / cmp_ratio)
                else:
                    raise ValueError(f"cmp_mask_mode only support 3, which is {cmp_mask_mode}")

                valid_blocks_max = max(0, cur_valid_s2_max)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                cmp_sparse_indices[i_B, i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
    return cmp_sparse_indices

def gen_cmp_sparse_indices_tnd(cmp_ratio, B, T1, N2, K, cu_seqlens_q, seqused_kv, cmp_mask_mode):
    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    cmp_sparse_indices = torch.full((T1, N2, K), fill_value=-1, dtype=torch.int32)
    for i_B in range(B):
        cur_act_q = cu_seqlens_q[i_B + 1] - cu_seqlens_q[i_B]
        s1_prefix = cu_seqlens_q[i_B]
        cur_act_kv = seqused_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                if cmp_mask_mode == 3:
                    cur_valid_s2_max = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio)
                valid_blocks_max = max(0, cur_valid_s2_max)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                cmp_sparse_indices[s1_prefix + i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
    return cmp_sparse_indices

# def gen_ori_kv(params):
#     layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K, block_num1, block_num2, \
#     block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, \
#     ori_win_left, ori_win_right = params
def gen_ori_kv(ori_kv_type, B, N2, D, block_num1, block_size1, seqused_kv):
    ori_max_s2 = max(seqused_kv)
    ori_max_block_num_per_batch = math.ceil(ori_max_s2 / block_size1)

    ori_k_bnsd = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
                              (B, N2, ori_max_s2, D))).to(ori_kv_type)
    ori_block_num_per_batch = []
    ori_block_num_sum = 0

    for cur_ori_act_kv in seqused_kv:
        cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
        ori_block_num_per_batch.append(cur_ori_kv_block_num)
        ori_block_num_sum += cur_ori_kv_block_num

    if block_num1 < ori_block_num_sum:
        raise ValueError(f"ori_kv actual_block_num < needed_block_num, which is {block_num1 < ori_block_num_sum}")

    ori_block_id_list = np.arange(block_num1)
    ori_block_id_list = np.random.permutation(ori_block_id_list).astype(np.int32)
    cur_block_id = 0
    ori_block_table = np.full((B, ori_max_block_num_per_batch), fill_value=-1, dtype=np.int32)
    batch_idx = 0
    for cur_block_id_threshold in ori_block_num_per_batch:
        for i_block_id in range(cur_block_id_threshold):
            ori_block_table[batch_idx][i_block_id] = ori_block_id_list[cur_block_id]
            cur_block_id += 1
        batch_idx += 1

    # [B, S, N, D] expand to [B, ori_max_block_num_per_batch * block_size1, N, D]
    ori_k_expand = torch.zeros((B, N2, ori_max_block_num_per_batch * block_size1, D), dtype=ori_kv_type)
    ori_k_expand[:, :, :ori_max_s2, :] = ori_k_bnsd
    ori_k_in_pa_shape = torch.zeros((block_num1, block_size1, N2, D), dtype=ori_kv_type)

    for i_B in range(B):
        for i_block, cur_block_id in enumerate(ori_block_table[i_B]):
            block_start_pos = i_block * block_size1
            if cur_block_id == -1:
                continue
            else:
                for i_N2 in range(N2):
                    ori_k_in_pa_shape[cur_block_id, :, i_N2, :] = \
                        ori_k_expand[i_B, i_N2, block_start_pos:block_start_pos + block_size1, :]

    ori_block_table = torch.tensor(ori_block_table).to(torch.int32)

    return ori_k_in_pa_shape, ori_block_table, ori_k_bnsd

def gen_cmp_kv(layout_q, cmp_kv_type, B, S1, T1, N2, D, K, block_num2, block_size2, cu_seqlens_q, seqused_kv, cmp_ratio,
               cmp_mask_mode, template_idx):
    if cmp_ratio is None:
        raise ValueError(f"cmp_ratio can't be None")

    if template_idx == 1:
        if cmp_ratio != 128:
            raise ValueError(f"unsupported cmp_ratio {cmp_ratio} in template_idx {template_idx}")
    elif template_idx == 2:
        if cmp_ratio != 4:
            raise ValueError(f"unsupported cmp_ratio {cmp_ratio} in template_idx {template_idx}")
    else:
        raise ValueError(f"unsupported template_idx: {template_idx}")


    ori_max_s2 = max(seqused_kv)
    cmp_max_s2 = math.floor(ori_max_s2 / cmp_ratio)
    cmp_max_block_num_per_batch = math.ceil(cmp_max_s2 / block_size2)

    cmp_k_bnsd = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
                              (B, N2, cmp_max_s2, D))).to(cmp_kv_type)
    cmp_block_num_per_batch = []
    cmp_block_num_sum = 0
    for cur_ori_act_kv in seqused_kv:
        cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
        cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
        cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
        cmp_block_num_sum += cur_cmp_kv_block_num
    if block_num2 < cmp_block_num_sum:
        raise ValueError(f"cmp_kv actual_block_num < needed_block_num, which is {block_num2 < cmp_block_num_sum}")

    cmp_block_id_list = np.arange(block_num2)
    cmp_block_id_list = np.random.permutation(cmp_block_id_list).astype(np.int32)
    cur_block_id = 0
    cmp_block_table = np.full((B, cmp_max_block_num_per_batch), fill_value=-1, dtype=np.int32)
    batch_idx = 0
    for cur_block_id_threshold in cmp_block_num_per_batch:
        for i_block_id in range(cur_block_id_threshold):
            cmp_block_table[batch_idx][i_block_id] = cmp_block_id_list[cur_block_id]
            cur_block_id += 1
        batch_idx += 1

    cmp_k_expand = torch.zeros((B, N2, cmp_max_block_num_per_batch * block_size2, D), dtype=cmp_kv_type)
    cmp_k_expand[:, :, :cmp_max_s2, :] = cmp_k_bnsd
    cmp_k_in_pa_shape = torch.zeros((block_num2, block_size2, N2, D), dtype=cmp_kv_type)
    for i_B in range(B):
        for i_block, cur_block_id in enumerate(cmp_block_table[i_B]):
            block_start_pos = i_block * block_size2
            if cur_block_id == -1:
                continue
            else:
                for i_N2 in range(N2):
                    cmp_k_in_pa_shape[cur_block_id, :, i_N2, :] = \
                        cmp_k_expand[i_B, i_N2, block_start_pos:block_start_pos + block_size2, :]

    # generate cmp_sparse_indices
    if template_idx == 2:
        if layout_q == "BSND":
            cmp_sparse_indices = gen_cmp_sparse_indices_bsnd(cmp_ratio, B, S1, N2, K, seqused_kv, cmp_mask_mode)
        elif layout_q == "TND":
            cmp_sparse_indices = gen_cmp_sparse_indices_tnd(cmp_ratio, B, T1, N2, K, cu_seqlens_q, seqused_kv,
                                 cmp_mask_mode)
    else:
        cmp_sparse_indices = None
    cmp_block_table = torch.tensor(cmp_block_table).to(torch.int32)
    return cmp_k_in_pa_shape, cmp_sparse_indices, cmp_block_table, cmp_k_bnsd

def test_sas_process(params):
    layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K, block_num1, block_num2, \
    block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, \
    ori_win_left, ori_win_right = params

    if len(seqused_kv) != B:
        raise ValueError(f"len(seqused_kv) != B, which is {len(seqused_kv)} != {B}")
    else:
        pass

    ori_max_s2 = max(seqused_kv)
    cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32)
    seqused_kv = torch.tensor(seqused_kv).to(torch.int32)

    # generate q
    if layout_q == "BSND":
        q = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT, (B, S1, N1, D))).to(q_type)
        act_q = B * [S1]
    elif layout_q == "TND":
        q = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT, (T1, N1, D))).to(q_type)
        if len(cu_seqlens_q) != (B + 1):
            raise ValueError(f"len(cu_seqlens_q) != B + 1, which is {len(cu_seqlens_q)} != {B + 1}")
        else:
            act_q = prefix_sum_to_original(cu_seqlens_q)
            max_s1 = get_max_adjacent_diff(cu_seqlens_q)
    else:
        raise ValueError(f"layout_q is not support {layout_q}")

    # generate ori_kv/cmp_kv (only support PA_ND)
    if layout_kv == "PA_ND":
        pass
    else:
        raise ValueError(f"layout_kv is not support {layout_kv}")

    # 路由到三个算子的逻辑：
    template_idx = 0
    if K is None :
        if cmp_ratio is None:
            template_idx = 0  # SWA
        else:
            template_idx = 1  # CFA
    else:
        template_idx = 2  # SCFA

    ori_k_in_pa_shape, ori_block_table, ori_k_bnsd = gen_ori_kv(ori_kv_type, B, N2, D, block_num1, block_size1,
                                                                seqused_kv)
    if template_idx == 1 or template_idx == 2:
        cmp_k_in_pa_shape, cmp_sparse_indices, cmp_block_table, cmp_k_bnsd = gen_cmp_kv(layout_q, cmp_kv_type, B, S1,
                                                                                        T1, N2, D, K, block_num2,
                                                                                        block_size2, cu_seqlens_q,
                                                                                        seqused_kv, cmp_ratio,
                                                                                        cmp_mask_mode, template_idx)
    else:
        cmp_k_in_pa_shape = None
        cmp_sparse_indices = None
        cmp_block_table = None
        cmp_k_bnsd = None

    sinks = torch.tensor(np.random.uniform(DATA_RANGE_LEFT/10, DATA_RANGE_RIGHT/10, (N1))).to(torch.float)

    test_sas = GeneralizedSFA(layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K,
                              block_num1, block_num2, block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale,
                              cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right)
    cpu_result = test_sas.forward(q, ori_k_bnsd, cu_seqlens_q, seqused_kv, sinks, template_idx, cmp_k_bnsd,
                                  cmp_sparse_indices)

    if template_idx == 1 or template_idx == 2:
        cmp_k_in_pa_shape = cmp_k_in_pa_shape.npu()
        cmp_block_table = cmp_block_table.npu()
    if template_idx == 2:
        cmp_sparse_indices = cmp_sparse_indices.npu()

    q = q.npu()
    ori_k_in_pa_shape = ori_k_in_pa_shape.npu()
    ori_block_table = ori_block_table.npu()
    cu_seqlens_q = cu_seqlens_q.npu()
    seqused_kv = seqused_kv.npu()
    sinks = sinks.npu()

    # 获取最长 q
    if layout_q == 'TND':
        seq_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_seqlen_q = torch.max(seq_lens).item()
    else:
        max_seqlen_q = torch.max(cu_seqlens_q).item()

    if template_idx == 0:
        metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=N1,
            num_heads_kv=N2,
            head_dim=D,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            batch_size=B,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=ori_max_s2,
            ori_mask_mode=ori_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            has_ori_kv=ori_k_in_pa_shape != None,
            has_cmp_kv=cmp_k_in_pa_shape != None)
        npu_result = torch.ops.custom.npu_sparse_attn_sharedkv(q,
                                                               ori_kv=ori_k_in_pa_shape,
                                                               ori_block_table=ori_block_table,
                                                               cu_seqlens_q=cu_seqlens_q,
                                                               seqused_kv=seqused_kv,
                                                               sinks=sinks,
                                                               metadata=metadata,
                                                               softmax_scale=softmax_scale,
                                                               ori_mask_mode=ori_mask_mode,
                                                               ori_win_left=ori_win_left,
                                                               ori_win_right=ori_win_right,
                                                               layout_q=layout_q,
                                                               layout_kv=layout_kv)
    elif template_idx == 1:
        metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=N1,
            num_heads_kv=N2,
            head_dim=D,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            batch_size=B,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=ori_max_s2,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            has_ori_kv=ori_k_in_pa_shape != None,
            has_cmp_kv=cmp_k_in_pa_shape != None)
        npu_result = torch.ops.custom.npu_sparse_attn_sharedkv(q,
                                                               ori_kv=ori_k_in_pa_shape,
                                                               cmp_kv=cmp_k_in_pa_shape,
                                                               ori_block_table=ori_block_table,
                                                               cmp_block_table=cmp_block_table,
                                                               cu_seqlens_q=cu_seqlens_q,
                                                               seqused_kv=seqused_kv,
                                                               sinks=sinks,
                                                               metadata=metadata,
                                                               softmax_scale=softmax_scale,
                                                               cmp_ratio=cmp_ratio,
                                                               ori_mask_mode=ori_mask_mode,
                                                               cmp_mask_mode=cmp_mask_mode,
                                                               ori_win_left=ori_win_left,
                                                               ori_win_right=ori_win_right,
                                                               layout_q=layout_q,
                                                               layout_kv=layout_kv)
    else:
        metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=N1,
            num_heads_kv=N2,
            head_dim=D,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            batch_size=B,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=ori_max_s2,
            topk=K,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            has_ori_kv=ori_k_in_pa_shape != None,
            has_cmp_kv=cmp_k_in_pa_shape != None)
        npu_result = torch.ops.custom.npu_sparse_attn_sharedkv(q,
                                                                ori_kv=ori_k_in_pa_shape,
                                                                cmp_kv=cmp_k_in_pa_shape,
                                                                cmp_sparse_indices=cmp_sparse_indices,
                                                                ori_block_table=ori_block_table,
                                                                cmp_block_table=cmp_block_table,
                                                                cu_seqlens_q=cu_seqlens_q,
                                                                seqused_kv=seqused_kv,
                                                                sinks=sinks,
                                                                metadata=metadata,
                                                                softmax_scale=softmax_scale,
                                                                cmp_ratio=cmp_ratio,
                                                                ori_mask_mode=ori_mask_mode,
                                                                cmp_mask_mode=cmp_mask_mode,
                                                                ori_win_left=ori_win_left,
                                                                ori_win_right=ori_win_right,
                                                                layout_q=layout_q,
                                                                layout_kv=layout_kv)

    torch.npu.synchronize()

    return npu_result, cpu_result