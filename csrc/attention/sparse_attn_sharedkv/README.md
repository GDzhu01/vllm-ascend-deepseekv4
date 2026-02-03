# SparseAttnSharedkv

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明
- API功能：`SparseAttentionSharedKV`算子旨在完成以下公式描述的Attention计算，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention。

- 计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

    其中$\tilde{K}=\tilde{V}$为基于ori_kv、cmp_kv以及cmp_kv等入参控制的实际参与计算的 $KV$。

## 函数原型

```
custom.npu_sparse_attn_sharedkv(q, *, ori_kv=None, cmp_kv=None, ori_sparse_indices=None, cmp_sparse_indices=None, ori_block_table=None, cmp_block_table=None, cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None, seqused_q=None, seqused_kv=None, sinks=None, metadata=None, softmax_scale=0, cmp_ratio=0, ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127, ori_win_right=0, layout_q='BSND', layout_kv='PA_ND', return_softmax_lse=False) -> (Tensor, Tensor)
```

## 参数说明

> [!NOTE]  
>- q、ori_kv、cmp_kv参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Hidden Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- Q_S和S1表示q shape中的S，S2表示ori_kv shape中的S，S3表示cmp_kv shape中的S；Q\_N和N1表示num\_q\_heads，KV\_N和N2表示num\_ori_kv\_heads和num\_cmp_kv\_heads；T1表示q shape中的T，T2表示ori_kv shape中的T，T3表示cmp_kv shape中的输入样本序列长度的累加和。

-   **q**（`Tensor`）：必选参数，对应公式中的$Q$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`。`layout_query`为TND时shape为[T1,N1,D]，其中N1仅支持64。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

-   **ori_kv**（`Tensor`）：可选参数，对应公式中的$\tilde{K}和\tilde{V}$的一部分，为原始不经压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`，`layout_kv`为PA_ND时shape为[block\_num1, ori\_block\_size, KV\_N, D]，其中block\_num1为PageAttention时block总数，ori\_block\_size为一个block的token数，ori\_block\_size取值为16的倍数，最大支持1024，KV_N仅支持1。

-   **cmp_kv**（`Tensor`）：可选参数，对应公式中的$\tilde{K}和\tilde{V}$的一部分，为经过压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`，`layout_kv`为PA_ND时shape为[block\_num, cmp\_block\_size, KV\_N, D]，其中block\_num2为PageAttention时block总数，cmp\_block\_size为一个block的token数，cmp\_block\_size取值为16的倍数，最大支持1024。

-   **ori_sparse_indices**（`Tensor`）：可选参数，代表离散取oriKvCache的索引，不支持非连续，数据格式支持ND，数据类型支持`int32`。当`layout_query`为TND时，shape需要传入[Q\_T, KV\_N, K1]，其中K1为对`ori_kv`一次离散选取的token数，K1仅支持512。**目前暂不支持对ori_kv进行稀疏计算，故设置此参数无效**。

-   **cmp_sparse_indices**（`Tensor`）：可选参数，代表离散取cmpKvCache的索引，不支持非连续，数据格式支持ND，数据类型支持`int32`。当`layout_query`为TND时，shape需要传入[Q\_T, KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数，K2仅支持512。

-   **ori_block_table**（`Tensor`）：可选参数，表示PageAttention中oriKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2对应的block数量，即S2\_max / block\_size向上取整。

-   **cmp_block_table**（`Tensor`）：可选参数，表示PageAttention中cmpKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S3对应的block数量，即S3\_max / block\_size向上取整。

-   **cu_seqlens_q**（`Tensor`）：可选参数，当`layout_query`为TND时，表示不同Batch中`q`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。

-   **cu_seqlens_ori_kv**（`Tensor`）：可选参数，当`layout_kv`为TND时，表示不同Batch中`ori_kv`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。**目前layout_kv仅支持PA_ND，故设置此参数无效。**

-   **cu_seqlens_cmp_kv**（`Tensor`）：可选参数，当`layout_kv`为TND时，表示不同Batch中`cmp_kv`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。**目前layout_kv仅支持PA_ND，故设置此参数无效。**

-   **seqused_q**（`Tensor`）：可选参数，表示不同Batch中`q`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。目前暂只在layout_q为"BSND"时传入该参数。

-   **seqused_kv**（`Tensor`）：可选参数，表示不同Batch中`ori_kv`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。

-   **sinks**（`Tensor`）：可选参数，注意力下沉tensor，数据格式支持ND，数据类型支持`float32`，shape为[N1]。

-   **metadata**（`Tensor`）：可选参数，为aicpu算子（npu_sparse_attn_sharedkv_metadata）的分核结果，数据格式支持ND，数据类型支持`int32`，shape固定为[1024]。

-   **softmax_scale**（`double`）：可选参数，代表缩放系数，作为q与ori_kv和cmp_kv矩阵乘后Muls的scalar值，数据类型支持`float`，默认值为None，None表示softmax_scale值为1/sqrt(D)。
    
-   **cmp_ratio**（`int`）：可选参数，表示对ori_kv的压缩率，数据类型支持`int`，数据范围支持4/128，默认值为None。

-   **ori_mask_mode**（`int`）：可选参数，表示q和ori_kv计算的mask模式，仅支持输入默认值4，代表band模式的mask，数据类型支持`int`。

-   **cmp_mask_mode**（`int`）：可选参数，表示q和cmp_kv计算的mask模式，仅支持输入默认值3，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，数据类型支持`int`。

-   **ori_win_left**（`int`）：可选参数，表示q和ori_kv计算中q对过去token计算的数量，数据类型支持`int`，仅支持默认值127。

-   **ori_win_right**（`int`）：可选参数，表示q和ori_kv计算中q对未来token计算的数量，数据类型支持`int`，仅支持默认值0。

-   **layout_q**（`str`）：可选参数，用于标识输入q的数据排布格式，输入仅支持传入"TND"和BSND"。

-   **layout_kv**（`str`）：可选参数，用于标识输入`ori_kv`和`cmp_kv`的数据排布格式，输入仅支持传入"PA_ND"。

-   **return_softmax_lse**（`bool`）：可选参数，表示是否返回softmax_lse。True表示返回，False表示不返回；默认值为False。**目前暂不支持返回softmax_lse。**

## 返回值说明

-   **attention\_out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`和`float16`。当layout\_query为BSND时shape为[B,S1,N1,D]，当layout\_query为TND时shape为[T1,N1,D]。
-   **softmax\_lse**（`Tensor`）：可选输出，输出q乘ori_kv的结果先取max得到softmax_max，query乘key的结果减去softmax_max，再取exp，最后取sum，得到softmax_sum，最后对softmax_sum取log，再加上softmax_max得到的结果。数据类型支持`float`。当layout\_query为TND时shape为[N2,T1,N1/N2]。**目前softmax_lse输出为无效值。**

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持aclgraph模式。
-   参数q中的D和ori_kv、cmp_kv的D值相等为512。
-   参数q、ori_kv、cmp_kv的数据类型必须保持一致。

## 调用示例
- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    import math
    import custom_ops

    data_type = torch.bfloat16
    softmax_scale = 0.041666666666666664
    b = 4
    s1 = 128
    s2 = 8192
    n1 = 128
    n2 = 1
    dn = 512
    k = 512
    ori_block_size = 128
    cmp_block_size = 128
    s2_act = 4096
    cmp_ratio = 4
    ori_win_left = 127
    ori_win_right = 0
    layout_q = 'TND'
    layout_kv = 'PA_ND'
    ori_mask_mode = 4
    cmp_mask_mode = 3
    q = torch.tensor(np.random.uniform(-10, 10, (b*s1, n1, dn))).to(data_type).npu()

    cu_seqlens_q = torch.arange(0, (b + 1) * s1, step=s1).to(torch.int32).npu()
    seqused_kv = torch.tensor([s2]*b).to(torch.int32).npu()

    cmp_kv_len = s2_act // cmp_ratio
    idxs = random.sample(range(cmp_kv_len - s1 + 1),  k)
    cmp_sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, k). \
        to(torch.int32).npu()
        
    ori_block_num =  math.ceil(s2_act/ori_block_size) * b
    ori_block_table = torch.tensor(np.random.permutation(range(ori_block_num))).to(torch.int32).reshape(b, -1).npu()
    ori_kv = torch.tensor(np.random.uniform(-5, 10, (ori_block_num, ori_block_size, n2, dn))).to(data_type).npu()

    block_num2 =  math.ceil(cmp_kv_len/ori_block_size) * b
    cmp_block_table = torch.tensor(np.random.permutation(range(block_num2))).to(torch.int32).reshape(b, -1).npu()
    cmp_kv = torch.tensor(np.random.uniform(-5, 10, (block_num2, cmp_block_size, n2, dn))).to(data_type).npu()
    sinks = torch.rand(n1).to(torch.float32).npu()
    metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=n1,
        num_heads_kv=n2,
        head_dim=dn,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=b,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        cmp_topk=k,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=True,
        has_cmp_kv=True
    )
    attn_out, softmax_lse = torch.ops.custom.npu_sparse_attn_sharedkv(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=None,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
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
        layout_kv=layout_kv,
        return_softmax_lse=False)
    ```

- aclgraph模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    import math
    import torchair
    import custom_ops

    data_type = torch.bfloat16
    softmax_scale = 0.041666666666666664
    b = 4
    s1 = 128
    s2 = 8192
    n1 = 64
    n2 = 1
    dn = 512
    k = 512
    ori_block_size = 128
    cmp_block_size = 128
    s2_act = 4096
    cmp_ratio = 4
    ori_win_left = 127
    ori_win_right = 0
    layout_q = 'TND'
    layout_kv = 'PA_ND'
    ori_mask_mode = 4
    cmp_mask_mode = 3
    q = torch.tensor(np.random.uniform(-10, 10, (b*s1, n1, dn))).to(data_type).npu()

    cu_seqlens_q = torch.arange(0, (b + 1) * s1, step=s1).to(torch.int32).npu()
    seqused_kv = torch.tensor([s2]*b).to(torch.int32).npu()

    cmp_kv_len = s2_act // cmp_ratio
    idxs = random.sample(range(cmp_kv_len - s1 + 1),  k)
    cmp_sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, k). \
        to(torch.int32).npu()
        
    ori_block_num =  math.ceil(s2_act/ori_block_size) * b
    ori_block_table = torch.tensor(np.random.permutation(range(ori_block_num))).to(torch.int32).reshape(b, -1).npu()
    ori_kv = torch.tensor(np.random.uniform(-5, 10, (ori_block_num, ori_block_size, n2, dn))).to(data_type).npu()

    block_num2 =  math.ceil(cmp_kv_len/ori_block_size) * b
    cmp_block_table = torch.tensor(np.random.permutation(range(block_num2))).to(torch.int32).reshape(b, -1).npu()
    cmp_kv = torch.tensor(np.random.uniform(-5, 10, (block_num2, cmp_block_size, n2, dn))).to(data_type).npu()
    sinks = torch.rand(n1).to(torch.float32).npu()

    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q, max_seqlen_kv,
            topk, has_ori_kv, has_cmp_kv, q, ori_kv, cmp_kv, cmp_sparse_indices, ori_block_table, 
            cmp_block_table, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, sinks,
            ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv):
            metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                head_dim=head_dim,
                cu_seqlens_q=cu_seqlens_q,
                seqused_kv=seqused_kv,
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                cmp_topk=topk,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                has_ori_kv=has_ori_kv,
                has_cmp_kv=has_cmp_kv,
                device="npu:0"
            )
            npu_out = torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=ori_kv,
                cmp_kv=cmp_kv,
                ori_sparse_indices=None,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=ori_block_table,
                cmp_block_table=cmp_block_table,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=None,
                cu_seqlens_cmp_kv=None,
                seqused_q=None,
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
                layout_kv=layout_kv,
                return_softmax_lse=False)
            return npu_out

    mod = torch.compile(Network().npu(), backend=npu_backend, fullgraph=True)
    attn_out, softmax_lse = mod(
        num_heads_q=n1,
        num_heads_kv=n2,
        head_dim=dn,
        batch_size=b,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        topk=k,
        has_ori_kv=True,
        has_cmp_kv=True,
        q=q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        sinks=sinks,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv)
    ```

更多使用示例见[pytest示例](./tests/pytest/README.md)。