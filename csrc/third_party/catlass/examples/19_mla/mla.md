# CATLASS MLA

CATLASS MLA是基于CATLASS Gemm Api实现的亲和昇腾AtlasA2硬件的Flash-MLA算子，算子的结构可以分为以下几部分
* Tiling计算；
* Kernel实现，具体有两种实现，通用的[mla_kernel.cpp](./mla_kernel.cpp)以及特化的[mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp)；
* Kernel中依赖适合Flash-MLA运算的Block组件；
* 使用的Block组件依赖模板库提供的Tile组件。

## Tiling

Tiling计算的逻辑位于[mla.cpp](./mla.cpp)文件中，在调用算子前，需要准备好tiling计算所需的各项参数，赋值给MLAInfo结构体，并调用`GetMLATilingParam`函数。[mla.cpp](./mla.cpp)中提供了一个示例

```c++
// 准备Tiling计算所需的中间结构体以及Host侧空间
MLATiling::MLAInfo mlaInfo;
...
MLATiling::GetMLATilingParam(mlaInfo, blockDim, (uint32_t *)tilingHost);
```
`GetMLATilingParam`函数中，调用了两个函数`GetMLATilingCommon`与`GetMLATilingSpec`，分别对应了通用场景下和特化场景下的分核逻辑

## Kernel
本算子提供了两种Kernel实现:
1. 通用的[mla_kernel.cpp](./mla_kernel.cpp)，在qHeadNum为16/32/64场景（分别对应模型侧TP8/4/2场景）性能更优。
2. 特化的[mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp)，在qHeadNum为128场景（对应模型侧TP1场景）性能更优。

[mla_kernel.cpp](./mla_kernel.cpp)具有以下特性：
* 采用FlashAttention的四阶段计算流程，对于输入的Q, QRope, K, KRope进行切块后运算。
* 对输入序列长度`kvSeqlen`按照`blockSize`为单位进行切块，每次Attention运算的基块为一个block，使能提前下发一个基块的QK Mmad与softmax，让不同基块的CUBE与VECTOR阶段互相掩盖。
* 在同一基块的QK与PV的矩阵乘之间，由于K与V共用同一段数据，使能K常驻在L1 buffer上，减少搬入带宽占用。

[mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp)具有以下特性：
* 采用FlashAttention的四阶段计算流程，对于输入的Q, QRope, K, KRope进行切块后运算。
* 对输入序列长度`kvSeqlen`按照`blockSize`为单位进行切块，每次Attention运算的基块为四个block，使能提前下发一个基块的QK Mmad与softmax，让不同基块的CUBE与VECTOR阶段互相掩盖。
* 由于基块大小的放大，使得该Kernel的PV Mmad阶段的搬出数据量降低，减少了搬出带宽占用，相应的，由于硬件buffer大小限制，取消了K的常驻。

在本算子中，使用了Block和Tile层级组件来组装Kernel，具体步骤为：
1. 组装attention计算中的两个BlockMmad（QK,PV）以及三个BlockEpilogue（softmax, rescaleO, flashDecoding）。
2. 将Block组合在一起构建成`MLAKernel`，并在Kernel类中完成对各个Block的循环调用。  

这一过程也体现在Kernel入口的代码中（以[mla_kernel.cpp](./mla_kernel.cpp)为例）：
```c++
// GEMM Block模块，实现Flash MLA的Q * K^T
using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;
using QType = Gemm::GemmType<ElementQ, LayoutQ>;
using KType = Gemm::GemmType<ElementK, LayoutK>;
using SType = Gemm::GemmType<ElementS, LayoutS>;
using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

// Epilogue Block模块, 实现Flash MLA中当前S基块的softmax
using PType = Gemm::GemmType<ElementP, LayoutP>;
using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
using EpilogueMLASoftmax =
    Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLASoftmax, PType, SType, MaskType>;

// GEMM Block模块，实现Flash MLA的P * V
using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;
using VType = Gemm::GemmType<ElementV, LayoutV>;
using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

// Epilogue Block模块, 实现Flash MLA中当前O基块的更新
using OType = Gemm::GemmType<ElementO, LayoutO>;
using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
using EpilogueMLARescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLARescaleO, OType, OUpdateType, OTmpType>;

// Epilogue Block模块, 实现Flash MLA中flash decoding
using lType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
constexpr uint32_t ComputeEleNum = 6144;
using EpilogueMLAFDRescaleO =
    Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;

// Kernel level
using MLAKernel = MLAKernel<BlockMmadQK, BlockMmadPV, EpilogueMLASoftmax,
                            EpilogueMLARescaleO, EpilogueMLAFDRescaleO>;
```

## Block Mmad
算子总共使用了两类Block Mmad组件，分别为：
* `BlockMmadQK`为BlockMmad模板类的偏特化，用于处理Flash-MLA中的Q与K的矩阵乘操作，头文件[block_mmad_mla_qk.hpp](../../include/catlass/gemm/block/block_mmad_mla_qk.hpp)中的实现对应通用的[mla_kernel.cpp](./mla_kernel.cpp)，头文件[block_mmad_mla_qk_tp1_spec.hpp](../../include/catlass/gemm/block/block_mmad_mla_qk_tp1_spec.hpp)中的实现则对应特化的[mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp).  
* `BlockMmadPV`为BlockMmad模板类的偏特化，用于处理Flash-MLA中的P与V的矩阵乘操作，头文件[block_mmad_mla_pv.hpp](../../include/catlass/gemm/block/block_mmad_mla_pv.hpp)中的实现对应通用的[mla_kernel.cpp](./mla_kernel.cpp)，头文件[block_mmad_mla_pv_tp1_spec.hpp](../../include/catlass/gemm/block/block_mmad_mla_pv_tp1_spec.hpp)中的实现则对应特化的[mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp).  

## Block Epilogue
算子总共使用了三类Block Epilogue组件，分别为：
* `EpilogueMLASoftmax`为BlockEpilogue模板类的偏特化，用于处理Flash-MLA中的online softmax操作，头文件[block_epilogue_mla_softmax.hpp](../../include/catlass/epilogue/block/block_epilogue_mla_softmax.hpp)中的实现对应通用的[mla_kernel.cpp](./mla_kernel.cpp)，头文件[block_epilogue_mla_tp1_softmax.hpp](../../include/catlass/epilogue/block/block_epilogue_mla_tp1_softmax.hpp)中的实现则对应特化的[mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp).  
* `EpilogueMLARescaleO`为BlockEpilogue模板类的偏特化，用于处理Flash-MLA中的rescaleO操作，头文件[block_epilogue_mla_rescale_o.hpp](../../include/catlass/epilogue/block/block_epilogue_mla_rescale_o.hpp)中的实现对应通用的[mla_kernel.cpp](./mla_kernel.cpp)，头文件[block_epilogue_mla_tp1_rescale_o.hpp](../../include/catlass/epilogue/block/block_epilogue_mla_tp1_rescale_o.hpp)中的实现则对应特化的[mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp).  
* `EpilogueMLAFDRescaleO`为BlockEpilogue模板类的偏特化，用于处理Flash-MLA中的flashDecoding操作（如有必要），头文件[block_epilogue_mla_fd_rescale_o.hpp](../../include/catlass/epilogue/block/block_epilogue_mla_fd_rescale_o.hpp)中的实现为[mla_kernel.cpp](./mla_kernel.cpp)与[mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp)两者共用。

## Tile Mmad & Tile Copy
在通用Kernel使用的Block组件中，使用了位于[tile_mmad.hpp](../../include/catlass/gemm/tile/tile_mmad.hpp)中的tileMmad组件和位于[tile_copy.hpp](../../include/catlass/gemm/tile/tile_copy.hpp)中的tileCopy组件，例如：

```c++
using TileMmad = TileMmad_;
using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
using ElementAccumulator =
    typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
```