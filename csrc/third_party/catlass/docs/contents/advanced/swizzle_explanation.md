# Swizzle策略说明

Swizzle策略决定了AI Core计算基本块的顺序。调整Swizzle策略有助于提高缓存命中率、减小数据读取开销，从而提高矩阵乘整体计算效率。

下方展示了3种Swizzle策略。图中每一个方块表示C矩阵的一个基本块，方块中的序号代表AI Core的编号（该示例中假设AI Core数量为20）。箭头方向指示了特定Swizzle策略下基本块的遍历顺序，我们按照该顺序将基本块依次分配给AI Core进行处理，编号0~19的20个基本块是并行计算的。

## 示例1

默认的Swizzle策略为SwizzleOffset=1、SwizzleDirection=0，即：

```c++
 using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<>;
```

<img src="../../images/swizzle10.png" width="60%">

## 示例2

SwizzleOffset=3、SwizzleDirection=0

```c++
 using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
```

<img src="../../images/swizzle30.png" width="60%">

## 示例3

SwizzleOffset=3、SwizzleDirection=1

```c++
 using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
```

<img src="../../images/swizzle31.png" width="60%">

## Swizzle策略选择

如果C矩阵的大小为M x N，那么当M >= N时，采用SwizzleOffset=3、SwizzleDirection=0，通常情况下能够达到较好的性能；当M < N时，采用SwizzleOffset=3、SwizzleDirection=1，通常情况下可以达到较好的性能。开发者也可以探索其他参数设置以达到更高的缓存命中率，从而进一步提高矩阵计算性能。
