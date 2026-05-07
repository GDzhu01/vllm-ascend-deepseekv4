# TLA Tensors

这篇文档描述了CATLASS的TLA(Tensor Layout Abstraction)下的`Tensor`。

本质上，`Tensor` （张量）表示一个多维数组。Tensor 抽象了数组元素在内存中的组织方式与存储方式的细节。这使得用户能够编写通用的访问多维数组的算法，并可根据张量的特性（traits）对算法进行特化。如张量的depth、rank、layout、数据的类型、位置等。

`Tensor` 包含4个模板参数: `BuiltinTensor`、 `Layout`、 `Coord`、 `Position`。
关于 `Layout` 的描述, 请参考 [Layout](./01_layout.md)。

## BuiltinTensor 和 Position

`BuiltinTensor` 为AscendC内的 `GlobalTensor` 或者 `LocalTensor`，`Position` 为AscendC定义的各层级位置。相关使用参考AscendC文档。

## Tensor 构造

当前提供 `MakeTensor` 接口构造`Tensor`， 包含四个模板参数： `BuiltinTensor`、 `Layout`、 `Coord`、 `Position`。

有如下两种方式构造：

```cpp
using namespace tla;
GlobalTensor<float> A = ...;

// 缺省Coord, 默认为（0， 0）
Layout w8xh16 = MakeLayout(MakeShape(8, Int<16>{}), MakeStride(Int<16>{},Int<1>{}));
Tensor tensor_8x16 = MakeTensor(A, w8xh16, Arch::PositionGM{});

// 用户指定Coord
Tensor tensor_8x16 = MakeTensor(A, w8xh16, tla::MakeCoord(1, 5), Arch::PositionGM{});
```

## Tensors 接口

TLA `Tensor` 提供获取相应特性的接口：

* `.data()`. 返回 `Tensor` 的内存。

* `.layout()`. 返回 `Tensor` 的 `layout`。

* `.coord()`. 返回 `Tensor` 的 `coord`。

* `.shape()`. 返回 `Tensor` 的 `shape`。

* `.stride()`. 返回 `Tensor` 的 `stride`。

## 获取 TileTensor

提供一个 `GetTile` 接口获取 `Tensor` 的一片子tensor，会根据输入坐标对coord进行更新，并依据新的Tile的shape变换layout（只是逻辑层面的数据组织形式），底层的数据实体不变更。

```cpp
using namespace tla;
Layout w8xh16 = MakeLayout(MakeShape(8, Int<16>{}), MakeStride(Int<16>{},Int< 1>{}));
Tensor tensor_8x16 = MakeTensor(A, w8xh16, Arch::PositionGM{});

auto tensor_tile = GetTile(tensor_8x16, tla::MakeCoord(2, 4), MakeShape(4, 8)); // (4,8):(_16,_1)
```
