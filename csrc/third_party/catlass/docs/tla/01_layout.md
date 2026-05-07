# TLA Layouts

这篇文档描述了CATLASS的TLA(Tensor Layout Abstraction)下的`Layout`数据结构，它提供了多维坐标与内存的映射关系。

`Layout`为多维数组的访问提供了一个通用接口，它将数组元素在内存中的组织细节抽象化。这使得用户能够编写通用的多维数组访问算法，从而在布局发生变化时，无需修改用户代码。

## 基础类型与概念

### Tuple

TLA以元组[`tla::tuple`](../../include/tla/tuple.hpp)为起始，tla::tuple 包含了若干个元素组成的有限序列元组，其行为与 std::tuple 类似，但引入了一些 C++ template arguments 的限制，并削减了部分实现以提升性能。

### IntTuple

TLA 还定义了[`IntTuple`](../../include/tla/int_tuple.hpp)概念。`IntTuple`既可作为一个整数，也可作为一个`Tuple`类型。这个递归定义允许我们构建任意嵌套的 Layout。

以下任何一个都是 `IntTuple` 的有效模板参数：

* `int{2}`: 运行时整数，也称为动态整数，就是 C++ 的正常整数类型比如`int`/`size_t`等等，只要是`std::is_integral<T>`的都是。

* `Int<3>{}`： 编译期整数，或称之为静态整数。TLA 通过 `tla::C<Value>` 来定义兼容的静态整数类型，使得这些整数的计算能在编译期内完成。TLA 将别名 _1、_2、_3等定义为`Int<1>`、`Int<2>`、`Int<3>`等类型。更多信息可查看[`integral_constant`](../../include/tla/numeric/integral_constant.hpp)。

* 带有任何模板参数的 IntTuple，例如 `make_tuple(int{2}, Int<3>{})`。

TLA 不仅将 `IntTuple` 用在了`Layout`上，还会在很多其他的地方比如 `Shape` 和 `Stride` 等用到它，详见
[`include/tla/layout.hpp`](../../include/tla/layout.hpp)。

`IntTuple` 的相关 API 操作：

* `rank(IntTuple)`: 返回 `IntTuple` 的元素个数。

* `get<I>(IntTuple)`: 返回 `IntTuple` 的第 `I` 个元素。

* `depth(IntTuple)`: 返回 `IntTuple` 的嵌套层数，整数为 0。

### Layout

`Layout` 本质上就是由2个 `IntTuple` 组成， `Shape` 和 `Stride` 。 `Shape` 定义了 `Tensor` 的形状，`Stride` 定义了元素间的距离。

## Layout 使用

`Layout` 也有许多与 `IntTuple` 类似的操作：

* `rank(Layout)`: `Layout` 的维度，等同于 `Stride` 的 rank(IntTuple)。

* `get<I>(Layout)`: 返回 `Layout` 的第 `Ith` 个元素，`I < rank`。

* `depth(Layout)`: 返回 `Layout` 的嵌套层数，整数为0。

* `shape(Layout)`: 返回 `Layout` 的 `Shape` 。

* `stride(Layout)`: 返回 `Layout` 的 `Stride` 。

此外，为了便于操作，还定义了下述一些函数：

* `get<I0,I1,...,IN>(x) := get<IN>(...(get<I1>(get<I0>(x)))...)`： 获取第 `I0` 个单元的第 `I1` 个单元的 ... 的第 `IN` 个单元。

* `rank<I...>(x)  := rank(get<I...>(x))`： 获取第 `I...` 个单元维度。

* `depth<I...>(x) := depth(get<I...>(x))`： 获取第 `I...` 个单元的嵌套层数。

* `shape<I...>(x)  := shape(get<I...>(x))`： 获取第 `I...` 个单元的形状。

* ...

### Layout 构造

`Layout` 有多种构造方式，可以是静态整数和动态整数的任意结合，可以定义任意维度。
**注：在昇腾CUBE核内部，存在 `zN` 、 `nZ` 、 `zZ` 、 `nN` 格式，因此目前昇腾算子模板库中只定义与使用`行优先`、`列优先`和前述4种格式**。

```c++
using namespace tla;
Layout w2xh4 = MakeLayout(MakeShape(Int<2>{}, 4),
                          MakeStride(Int<12>{}, Int<1>{}));

Layout w32xh48 = MakeLayout(MakeShape(MakeShape(16,2), MakeShape(16,3)),
                            MakeStride(MakeStride(16,256), MakeStride(1,512)));
```

 `MakeLayout` 函数返回 `Layout`。  `MakeShape` 函数返回 `Shape`。  `MakeStride` 函数返回 `Stride`。

上述Layout格式如下

```
w2xh4     :  (_2,4):(_12,_1)
w32xh48   :  ((16,2),(16,3)):((16,256),(1,512))
```

`(_2,4):(_12,_1)` 中前面的括号表示 `Tensor` 的形状，后面的括号表示在不同维度下的 `Stride`。

### Matrix examples

可以定义一个matrix的 `layout` 如下几种类型：

2x3 `行优先` layout

```
(2,3):(3,1)
      0   1   2
    +---+---+---+
 0  | 0 | 1 | 2 |
    +---+---+---+
 1  | 3 | 4 | 5 |
    +---+---+---+
```

定义了一个 2x3 的 tensor，2 行 3 列。至于 stride，在前一维度（行维度），stride=3，表示映射到一维空间中，按行方向递增时，内存跨度为3；在后一维度（列维度），stride=1，表示映射到一维空间中，按列方向递增时，内存跨度为1。

2x3 `列优先` layout

```
(2,3):(1,2)
      0   1   2
    +---+---+---+
 0  | 0 | 2 | 4 |
    +---+---+---+
 1  | 1 | 3 | 5 |
    +---+---+---+
```

定义了一个 2x3 的 tensor，2 行 3 列。至于 stride，在前一维度（行维度），stride=1，表示映射到一维空间中，按行方向递增时，内存跨度为1；在后一维度（列维度），stride=2，表示映射到一维空间中，按列方向递增时，内存跨度为2。

`zN` layout（ `示例展示为4x4块，实际核内为 16x(32/sizeof(ElemType))` ），其余3种内部格式类似

```
((4,2),(4,3)):((4,16),(1,32))
      0    1    2    3    4    5    6    7    8    9    10   11
    +----+----+----+----+----+----+----+----+----+----+----+----+
 0  | 0  | 1  | 2  | 3  | 32 | 33 | 34 | 35 | 64 | 65 | 66 | 67 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 1  | 4  | 5  | 6  | 7  | 36 | 37 | 38 | 39 | 68 | 69 | 70 | 71 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 2  | 8  | 9  | 10 | 11 | 40 | 41 | 42 | 43 | 72 | 73 | 74 | 75 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 3  | 12 | 13 | 14 | 15 | 44 | 45 | 46 | 47 | 76 | 77 | 78 | 79 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 4  | 16 | 17 | 18 | 19 | 48 | 49 | 50 | 51 | 80 | 81 | 82 | 83 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 5  | 20 | 21 | 22 | 23 | 52 | 53 | 54 | 55 | 84 | 85 | 86 | 87 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 6  | 24 | 25 | 26 | 27 | 56 | 57 | 58 | 59 | 88 | 89 | 90 | 91 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 7  | 28 | 29 | 30 | 31 | 60 | 61 | 62 | 63 | 92 | 93 | 94 | 95 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
```

在行维度上，我们有个子 tensor ，该子 tensor 有四行（此为第一个 shape 的第一个4），行之间的 stride 为 4（所以第一个 stride 第一个数为 4）；然后该子 tensor 在整个大 tensor 行中会重复两次（此为第一个 shape 的第二个 2），相对应地，子 tensor 间的 stride 为 16（此为第一个 stride 的第二个 16）。
同理，在列维度上，我们有个子 tensor ，该子 tensor 有四列（此为第二个 shape 第一个4），列之间的 stride 为 1（所以第二个 stride 第一个数为 1）；然后该子 tensor 在整个大 tensor 列中会重复三次（此为第二个 shape 的第二个 3），相对应地，子 tensor 间的 stride 为 32（此为第二个 stride 的第二个 32）。

### Layout 坐标与索引

在 TLA 中，可使用 `tla::crd2offset(c, shape, stride)`  将坐标转换到索引，目前坐标需为二维。

```cpp
auto shape  = Shape<Shape<_4,_2>,Shape<_4,_3>>{};
auto stride = Stride<Stride<_4,_16>,Stride<_1,_32>>{};
print(crd2offset(tla::MakeCoord(1, 5), shape, stride));  // 37
```

### 获取 Tilelayout

Tilelayout 可以用下列方式获取：

```cpp
using namespace tla;
Layout a   = Layout<Shape<Shape<_4,_2>,Shape<_4,_3>>, Stride<Stride<_4,_16>,Stride<_1,_32>>>{}; // ((4,2),(4,3)):((4,16),(1,32))
Layout a0  = MakeLayoutTile(a, MakeShape(4, 4)); // ((4,1),(4,1)):((4,16),(1,32))
```

`MakeLayoutTile`接口改变了原本layout的shape，不改变stride。
