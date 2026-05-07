# OptimizedMatmul Example Readme
## 代码组织
```
├── 06_optimized_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── optimized_matmul.cpp # 主文件
```
## 功能介绍
matmul矩阵乘，相比00_basic_matmul样例替换dispatchPolicy为`MmadAtlasA2Preload`，并增加输入矩阵的padding前处理，提升数据搬入性能。
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 06_optimized_matmul
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./06_optimized_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```
## 说明
样例里当前padding动作使用的是`PADDING_NZ`，也可以替换为`PADDING_BLOCK_ND`来测试性能表现
- **PADDING_NZ**
代码位置如下
```cpp
    constexpr PaddingTag paddingTagA = (std::is_same_v<LayoutA, layout::zN> || std::is_same_v<LayoutA, layout::nZ>)
                                           ? PaddingTag::NO_PADDING
                                           : PaddingTag::PADDING_NZ;
    constexpr PaddingTag paddingTagB = (std::is_same_v<LayoutB, layout::zN> || std::is_same_v<LayoutB, layout::nZ>)
                                           ? PaddingTag::NO_PADDING
                                           : PaddingTag::PADDING_NZ;
```
基于`PADDING_NZ`策略的UB上的COMPUTE_LENGTH为48KB
```cpp
static const uint32_t COMPUTE_LENGTH_A = 48 * 1024 / sizeof(ElementA);
static const uint32_t COMPUTE_LENGTH_B = 48 * 1024 / sizeof(ElementB);
```
- **PADDING_BLOCK_ND**
替换`PADDING_BLOCK_ND`的代码修改如下，当输入矩阵非NZ格式时使能，会将矩阵按照`L1TileShape`对齐来做padding
```diff
    constexpr PaddingTag paddingTagA = (std::is_same_v<LayoutA, layout::zN> || std::is_same_v<LayoutA, layout::nZ>)
                                           ? PaddingTag::NO_PADDING
-                                          : PaddingTag::PADDING_NZ;
+                                          : PaddingTag::PADDING_BLOCK_ND;
    constexpr PaddingTag paddingTagB = (std::is_same_v<LayoutB, layout::zN> || std::is_same_v<LayoutB, layout::nZ>)
                                           ? PaddingTag::NO_PADDING
-                                          : PaddingTag::PADDING_NZ;
+                                          : PaddingTag::PADDING_BLOCK_ND;
```
基于`PADDING_BLOCK_ND`策略的UB上的COMPUTE_LENGTH为96KB
```diff
-static const uint32_t COMPUTE_LENGTH_A = 48 * 1024 / sizeof(ElementA);
-static const uint32_t COMPUTE_LENGTH_B = 48 * 1024 / sizeof(ElementB);
+static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
+static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
```