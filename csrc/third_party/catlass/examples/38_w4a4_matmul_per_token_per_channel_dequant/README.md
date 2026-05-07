# W4A4MatmulPerTokenPerChannelDequant Example Readme

## 功能说明

- 算子功能：完成Int4（`AscendC::int4b_t`）类型的矩阵乘计算，包含per-token和per-channel级别的反量化系数

- 计算公式

$$

out = perTokenScale \times x @ weight \times perChannelScale


$$


其中`x`是矩阵乘的左矩阵（形如`(m, k)`），`weight`是右矩阵（形如`(k,n)`），`perChannelScale`是一个形如`(n)`的一维向量，`perTokenScale`是一个形如`(m)`的一维向量。

## 参数说明

以下是本样例的运行参数：
| 参数名 | 描述 | 约束 | 
| ----- | -------- | ------ | 
| `m` | 矩阵乘中左矩阵A的行（int4格式）| - | 
| `n` | 矩阵乘中右矩阵B的列 (int4格式) | 须为偶数 |
| `k` | 矩阵乘中左矩阵A的列<br>（也即右矩阵的行数） | 须为偶数 | 
| `deviceId` | 使用的NPU卡ID（默认0） | 在设备NPU有效范围内 | 

- `AscendC::int4b_t`底层处理方式是，在`1Byte`内表示两个`AscendC::int4b_t`类型数据，如以`1Byte`为基本类型视图，则左矩阵形为`(m, k/2)`，右矩阵形为`(k, n/2)`
- 更多约束详见约束说明

样例涉及的关键模板参数如下:

| 模板参数 | 说明 | 有效范围 | 
| ----- | -------- | -------------- |
| `ElementA` | 左矩阵的数据类型 | `AscendC::int4b_t` | 
| `ElementB` | 右矩阵的数据类型 | `AscendC::int4b_t` |
| `ElementD` | 结果矩阵的数据类型 | `bfloat16_t` | 
| `LayoutA` | 左矩阵的排布方式 | `layout::RowMajor` | 
| `LayoutB` | 右矩阵的排布方式 | `layout::zN`\| `layout::nZ` | 
| `LayoutD` | 结果矩阵的排布方式 | `layout::RowMajor` | 

## 约束说明
- n, k必须为偶数
- 当`LayoutB`为`layout::zN`时：
    - n需要能够整除64
    - k需要能够整除16
- 当`LayoutB`为`layout::nZ`时：
    - n需要能够整除16
    - k需要能够整除64

## 代码组织
```
├── 38_w4a4_matmul_per_token_per_channel_dequant
│   ├── CMakeLists.txt # CMake编译文件
│   ├── gen_data.py
│   ├── w4a4_matmul_per_token_per_channel_dequant.cpp
│   └── README.md
```
## 功能介绍
- 提供了W4A4量化模式下矩阵乘实现，使用per channel和per token量化

## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)   

- 执行`gen_data.py`，生成测试样例
 
- 执行算子

以下是一个完整的shell脚本示例
```
# 编译算子
bash scripts/build.sh 38_w4a4_matmul_per_token_per_channel_dequant

# 生成测试数据
cd examples/38_w4a4_matmul_per_token_per_channel_dequant/
# python gen_data.py <M> <N> <K>
python gen_data.py 256 512 1024
cd ../..

# 进行测试
cd output/bin/
./38_w4a4_matmul_per_token_per_channel_dequant 256 512 1024 0
```

执行结果如下，说明精度比对成功。
```
Compare success.
```
--------------
当前样例右矩阵采用NZ排布(即`LayoutB`为`layout::zN`，详见[`layout.hpp`](../../include/catlass/layout/layout.hpp))，如需修改为`layout::nZ`格式，请对`example/38_w4a4_matmul/w4a4_matmul.cpp`做调整：
```diff
- using LayoutB = layout::zN;
+ using LayoutB = layout::nZ;
```

并在生成测试例时补充`transB`参数置1（默认为0），完整测试过程如下：
```
# 算子编译
bash scripts/build.sh 38_w4a4_matmul_per_token_per_channel_dequant --clean 

# 生成测试数据
cd examples/38_w4a4_matmul_per_token_per_channel_dequant/
# python gen_data.py <M> <N> <K> <transB>
python gen_data.py 256 512 1024 1

cd ../..

# 进行测试
cd output/bin/
./38_w4a4_matmul_per_token_per_channel_dequant 256 512 1024 0
```


