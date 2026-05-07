# ConvBias Example Readme
## 代码组织
```
├── 24_conv_bias
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
|   ├── gen_data.py   # 生成输入及标杆数据
│   └── conv_bias.cpp # 主文件
```

## 功能介绍
- 实现3D卷积功能。
- 计算公式：
  我们假定输入（input）的shape是 $(N, C_{\text{in}}, D_i, H_i, W_i)$ ，（weight）的shape是 $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$，输出（output）的shape是 $(N, C_{\text{out}}, D_o, H_o, W_o)$，那输出将被表示为：

  $$
    \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
  $$

  其中，$\star$表示互相关的计算。$N$代表batch size，$C$代表通道数，$D$、$H$和$W$分别代表深度、高度和宽度，相应输出维度的计算公式如下：

  $$
    D_o=[(D_i + 2 * padding[0] - dilation[0] * (K_d - 1) - 1 ) / stride[0]] + 1 \\
    H_o=[(H_i + 2 * padding[1] - dilation[1] * (K_h - 1) - 1 ) / stride[1]] + 1 \\
    W_o=[(W_i + 2 * padding[2] - dilation[2] * (K_w - 1) - 1 ) / stride[2]] + 1
  $$

- 当前实现相较于cann仅支持w轴全载的基础Conv3D功能，不涉及weight bypass、L1开doublebuffer、pointwise以及w轴切分等优化手段和分支，输入input、weight和bias在L1上的搬运量不能超过硬件限制，即需要满足以下条件：
  $$
    weightL1Size = K_h * K_w * 512 \\
    hoInL1Max = 16 / W_o + 2 \\
    hiInL1Max = (hoInL1Max - 1) * stride[1] + 1 + (K_h - 1) * dilation[1] \\
    hiInL1Max = min(H_i, hiInL1Max) \\
    inputL1Size = hiInL1Max * W_i * 32 \\
    biasL1Size = 64 \\
    weightL1Size + inputL1Size + biasL1Size < 524288
  $$

## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 第一步， 首先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入。
```
# python3 ./examples/24_conv_bias/gen_data.py |batch|cin|di|hi|wi|cout|kd|kh|kw|sD|sH|sW|dD|dH|dW|pD|pH|pW|dtype
# 最后一个参数指明数据类型为**float16**或 **bfloat16**
python3 ./examples/24_conv_bias/gen_data.py 32 64 1 32 48 128 1 1 1 1 1 1 1 1 1 0 0 0 float16
```
执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据
```
├── data
│   ├── fmap.bin   # 卷积的featureMap（NDC1HWC0的私有格式，数据排布为[batch, di, cin1, hi, wi, cin0]，其中cin0 = 16，cin1 = ceilDiv(cin, cin0)）
│   ├── weight.bin  # 卷积的weight（FRACTAL_Z_3D的私有格式，数据排布为[kdc1khkw, n1, n0, cin0]，其中n0 = 16，n1 = ceilDiv(cout, n0)）
|   ├── bias.bin   # 卷积的bias（ND格式，数据排布为[cout]）
│   └── golden.bin # cpu计算卷积的标杆结果 （NDC1HWC0的私有格式，数据排布为[batch, do, cout1, ho, wo, cout0]，其中cout0=16，cout1 = ceilDiv(cout, cout0)）
```
- 第二步，执行算子，这里需要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。
```
# 编译指定用例
bash scripts/build.sh 24_conv_bias
cd output/bin
# 可执行文件名 |batch|di|cin1|hi|wi|cin0|cout|kd|kh|kw|sD|sH|sW|dD|dH|dW|pD|pH|pW|Device ID
# Device ID可选，默认为0
./24_conv_bias 32 1 4 32 48 16 128 1 1 1 1 1 1 1 1 1 0 0 0 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```