# MatmulGelu Example Readme
## 代码组织
```
├── 27_matmul_gelu
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── matmul_gelu.cpp # 主文件
```
## 功能介绍

完成以下函数的计算

Gelu:
$$
out = Gelu(a × b)

$$
其中Gelu的公式为：
$$
Gelu(x) =0.5∗x∗(1+Tanh(\sqrt {2/π}∗(x+0.044715∗x^3 )))
$$
Sigmoid：
$$
Sigmoid(x)=\frac{1}{1+e^{-x}}
$$
Tanh:
$$
\begin{aligned}
Tanh(x) &= \frac{(e^x - e^{-x})}{(e^x + e^{-x})}\\
&= \frac{(e^x - e^{-x})\times e^{-x}}{(e^x + e^{-x}) \times e^{-x} }\\
&= \frac{1 - e^{-2x} }{1 + e^{-2x}}\\
&= 1 - 2\times \frac{e^{-2x}}{1 + e^{-2x}}\\
&= 1 - 2\times (1 - \frac{1}{1 + e^{-2x}})\\
&= 1 - 2\times (1 - Sigmoid(2x))
\end{aligned}
$$
因此可化简为：
$$
Tanh(x) = 2\times Sigmoid(2x) - 1
$$

基于上述讨论，回顾Gelu的计算式：
$$
Gelu(x) =0.5∗x∗(1+Tanh(\sqrt {2/π}∗(x+0.044715∗x^3 )))
$$
让$Z=\sqrt {2/π}∗(x+0.044715∗x^3)$，代入上式化简得：
$$
Gelu(x) =x∗Sigmoid( 2Z )
$$
再次展开，并取近似：$\sqrt {8/π}\approx 1.595769$，有:
$$
Gelu(x) \approx x∗Sigmoid( 1.595769∗(x+0.044715∗x^3 ))
$$
在上式中展开$Sigmoid$函数，最终Gelu的计算形式为：
$$
Gelu(x) = \frac{x}{1+e^{-1.595769∗(x+0.044715∗x^3 )}}
$$

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 27_matmul_gelu
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./27_matmul_gelu 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```