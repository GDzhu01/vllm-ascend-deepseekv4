# BasicMatmul Example Readme

## 功能说明

 - 算子功能：完成基础矩阵乘计算
 - 计算公式：
  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
    \end{aligned}
  $$
  其中$A$和$B$分别是形如`(m,k)`，`(k,n)`的输入矩阵，$C$是形如`(m,n)`的输出矩阵。

## 参数说明

以下是本样例的运行参数：
| 参数名 | 描述 | 约束 | 
| ----- | -------- | ------ | 
| `m` | 矩阵乘中左矩阵A的行 | - | 
| `n` | 矩阵乘中右矩阵B的列 | - | 
| `k` | 矩阵乘中左矩阵A的列<br>（也即右矩阵的行数） | - | 
| `deviceId` | 使用的NPU卡ID（默认0） | 在设备NPU有效范围内 | 


BasicMatmul所涉及的关键模板参数如下:

| 模板参数 | 说明 | 有效范围 | 
| ----- | -------- | -------------- |
| `ElementA` | 左矩阵的数据类型 | `float` \| `fp16_t` \| `bfloat16_t` \| `int8_t` | 
| `ElementB` | 右矩阵的数据类型 | `float` \| `fp16_t` \| `bfloat16_t` \| `int8_t` | 
| `ElementC` | 右矩阵的数据类型 | `float` \| `fp16_t` \| `bfloat16_t` \| `int8_t` | 
| `LayoutA` | 左矩阵的排布方式 | `layout::RowMajor` \| `layout::ColumnMajor` | 
| `LayoutB` | 右矩阵的排布方式 | `layout::RowMajor` \| `layout::ColumnMajor` | 
| `LayoutC` | 结果矩阵的排布方式 | `layout::RowMajor` | 


## 约束说明

左、右矩阵及结果矩阵的类型应满足下述类型映射条件。

| `ElementA` | `ElementB` | `ElementC` | 
| ----- | ----- | ---------- | 
| `float` | `float` | `float` \| `fp16_t` \| `bfloat16_t` | 
| `fp16_t` | `fp16_t` | `float` \| `fp16_t` \| `bfloat16_t` | 
| `bfloat16_t` | `bfloat16_t` | `float` \| `fp16_t` \| `bfloat16_t` | 
| `int8_t` | `int8_t` | `int32_t` | 


## 代码组织
```
├── 00_basic_matmul
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── basic_matmul.cpp # 主文件
```

## 使用示例
1. 编译样例代码，并编译生成相应的算子可执行文件。
```
bash scripts/build.sh 00_basic_matmul
```

2. 切换到可执行文件的编译目录`output/bin`下，执行算子样例程序。测试样例数据随机生成，尺寸从命令行输入。
```
cd output/bin
./00_basic_matmul 256 512 1024 0
```
• 256：矩阵m轴

• 512：n轴

• 1024：k轴

• 0：Device ID，可选，默认为0


执行结果如下，说明样例执行成功。
```
Compare success.
```