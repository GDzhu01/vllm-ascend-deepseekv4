# PaddingSplitkMatmul Example Readme
## 代码组织
```
├── 22_padding_splitk_matmul
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── padding_splitk_matmul.cpp # 主文件
```
## 功能介绍
该算子支持A、B矩阵做完padding后，附加k轴方向切分用于分核，在`m`，`n`较小时提高Cube核利用率。
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 22_padding_splitk_matmul
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./22_padding_splitk_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```