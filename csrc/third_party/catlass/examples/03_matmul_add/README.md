# MatmulAdd Example Readme
## 代码组织
```
├── 03_matmul_add
│   ├── CMakeLists.txt  # CMake编译文件
│   ├── README.md
│   └── matmul_add.cpp  # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 03_matmul_add
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./03_matmul_add 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```