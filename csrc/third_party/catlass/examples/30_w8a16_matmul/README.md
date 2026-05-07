# W8A16Matmul Example Readme
## 代码组织
```
├── 30_w8a16_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── w8a16_matmul.cpp # 主文件
```
## 功能介绍
- 增加反量化功能，将输入B矩阵从`int8`转换为`fp16_t`(`half`)，再做反量化：与`deqZeroPoint`求和后和`deqScalar`相乘，而后与A矩阵做Matmul。
- 当前实现仅支持RowMajor、ColumnMajor数据排布。
## 使用示例
- 获取代码后，编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 30_w8a16_matmul
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./30_w8a16_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```