# GroupGemm Example Readme
## 代码组织
```
├── 16_group_gemm
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── group_gemm.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 16_group_gemm
cd output/bin
# 可执行文件名 |矩阵个数|矩阵m轴组|n轴组|k轴组|Device ID
# Device ID可选，默认为0
./16_group_gemm 3 "128,256,512" "256,512,128" "512,256,128" 0
```
执行结果如下，说明精度验证通过。
```
Compare success.
```