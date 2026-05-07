# GroupedMatmul Example Readme
## 代码组织
```
├── 08_grouped_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── grouped_matmul.cpp # 主文件
```
## 示例说明
- 本grouped_matmul为通用kernel，示例为沿k轴切分场景。
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 08_grouped_matmul
cd output/bin
# 可执行文件名 group数量|m轴|n轴|k轴|Device ID
./08_grouped_matmul 128 512 1024 2048 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```