# GroupedMatmulSliceM Example Readme
## 代码组织
```
├── 02_grouped_matmul_slice_m
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── grouped_matmul_slice_m.cpp # 主文件
```
## 功能介绍
该算子支持A矩阵在m轴切分，随后与B矩阵按照group分组进行矩阵乘。
## 使用示例
因为GroupedMatmul参数较多，所以该示例直接在代码中承载输出参数列表`groupList`, 通过`golden::GenerateGroupList`来生成随机切分的序列。
相关输入配置具体详见[grouped_matmul_slice_m.cpp](grouped_matmul_slice_m.cpp)。
如果需要输入grouplist配置（例如通过tensorList方式构造输入），可以参考python_extension中相应实现

example使用
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 02_grouped_matmul_slice_m
cd output/bin
# 可执行文件名|group数量|矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./02_grouped_matmul_slice_m 128 512 1024 2048 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```