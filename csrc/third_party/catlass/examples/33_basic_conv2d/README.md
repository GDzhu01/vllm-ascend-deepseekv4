# BasicConv2d Example Readme
## 代码组织
```
├── 33_basic_conv2d
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── basic_conv2d.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 33_basic_conv2d
cd ./output/bin
# 可执行文件名 |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# Device ID可选，默认为0
./33_basic_conv2d 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```
执行结果如下，表明精度验证通过。
```
Compare success.
```