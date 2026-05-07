# StreamkMatmul Example Readme
## 代码组织
```
├── 37_streamk_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── streamk_matmul.cpp # 主文件
```
## 模板说明
该模板为切尾轮基本块的多核切K模板，通过切分尾轮基本块的`K`，划分出更多的任务块，从而利用更多的核心参与尾轮的基本块计算。
该模板切分是将尾轮的计算量直接均分到所有计算核心上。

```sh
# 编译指定用例
bash scripts/build.sh 37_streamk_matmul
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./37_streamk_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```

## 使用场景说明
设矩阵Shape为`M N K`, L1上的分块大小为`m1 n1 k1`，AI Core数量为`C`，那么可以划分出来的基本块数量为`B = CeilDiv(M, m1) * CeilDiv(N, n1)`，计算轮次为`B / C`，如果`B % C > 0`，那么还需要计算一个尾轮，
当`B / C > 1`且`B % C <= C * 0.8`时候，采用该模板可能获取较优的性能。

**测试性能建议注释掉精度比较代码。**
