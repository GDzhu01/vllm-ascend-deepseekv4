# 950BasicMatmulTla Example Readme

**注意：社区包暂不支持950能力，后续支持的版本敬请期待**

## 代码组织

```
├── 43_ascend950_basic_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── basic_matmul_tla.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)，本用例为950算子，编译时需加-DCATLASS_ARCH=3510
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 43_ascend950_basic_matmul -DCATLASS_ARCH=3510
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./43_ascend950_basic_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```

## 使用说明
BasicMatmul默认使用的DispatchPolicy MmadPingpong支持以下几个模板参数：

|模板参数|默认值|参数说明|
|---------|-----------------|-----------------|
|ArchTag| 无 | 指定架构型号 | 
|enableUnitFlag| false | 是否开启Unitflag，开启L0C多缓冲时必须设置为false |
|useHF32| false | 是否开启HF32，仅float类型支持 |
|l0CStages| 1 | 指定L0C的缓冲区数量，设置为2即可开启L0C双缓冲|
|enableL1Resident| false | 是否开启L1常驻 |
|l1AStages | 2 | L1上加载矩阵A的Buffer数量 |
|l1BStages | 2 | L1上加载矩阵B的Buffer数量 |
|l0AStages | 2 | L0上加载矩阵A的Buffer数量 |
|l0AStages | 2 | L0上加载矩阵B的Buffer数量 |

设矩阵Shape为`M N K`, L1上的分块大小为`m1 n1 k1`，M方向的分块数量`mTiles = CeilDiv(M, m1)`，N方向的分块数量`nTiles = CeilDiv(N, n1)`，总任务数为`taskBlocks = mTiles * nTiles`，在以下两种情况下可以选择开启enableL1Resident：

1.`mTies = 1`，且`nTiles > CoreNum`，且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`n1`设置为原来的一半。

2.`nTies = 1`，且`mTiles > CoreNum`, 且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`m1`设置为原来的一半。

BasicMatmul还支持DispatchPolicy MmadPreloadAsyncWithCallback，支持以下几个模板参数：

|模板参数|默认值|参数说明|
|---------|-----------------|-----------------|
|ArchTag| 无 | 指定架构型号 | 
|preloadStages| 无 | 指定预加载的次数 | 
|l1AStages | 2 | L1上加载矩阵A的Buffer数量 |
|l1BStages | 2 | L1上加载矩阵B的Buffer数量 |
|l0AStages | 2 | L0上加载矩阵A的Buffer数量 |
|l0AStages | 2 | L0上加载矩阵B的Buffer数量 |
|l0CStages| 1 | 指定L0C的缓冲区数量，设置为2即可开启L0C双缓冲|
|enableUnitFlag| false | 是否开启Unitflag，开启L0C多缓冲时必须设置为false |
|enableShuffleK| false | 是否开启K方向错位读取 |
|useHF32| false | 是否开启HF32，仅float类型支持 |
|enableL1Resident| false | 是否开启L1常驻 |

相比`MmadPingpong`，`MmadPreloadAsyncWithCallback`多两个模板参数，一个是`preloadStages`，此参数一般设置为1即可，指定预加载次数，当该参数设置为1时，第一个循环之进行数据加载，不进行matmul计算，第二轮循环先加载第二轮循环的数据，然后完成上一轮循环的Matmul计算，依此类推，最后循环结束后，会补一次Matmul计算。这样做的好处是，当前计算Matmul需要的数据，在上一轮就搬运好了，做到了指令的提前发射，缓解了指令发射延迟造成的性能损失。

第二个参数是`enableShuffleK`，该参数主要为了规避同地址访问冲突造成带宽损失，主要原理是错开各个核心的数据读取地址。950上不需要启用该参数。

相比`MmadPingpong`，`MmadPreloadAsyncWithCallback`的优化点更多，但是逻辑也更复杂，Scalar开销更大，需要根据场景酌情使用，尤其是小Shape场景。