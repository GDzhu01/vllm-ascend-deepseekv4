# 在CATLASS样例工程使用msDebug

[msDebug](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/optool/atlasopdev_16_0062.html)是用于调试在NPU侧运行的算子程序的一个工具，该工具向算子开发人员提供了在昇腾设备上调试算子的手段。调试手段包括了读取昇腾设备内存与寄存器、暂停与恢复程序运行状态等。

- ⚠️ **注意** 此功能依赖社区版`CANN`包版本为[8.2.RC1.alpha003](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha003)。
- ⚠️ **注意** 请检查驱动是否支持/设备节点`/dev/drv_debug`是否映射至容器内（参考[驱动检查](https://www.hiascend.com/document/caselibrary/detail/atlasopdev_0006)）

## 使用示例

下面以`00_basic_matmul`为例，进行msDebug调试的使用说明。

### 使能驱动的调试功能

参考[msDebug工具概述](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/optool/atlasopdev_16_0062.html)，以`debug`模式安装驱动，或在`full`模式安装的驱动下执行`echo 1 > /proc/debug_switch`打开调试通道。

为了避免出现安全问题，请勿在生产环境启用调试通道。

- 若出现以下问题，说明驱动版本较低，需更新驱动。

```bash
msdebug failed to initialize. please install HDK.
[ERROR] error code: 0x20102
terminate called after throwing an instance of 'MSDEBUG_ERROR_CODE'
```

### 编译运行

1. 基于[快速上手](../quickstart.md)，打开工具的编译开关`--debug --msdebug`，使能`debug`与`msdebug`编译算子样例。

```bash
bash scripts/build.sh --debug --msdebug 00_basic_matmul
```

- `--debug`同时控制host与device侧代码的debug开关，`--msdebug`控制device侧代码的debug开关。
- 若只增加`--debug`，只会启用host的调试功能，仅能用gdb/lldb调试host侧代码。

2. 切换到可执行文件的编译目录 `output/bin` 下，使用`msdebug`执行算子样例程序。

```bash
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
msdebug ./00_basic_matmul 256 512 1024 0
```

```bash
msdebug ./00_basic_matmul 256 512 1024 0
msdebug(MindStudio Debugger) is part of MindStudio Operator-dev Tools.
The tool provides developers with a mechanism for debugging Ascend kernels running on actual hardware.
This enables developers to debug Ascend kernels without being affected by potential changes brought by simulation and emulation environments.
(msdebug) target create "./00_basic_matmul"
Current executable set to '/home/catlass/output/bin/00_basic_matmul' (aarch64).
(msdebug) settings set -- target.run-args  "256" "512" "1024" "0"
(msdebug) 
```

### 命令行调试

#### 设置断点和程序执行

通过命令`b basic_matmul.cpp:45`和`b basic_matmul.cpp:90`（在[`00_basic_matmul.cpp`](../../examples/00_basic_matmul/basic_matmul.cpp)中90~101行为类型别名定义，非运行时机器代码）设置两个断点，再用`breakpoint list`查看已有断点。

```bash
(msdebug) b basic_matmul.cpp:45
Breakpoint 1: where = 00_basic_matmul`Run(GemmOptions const&) + 460 at basic_matmul.cpp:45:18, address = 0x000000000016df8c
(msdebug) b basic_matmul.cpp:90
Breakpoint 2: where = 00_basic_matmul`Run(GemmOptions const&) + 2816 at basic_matmul.cpp:101:39, address = 0x000000000016e8c0
(msdebug) breakpoint list
Current breakpoints:
1: file = 'basic_matmul.cpp', line = 45, exact_match = 0, locations = 1
  1.1: where = 00_basic_matmul`Run(GemmOptions const&) + 460 at basic_matmul.cpp:45:18, address = 00_basic_matmul[0x000000000016df8c], unresolved, hit count = 0 

2: file = 'basic_matmul.cpp', line = 90, exact_match = 0, locations = 1
  2.1: where = 00_basic_matmul`Run(GemmOptions const&) + 2816 at basic_matmul.cpp:101:39, address = 00_basic_matmul[0x000000000016e8c0], unresolved, hit count = 0 

(msdebug) 
```

执行命令`r`，程序将开始运行直到第一个断点处，再执行命令`c`，程序将运行到下一个断点。需要注意的是，对于多核程序而言，算子程序通常会被下发至多个加速核并发运行，一旦某一个加速核命中了断点，会通过中断通知其他的加速核立即停下，因此不保证其他的加速核也一定同时在该断点停下，而且相同的断点也可能被其他的加速核再次命中，开发者可配合禁用/删除断点命令来防止加速核不停命中同一个断点的情况。

```bash
(msdebug) r
Process 813993 launched: '/home/catlass/output/bin/00_basic_matmul' (aarch64)
Process 813993 stopped
* thread #1, name = '00_basic_matmul', stop reason = breakpoint 1.1
    frame #0: 0x0000aaaaaac0df8c 00_basic_matmul`Run(options=0x0000ffffffffe340) at basic_matmul.cpp:45:18
   42  
   43       uint32_t m = options.problemShape.m();
   44       uint32_t n = options.problemShape.n();
-> 45       uint32_t k = options.problemShape.k();
   46  
   47       size_t lenA = static_cast<size_t>(m) * k;
   48       size_t lenB = static_cast<size_t>(k) * n;
(msdebug) c
Process 813993 resuming
Process 813993 stopped
* thread #1, name = '00_basic_matmul', stop reason = breakpoint 2.1
    frame #0: 0x0000aaaaaac0e8c0 00_basic_matmul`Run(options=0x0000ffffffffe340) at basic_matmul.cpp:101:39
   98      using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
   99 
   100      using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
-> 101      MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};
   102      MatmulAdapter matmulOp;
   103      matmulOp.CanImplement(arguments);
   104      size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
(msdebug) c
Process 813993 resuming
[Launch of Kernel _ZN7Catlass13KernelAdapterINS_4Gemm6Kernel11BasicMatmulINS1_5Blo on Device 0]
Compare success.
Process 813993 exited with status = 0 (0x00000000) 
(msdebug) 
```

#### 查看变量和内存

如果想查看标量，通过`p`指令，可以直接查看当前n变量的值。

```bash
Process 813993 launched: '/home/catlass/output/bin/00_basic_matmul' (aarch64)
Process 813993 stopped
* thread #1, name = '00_basic_matmul', stop reason = breakpoint 1.1
    frame #0: 0x0000aaaaaac0df8c 00_basic_matmul`Run(options=0x0000ffffffffe340) at basic_matmul.cpp:45:18
   42  
   43       uint32_t m = options.problemShape.m();
   44       uint32_t n = options.problemShape.n();
-> 45       uint32_t k = options.problemShape.k();
   46  
   47       size_t lenA = static_cast<size_t>(m) * k;
   48       size_t lenB = static_cast<size_t>(k) * n;
(msdebug) p n
(uint32_t) $0 = 512
```

如果想查看内存，先通过`p`指令，查看当前内存的信息。

通过`x -m UB -f float16[] 65536 -c 4 -s 4`命令，可以打印accumulatorBuffer内存中的值，一次最多打印1024字节。

```bash
(msdebug) c
Process 814339 resuming
Process 814339 stopped
[Switching to focus on Kernel _ZN7Catlass13KernelAdapterINS_4Gemm6Kernel12SplitkMatmulINS1_5Bl, CoreId 0, Type aiv]
* thread #1, name = '09_splitk_matmu', stop reason = breakpoint 2.1
    frame #0: 0x000000000000bf98 device_debugdata`_ZN7Catlass4Gemm6Kernel9ReduceAddINS_4Arch7AtlasA2EfDhLj8192EEclERKN7AscendC12GlobalTensorIDhEERKNS7_IfEEmj_mix_aiv(this=0x00000000001cf838, dst=0x00000000001cf930, src=0x00000000001cf908, elementCount=131072, splitkFactor=2) at splitk_matmul.hpp:136:19
   133 
   134              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
   135              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
-> 136              Ub2Gm(dst[loopIdx * tileLen], outputBuffer[bufferIndex], actualTileLen);
   137              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
   138 
   139              bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
(msdebug) p outputBuffer
(AscendC::LocalTensor<__fp16>[2]) $2 = {
  [0] = {
    AscendC::BaseLocalTensor<__fp16> = { # 内存、数据类型
      address_ = (dataLen = 131072, bufferAddr = 65536, bufferHandle = "", logicPos = '\v') # 起始地址、数据长度
    }
    shapeInfo_ = {
      shapeDim = '\x88'
      originalShapeDim = '\xf8'
      shape = {}
      originalShape = {}
      dataFormat = ND
    }
  }
  [1] = {
    AscendC::BaseLocalTensor<__fp16> = {
      address_ = (dataLen = 49152, bufferAddr = 147456, bufferHandle = "", logicPos = '\v')
    }
    shapeInfo_ = {
      shapeDim = '\x88'
      originalShapeDim = '\xf8'
      shape = {}
      originalShape = {}
      dataFormat = ND
    }
  }
}
(msdebug) x -m UB -f float16[] 65536 -c 4 -s 4 # 在UB内存中从65536的地址分打印4行4字节的fp16数据
0x00010000: {355.5 188.75}
0x00010004: {244.125 -364.75}
0x00010008: {-104.875 -156}
0x0001000c: {232 -100.75}
(msdebug) x -m UB -f float16[] 65536 -c 4 -s 8 # 在UB内存中从65536的地址分打印4行8字节的fp16数据
0x00010000: {355.5 188.75 244.125 -364.75}
0x00010008: {-104.875 -156 232 -100.75}
0x00010010: {-47.4062 105.875 -322.5 -265.75}
0x00010018: {260 200.125 -139.25 -190.625}
(msdebug) 
```

如果想逐行调试，运行命令`n`，使程序运行至下一行

```bash
(msdebug) n
Process 814339 stopped
[Switching to focus on Kernel _ZN7Catlass13KernelAdapterINS_4Gemm6Kernel12SplitkMatmulINS1_5Bl, CoreId 0, Type aiv]
* thread #1, name = '09_splitk_matmu', stop reason = step over
    frame #0: 0x000000000000bfe4 device_debugdata`_ZN7Catlass4Gemm6Kernel9ReduceAddINS_4Arch7AtlasA2EfDhLj8192EEclERKN7AscendC12GlobalTensorIDhEERKNS7_IfEEmj_mix_aiv(this=0x00000000001cf838, dst=0x00000000001cf930, src=0x00000000001cf908, elementCount=131072, splitkFactor=2) at splitk_matmul.hpp:137:73
   134              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
   135              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
   136              Ub2Gm(dst[loopIdx * tileLen], outputBuffer[bufferIndex], actualTileLen);
-> 137              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
   138 
   139              bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
   140          }
(msdebug) n
Process 814339 stopped
[Switching to focus on Kernel _ZN7Catlass13KernelAdapterINS_4Gemm6Kernel12SplitkMatmulINS1_5Bl, CoreId 0, Type aiv]
* thread #1, name = '09_splitk_matmu', stop reason = step over
    frame #0: 0x000000000000c000 device_debugdata`_ZN7Catlass4Gemm6Kernel9ReduceAddINS_4Arch7AtlasA2EfDhLj8192EEclERKN7AscendC12GlobalTensorIDhEERKNS7_IfEEmj_mix_aiv(this=0x00000000001cf838, dst=0x00000000001cf930, src=0x00000000001cf908, elementCount=131072, splitkFactor=2) at splitk_matmul.hpp:139:28
   136              Ub2Gm(dst[loopIdx * tileLen], outputBuffer[bufferIndex], actualTileLen);
   137              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
   138 
-> 139              bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
   140          }
   141 
   142          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[0]);
(msdebug) n
Process 814339 stopped
[Switching to focus on Kernel _ZN7Catlass13KernelAdapterINS_4Gemm6Kernel12SplitkMatmulINS1_5Bl, CoreId 0, Type aiv]
* thread #1, name = '09_splitk_matmu', stop reason = step over
    frame #0: 0x000000000000c014 device_debugdata`_ZN7Catlass4Gemm6Kernel9ReduceAddINS_4Arch7AtlasA2EfDhLj8192EEclERKN7AscendC12GlobalTensorIDhEERKNS7_IfEEmj_mix_aiv(this=0x00000000001cf838, dst=0x00000000001cf930, src=0x00000000001cf908, elementCount=131072, splitkFactor=2) at splitk_matmul.hpp:96:68
   93           AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[1]);
   94  
   95           uint32_t loops = (elementCount + tileLen - 1) / tileLen;
-> 96           for (uint32_t loopIdx = aivId; loopIdx < loops; loopIdx += aivNum) {
   97               uint32_t actualTileLen = tileLen;
   98               if (loopIdx == loops - 1) {
   99                   actualTileLen = elementCount - loopIdx * tileLen;
(msdebug) 
```

通过`var`命令，可以查看当前栈帧的全部变量。

```bash
(msdebug) var
(Catlass::Gemm::Kernel::ReduceAdd<Catlass::Arch::AtlasA2, float, __fp16, 8192> *__stack__) this = 0x00000000001cf838
(const AscendC::GlobalTensor<__fp16> &__stack__) dst = 0x00000000001cf930: {
  AscendC::BaseGlobalTensor<__fp16> = {
    address_ = 0x000012c0c0094000
    oriAddress_ = 0x000012c0c0094000
  }
  bufferSize_ = 1898896
  shapeInfo_ = {
    shapeDim = 'h'
    originalShapeDim = '\xf9'
    shape = {}
    originalShape = {}
    dataFormat = ND
  }
  cacheMode_ = CACHE_MODE_NORMAL
}
(const AscendC::GlobalTensor<float> &__stack__) src = 0x00000000001cf908: {
  AscendC::BaseGlobalTensor<float> = {
    address_ = 0x000012c041400000
    oriAddress_ = 0x000012c041400000
  }
  bufferSize_ = 1898904
  shapeInfo_ = {
    shapeDim = 'H'
    originalShapeDim = '\xf9'
    shape = {}
    originalShape = {}
    dataFormat = ND
  }
  cacheMode_ = CACHE_MODE_NORMAL
}
(uint64_t) elementCount = 131072
(uint32_t) splitkFactor = 2
(const uint32_t) ELE_PER_VECTOR_BLOCK = 64
(uint32_t) aivNum = 48
(uint32_t) aivId = 26
(uint64_t) taskPerAiv = 2752
(uint32_t) tileLen = 2752
(uint32_t) loops = 48
(uint32_t) loopIdx = 26
(msdebug) 
```

#### 退出调试

调试完成后，通过命令`q`退出`msdebug`，若通过`Ctrl+C`等手段强行退出，则`msdebug`进程不会结束，仍在后台运行，此时可通过`ps -ef | grep msdebug`查找对应的进程pid，再用`kill -9 进程pid`终止对应进程即可。不能同时启动多个`msdebug`进程进行调试。

```bash
(msdebug) q
Quitting LLDB will kill one or more processes. Do you really want to proceed: [Y/n] y
```

#### 常用命令表

|  命令  |  命令缩写  |  作用  |  示例  |
| ------ | ---------- | ------ | ------- |
|  breakpoint filename:lineNo | b | 增加断点 |  b add\_custom.cpp:85<br>b my\_function |
|  run  | r | 重新运行 | r |
|  continue | c |  继续运行 | c |
|  print | p | 打印变量| p zLocal |
|  frame variable | var | 打印当前帧所有变量 | var |
|  memory read | x | 读内存<br>-m 指定内存位置，支持GM/UB/L0A/L0B/L0C<br>-f 指定[字节转换格式](#附录1)<br>-s 指定每行打印字节数<br>-c 指定打印的行数 |  x -m GM -f float16[] 1000 -c 2 -s 128  |
|  register read | re r | 读取寄存器值<br>-a 读取所有寄存器值<br>\$REG\_NAME 读取指定名称的寄存器值 | register read -are r \$PC |
|  thread step-over |  next<br>n                          |  在同一个调用栈中，移动到下一个可执行的代码行                                                                                                                              |  n |
|  ascend info devices |  /  |  查询device信息  |  ascend info devices |
|  ascend info cores  |  /  |  查询算子所运行的aicore相关信息 |  ascend info cores  |
|  ascend info tasks  |  /  |  查询算子所运行的task相关信息  |  ascend info tasks  |
|  ascend info stream  |  /    |  查询算子所运行的stream相关信息 |  ascend info stream  |
|  ascend info blocks  |  /  |  查询算子所运行的block相关信息<br>可选参数： -d/–details显示所有blocks当前中断处代码                                                                                        |  ascend info blocks  |
|  ascend aic core  |  /  |  切换调试器所聚焦的cube核 |  ascend aic 1  |
|  ascend aiv core  |  /  |  切换调试器所聚焦的vector核  |  ascend aiv 5 |
|  target modules addkernel.o  |  image addkernel.o  |  PyTorch框架拉起算子时，导入算子调试信息 <br>（注：当程序执行run命令后再执行本命令导入调试信息，<br>则还需额外执行image load命令以使调试信息生效） |  image addAddCustom\_xxx.o  |
|  target modules load –f kernel.o –s address  |  image load -f kernel.o -s address  |  在程序运行后，使导入的调试信息生效                                                                                                                                       |  image load -f AddCustom\_xxx.o -s 0  |

## 附录

### msdebug支持的数据格式

```bash
Valid values are:
"default"
'B' or "boolean"
'b' or "binary"
'y' or "bytes"
'Y' or "bytes with ASCII"
'c' or "character"
'C' or "printable character"
'F' or "complex float"
's' or "c-string"
'd' or "decimal"
'E' or "enumeration"
'x' or "hex"
'X' or "uppercase hex"
'f' or "float"
"brain float16"
'o' or "octal"
'O' or "OSType"
'U' or "unicode16"
"unicode32"
'u' or "unsigned decimal"
'p' or "pointer"
"char[]"
"int8_t[]"
"uint8_t[]"
"int16_t[]"
"uint16_t[]"
"int32_t[]"
"uint32_t[]"
"int64_t[]"
"uint64_t[]"
"bfloat16[]"
"float16[]"
"float32[]"
"float64[]"
"uint128_t[]"
'I' or "complex integer"
'a' or "character array"
'A' or "address"
"hex float"
'i' or "instruction"
'v' or "void"
'u' or "unicode8"
```

### 指定调试使用的NPU卡

配置环境变量`ASCEND_RT_VISIBLE_DEVICES`为需要使用的NPU卡号，例如

```bash
# 指定当前进程仅使用Device ID为2的Device
export ASCEND_RT_VISIBLE_DEVICES=2
```
