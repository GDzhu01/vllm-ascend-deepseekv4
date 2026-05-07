# MLA Example Readme
## 代码组织
```
├── 19_mla
│   ├── CMakeLists.txt # CMake编译文件
│   ├── gen_data.py
│   ├── kernel_common.hpp #两个不同的kernel实现中的共同变量与宏
│   ├── main.cpp
│   ├── mla_kernel.cpp # MLA TP 2/4/8 模板
│   ├── mla_kernel_tp1_spec.cpp # MLA TP 1 模板
│   └── README.md
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 第一步，首先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入。
```
# 在./examples/19_mla目录下执行
python gen_data.py 1 1 128 16 16 128 half
# 输入参数分别对应 batchSize，qSeqlen，kvSeqlen, qheadNum，numBlock, blockSize
# qSeqlen表示需要推理的token个数，支持范围为1~4，即常规decode与mtp场景
# kvSeqlen表示输入的序列长度
# blockSize参数当前支持128
# 最后一个参数需要指明数据类型为“half”或“bf16”
```
执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据
```
├── data
│   ├── block_table.bin
│   ├── golden.bin
│   ├── k.bin
│   ├── k_rope.bin
│   ├── kv_seqlen.bin
│   ├── q.bin
│   ├── q_ntokens.bin
│   ├── q_rope.bin
│   └── q_seqlen.bin
```
第二步，执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。
```
# 编译指定用例（在CATLASS代码仓目录下）
bash scripts/build.sh 19_mla
cd output/bin
./19_mla 1 1 128 16 16 128
# 此处的参数和生成数据的参数保持一致
# 完整参数为 batchSize, qSeqlen, kvSeqlen, qheadNum, numBlock, blockSize [--dtype DTYPE --datapath DATA_PATH --device DEVICE_ID]，dtype默认为half, datapath默认为../../examples/19_mla/data, device默认为0。
```
执行结果如下，说明精度比对成功。
```
Compare success.
```