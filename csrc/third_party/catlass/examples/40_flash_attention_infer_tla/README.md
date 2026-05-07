# FlashAttentionInferTla Example Readme
## 代码组织
```
├── 40_flash_attention_infer_tla
│   ├── CMakeLists.txt # CMake构建配置文件
│   ├── gen_data.py
│   ├── kernel_common.hpp 
│   ├── fai.cpp
│   ├── fai_kernel.cpp
│   ├── fai_tiling.cpp
│   └── README.md
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)   

- 接下来，先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据。   
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

以下是一个完整的shell脚本示例
```
batch=1
qSeqlen=177
kvSeqlen=512
numHeads=1
kvHeads=1
headSize=128
isVariedLen=0
maskType=1
dtype="bf16"
cacheMode=1
layout_dtype=0
num_blocks=2048
inner_prec=0
lse_flag=0
device=0

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh 40_flash_attention_infer_tla
}

function gen_data() {
    python3 examples/40_flash_attention_infer_tla/gen_data.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType "$dtype" $cacheMode $layout_dtype $num_blocks $inner_prec $lse_flag
    echo "Data gen finished"
}

function run_kernel() {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' D=' $headSize ' mask=' $maskType
    cd output/bin/
    ./40_flash_attention_infer_tla $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType --device $device --dtype $dtype
}

build
gen_data
run_kernel
```

执行结果如下，说明精度比对成功。
```
Compare success.
```