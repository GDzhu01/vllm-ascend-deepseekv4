# SparseMatmulTla Example Readme
## 代码组织
```
├── 41_sparse_matmul_tla
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   ├── sparse_gen_data.py
│   └── sparse_matmul_tla.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)   

- 接下来，先执行`sparse_gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在指定路径下生成input和output目录，包含算子的输入数据和用于精度验证的golden数据。   
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致，并且本样例只支持A/B矩阵输入为int8_t数据类型。

以下是一个完整的shell脚本示例（在工程目录下执行）
```
m=160
n=320
k=64
device=0

function build() {
    bash scripts/build.sh 41_sparse_matmul_tla
}

function gen_data() {
    cd examples/41_sparse_matmul_tla
    python3 sparse_gen_data.py $m $n $k
    echo "Data gen finished"
}

function run_kernel {
    echo 'Case: m=' $m ' k=' $k ' n=' $n
    cd ../../output/bin/
    cp -r ../../examples/41_sparse_matmul_tla/input .
    cp -r ../../examples/41_sparse_matmul_tla/output .
    ./41_sparse_matmul_tla $m $n $k $device
}

build
gen_data
run_kernel
```
执行结果如下，说明精度比对成功。
```
Compare success.
```