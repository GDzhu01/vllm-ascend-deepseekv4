## msTuner_CATLASS (MindStudio Tuner for CATLASS) - Tiling自动寻优工具

msTuner_CATLASS 是一款用于 CATLASS 模板库算子 Tiling 参数寻优的工具，支持用户自定义搜索空间，能够实例化搜索空间内的所有算子，并批量完成在板性能测试，为算子 Tiling 参数的寻优提供参考依据。

### 快速上手

以m=256，n=512，k=1024 的**00_basic_matmul**的tiling参数寻优为例，使用默认的搜索空间配置，执行以下命令完成工具的编译。

```bash
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=00_basic_matmul mstuner_catlass
```

输入`mstuner_catlass`命令，启动性能测试。

```bash
export LD_LIBRARY_PATH=$PWD/output/lib64/:$LD_LIBRARY_PATH
./output/bin/mstuner_catlass --m=256 --n=512 --k=1024 --device=0 --output=results.csv
```

运行成功后，将会出现以下回显信息（实际运行结果因硬件差异与硬件性能波动不一定完全相同）。

```bash
$ ./output/bin/mstuner_catlass --m=256 --n=512 --k=1024 --device=0 --output=results.csv
[INFO ] Set profile output file /path_to_my_repo/catlass/output/results.csv
[INFO ] Start to initialize device 0
[INFO ] Initializing device 0 success
[INFO ] Initializing 1701 operations
[WARN ] Current freq 800 is lower than rated freq 1800, run warm up
[INFO ] Warm up finished, rated freq 1800, current freq 1800
================================

             case_id : 1
   task_duration(us) : 19.380
           device_id : 0
           operation : Gemm
         description : catlass_gemm_00_basic_matmul_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_32x128x128_32x128x32_swizzle3x0
       l0_tile_shape : 32x128x32
       l1_tile_shape : 32x128x128
             swizzle : swizzle3x0
                   m : 256
                   n : 512
                   k : 1024
                   A : fp16:row
                   B : fp16:row
                   C : fp16:row

================================

...

================================
Top 10:
case_id,task_duration(us),device_id,operation,description,m,n,k,A,B,C
489,12.740,7,Gemm,catlass_gemm_00_basic_matmul_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_64x128x128_64x128x64_swizzle3x1,256,512,1024,fp16:row,fp16:row,fp16:row
...
[INFO ] Save profile data to /path_to_my_repo/catlass/output/results.csv success
```

### 编译

支持通过`-DCATLASS_LIBRARY_KERNELS=<kernel_name>`命令过滤算子，当算子的`description`信息包含`kernel_name`时，该算子用例代码会被生成并编译，比如通过如下命令指定编译`00_basic_matmul`类算子。

```bash
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=00_basic_matmul mstuner_catlass
```

可直接指定具体的单个算子实例的description信息，比如使用如下命令，仅编译快速上手中所展示的case_id为1的算子。

```bash
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=catlass_gemm_00_basic_matmul_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_32x128x128_32x128x32_swizzle3x1 mstuner_catlass
```

当前已支持如下算子类型。

- 00_basic_matmul
- 02_grouped_matmul_slice_m
- 06_optimized_matmul
- 08_grouped_matmul
- 12_quant_matmul
- 27_matmul_gelu

注意：
- 06_optimized_matmul 算子在不同的m/n/k输入时，会启用不同的kernel，包括：06_optimized_matmul_padding_ab, 06_optimized_matmul_padding_a_only, 06_optimized_matmul_padding_b_only, 06_optimized_matmul_without_padding，可直接指定预期运行的kernel以加速寻优过程，kernel匹配逻辑请参考 [optimized_matmul.cpp](../../examples/06_optimized_matmul/optimized_matmul.cpp)。

除直接使用上述命令外，编译也可通过cmake命令完成。

```bash
mkdir build
cd build
cmake .. -DCATLASS_LIBRARY_KERNELS=00_basic_matmul
make mstuner_catlass -j
cmake --install . --component catlass_kernels
cmake --install . --component mstuner_catlass
```

### 工具运行命令

`mstuner_catlass`工具支持以下命令。

| 命令          | 示例                          | 默认值 | 描述                                                         |
| ------------- | ----------------------------- |-| ------------------------------------------------------------ |
| --help, -h    | --help                        | / | 展示工具支持的命令。                                           |
| --kernels     | --kernels=00_basic_matmul        | / | 过滤寻优的算子类型，其与算子的description列字符串进行子串匹配，未匹配时该算子会被跳过。 |
| --output      | --output=./profile_result.csv | / | 指定算子性能数据落盘文件路径。                                 |
| --device      | --device=0                    | 0 | 指定运行的单卡ID。                                             |
| --m           | --m=256                       | 256 | 指定输入矩阵的维度m。                                          |
| --n           | --n=512                       | 512 | 指定输入矩阵的维度n。                                          |
| --k           | --k=1024                      | 1024 | 指定输入矩阵的维度k。                                          |
| --A           | --A=fp16:row                  | / | 通过指定矩阵A的数据类型与内存排布过滤算子。                      |
| --B           | --B=fp16:column               | / | 通过指定矩阵B的数据类型与内存排布过滤算子。                    |
| --C           | --C=fp16:row                  | / | 通过指定矩阵C的数据类型与内存排布过滤算子。                    |
| --group_count | --group_count=128             | 128 | 指定grouped_matmul类算子的group数量。                          |

当搜索空间配置并生成了多种A、B、C的数据类型与内存排布时，支持通过`--A/--B/--C=<数据类型>:<内存排布>`命令对算子进行过滤。
- 数据类型支持`u8, int8, int32, fp16, bf16, fp32`。
- 内存排布支持`row, column, nZ, zN, zZ, padding_row_major, padding_column_major, nN`。
- 要求输入`<data:layout>`的格式，如`fp16:row`，`fp32:zZ`。
注意：不指定`--output`时，不会落盘算子性能数据。

### 搜索空间配置

msTuner_CATLASS支持对算子tiling参数的搜索空间进行自定义配置，支持自定义配置排布方式（layouts）、数据类型（data types）、L1/L0的TileShape、Swizzle策略等参数，自动正交生成全量搜索空间，并可自定义剪枝函数过滤搜索空间，最终每种正交配置组合会实例化为一个独立算子，生成的算子实例化代码位于`build/tools/library/generated`目录中。

当搜索空间范围配置较广时，可能导致上万个算子被实例化，代码膨胀致使编译耗时较长。此外，过多的算子可能超过硬件限制，无法保证编译成功。同时，算子数量较多时，算子下发前注册耗时也较长。因此建议将搜索空间的规模控制在5000以内，以确保工具运行顺畅，获得最佳体验。

算子数量可通过查看日志文件`build/tools/library/catlass_library_code_generation.log`，如下所示，00_basic_matmul的搜索空间实例了1701个算子。

```txt
INFO:search_space:00_basic_matmul tile_shapes size=1701
INFO:search_space:08_grouped_matmul tile_shapes size=576
INFO:manifest:operations that will be generated in total: 1701
...
```

搜索空间配置支持入门级配置与高级配置。

#### 入门级配置

msTuner_CATLASS支持对算子tiling参数搜索空间进行入门级的简化配置（见`tools/library/scripts/search_space_config.py`文件），开发者可自由调整以下参数来设置搜索范围。

- kernel_type，算子类型
- data_type_a/data_type_b/data_type_c：A/B/C输入矩阵的元素类型
- layout_a/layout_b/layout_c：A/B/C输入矩阵的内存排布
- l1_tile_m_range：L1 Tile Shape的m轴取值搜索范围
- l1_tile_n_range：L1 Tile Shape的n轴取值搜索范围
- l1_tile_k_range：L1 Tile Shape的k轴取值搜索范围
- block_swizzle：Swizzle策略

00_basic_matmul算子的一种搜索空间配置如下：

```python
@OperationRegistry.register_high_priority('00_basic_matmul')
def register(manifest):
    config = search_space.SearchSpaceConfiguration(
        kernel_type='00_basic_matmul',

        data_type_a=library.DataType.fp16,
        data_type_b=library.DataType.fp16,
        data_type_c=library.DataType.fp16,

        layout_a=library.LayoutType.RowMajor,
        layout_b=library.LayoutType.RowMajor,
        layout_c=library.LayoutType.RowMajor,

        l1_tile_m_range=(32, 128),  # min and max of a range are set here
        l1_tile_n_range=(128, 256),
        l1_tile_k_range=(128, 256),

        block_swizzle='Gemm::Block::GemmIdentityBlockSwizzle<3, 0>',
    )

    search_space.register_custom_kernel(config, manifest)
```

入门级配置的算子搜索空间会**覆盖**高级配置中同类型算子的配置，若暂不需要使能入门级配置而使用高级配置，可把该行代码注释掉。

```python
# @OperationRegistry.register_high_priority('00_basic_matmul')
```

#### 高级配置

msTuner_CATLASS支持在 `tools/library/scripts/search_space.py`文件中对算子tiling参数的搜索空间进行更为灵活的自定义配置，支持自定义配置layouts、data types、L1/L0 Tile Shapes、Swizzle策略等参数的正交组合方式，自定义剪枝函数过滤筛选搜索空间遍历。

以`00_basic_matmul`算子的搜索空间为例，其配置位于函数`register_gemm_00_basic_matmul_operation`中。

- layouts配置

  ```python
layouts = [
  [library.LayoutType.RowMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
]
  ```

- data types配置

  ```python
      data_types = [
          [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16]
      ]
  ```

- L1/L0 Tile Shapes配置与自定义剪枝函数`tile_shape_constraint_for_pingpong`

  ```python
      tile_shapes = list(generate_tile_shapes(
          tile_shape_constraint_for_pingpong, # set constraint function based on dispatch policy
          # below are arguments for constraint function
          element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
          stages=[2], # stages of dispatch policy for estimating boundary conditions, e.g. 2 for UB stages
          step=16, # step size for iterating the next tile shape on each dimension of L1/L0 tile shape
          tile_shape_range=TileShapeRange(
            l1_tile_m_range=(32, 128),  # range of L1TileShape::M/N/K
            l1_tile_n_range=(128, 256),
            l1_tile_k_range=(128, 256),
            l0_tile_m_range=(32, 128),  # range of L0TileShape::M/N/K
            l0_tile_n_range=(128, 256),
            l0_tile_k_range=(32, 64)
          )
      ))
  ```

- Swizzle策略配置

  ```python
      block_swizzle_descriptions = [
          'Gemm::Block::GemmIdentityBlockSwizzle<3, 0>',
      ]
  ```

类似的，`08_grouped_matmul`算子的搜索空间配置位于函数`register_gemm_08_grouped_matmul_operation`中，支持自定义配置。