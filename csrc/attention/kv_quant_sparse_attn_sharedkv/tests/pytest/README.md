# kv_quant_sparse_attn_sharedkv算子测试框架
## 功能说明
基于pytest测试框架，实现kv_quant_sparse_attn_sharedkv算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调、图模式调用获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围
### 参数限制

- **数据格式**:
    - **layout_q**：BSND、TND
    - **layout_kv**：PA_ND

### 环境配置

#### 前置要求
1、确认torch_npu为最新版本  
2、参考[Attention融合算子Experimental使用说明](../../../Attention融合算子Experimental使用说明.md)激活CANN包合自定义算子包

#### custom包调用
支持custom包调用

## 文件结构
#### pytest文件结构说明
- test_run.sh                                               # 用例执行脚本
- kv_quant_sparse_attn_sharedkv_golden.py                   # 算子入参处理及cpu侧golden实现
- result_compare_method.py                                  # cpu golden与npu结果精度对比脚本
- pytest.ini                                                # 创建测试标记

单用例测试：
- test_kv_quant_sparse_attn_sharedkv_single.py              # pytest测试单用例运行主程序
- kv_quant_sparse_attn_sharedkv_paramset.py                 # 单用例入参数配置

批量测试：
- test_kv_quant_sparse_attn_sharedkv_batch.py               # 用例批量测试主程序，读取pt用例并获取npu输出，结果保存至excel文件
- ./batch/test_kv_quant_sparse_attn_sharedkv_pt_save.py     # 读取excel表格批量生成用例pt文件
- ./batch/kv_quant_sparse_attn_sharedkv_process.py          # 调用算子接口获取npu输出

## 使用方法
在pytest文件夹路径下执行：

### 运行测试用例
#### 单用例测试
1、手动配置kv_quant_sparse_attn_sharedkv_paramset.py的参数

2、执行指令：
``` bash
bash test_run.sh single
```
#### 用例的批量生成与测试
1、excel路径下存放用例excel表格

2、执行指令：
``` bash
bash test_run.sh batch
```