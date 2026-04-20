# sparse_attn_sharedkv算子测试框架

## 功能说明
基于pytest测试框架，实现sparse_attn_sharedkv算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围

### 参数限制
- **数据格式：**
    - **layout_q**：BSND、TND
    - **layout_kv**：PA_BSND

### 环境配置
#### 前置要求
1、 确认torch_npu为最新版本  
2、 参考[Attention融合算子Experimental使用说明](../../../Attention融合算子Experimental使用说明.md)激活CANN包和自定义算子包
#### custom包调用
支持custom包调用

## 文件结构
### pytest文件结构说明
- test_run.sh                               # 执行脚本
- sparse_attn_sharedkv_golden.py            # cpu侧算子golden实现
- result_compare_method.py                  # cpu golden与npu输出精度对比
- pytest.ini                                # 创建测试标记

单用例测试：
- test_sparse_attn_sharedkv_single.py       # pytest测试单用例运行主程序
- sparse_attn_sharedkv_paramset.py          # 单用例入参配置

批量测试：
- test_sparse_attn_sharedkv_batch.py        # 读取pt文件并进行用例批量测试主程序，生成excel文件保存结果
- ./batch/sparse_attn_sharedkv_process.py     # 调用算子获取npu输出
- ./batch/sparse_attn_sharedkv_pt_save.py     # 读取excel表格批量生成用例pt文件

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例
#### 单用例调测
1、手动配置sparse_attn_sharedkv_paramset.py的参数  
2、执行指令：
```bash
bash test_run.sh single
```
#### 用例的批量生成与测试
1、excel路径下存放用例excel表格  
2、执行指令批量生成：
``` bash
bash test_run.sh save # 默认路径执行用例生成
```
支持从指定路径下读取excel表格并指定sheet页，pt文件保存到指定文件夹：
``` bash
bash test_run.sh save -E "./excel/example.xlsx" -S "Sheet1" -P "./data"
```
其中，-E设置excel文件存储路径，-S设置Sheet名，-P设置pt存储文件夹，各参数可单独配置

3、执行指令批量测试：
``` bash
bash test_run.sh load
```
支持从指定路径下读取pt文件，结果保存到指定文件夹：
``` bash
bash test_run.sh load -P "./data" -R "./result/sas_result.xlsx"
```
其中，-P设置pt读取文件夹，-R设置结果存储路径，各参数可单独配置
