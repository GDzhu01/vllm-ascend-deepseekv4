# quant_lightning_indexer算子测试框架
## 功能说明
基于pytest测试框架，实现quant_lightning_indexer算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围
### 参数限制

- **数据格式**:
  - **query_layout**：BSND、TND
  - **key_layout**: PA_BSND

### 环境配置

#### 前置要求
1、 确认torch_npu为最新版本  
2、 参考[Attention融合算子Experimental使用说明](../../../Attention融合算子Experimental使用说明.md)激活CANN包和自定义算子包

#### custom包调用
支持custom包调用

## 文件结构
#### pytest文件结构说明
- test_run_sh                                  # 执行脚本
- quant_lightning_indexer_golden.py            # cpu侧算子golden实现
- result_compare_method.py                     # cpu golden与npu输出精度对比
- pytest.ini                                   # 创建测试标记

单用例测试：
- test_quant_lightning_indexer_single.py       # pytest测试单用例运行主程序 
- test_quant_lightning_indexer_paramset.py     # 单用例入参配置

批量测试：
- test_quant_lightning_indexer_batch.py        # 用例批量测试主程序并生成excel文件保存结果
- ./batch/quant_lightning_indexer_pt_loadprocess.py    # 读取pt文件并调用算子获取npu输出
- ./batch/quant_lightning_indexer_pt_save.py           # 读取excel表格批量生成用例pt文件
- ./batch/replace_path.py                              # test_quant_lightning_indexer_batch.py占位符替换
 

## 使用方法
在pytest文件夹路径下执行：

### 运行测试用例
#### 单用例调测
1、手动配置test_quant_lightning_indexer_paramset.py的参数

2、执行指令：
``` bash
bash test_run.sh single
```
#### 用例的批量生成与测试
1、excel路径下存放用例excel表格

2、test_run.sh中设置读取的用例excel表格路径和pt文件存放路径

3、执行指令：
``` bash
bash test_run.sh batch
```