# 性能数据分析脚本
rm result.xlsx
msprof python3 -m pytest -rA -s test_kv_quant_sparse_attn_sharedkv_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee res.log
python process_perf_data.py --test_result_path="result.xlsx" --is_compare True --delete_perf_file True