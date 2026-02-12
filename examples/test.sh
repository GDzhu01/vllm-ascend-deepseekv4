# export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
# export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-8.5.0/opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=2048
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_VERSION=0.13.0
export USE_MULTI_BLOCK_POOL=1
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=8,9
nohup vllm serve /home/models/hello2026/ \
  --max_model_len 32000 \
  --max-num-batched-tokens 32000 \
  --served-model-name qwen \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --async-scheduling \
  --max-num-seqs 64 \
  --port 8666 \
  --block-size 128 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"num_speculative_tokens": 2,"method": "deepseek_mtp"}' \
  --additional_config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": true, "multistream_dsa_preprocess":true}' \
  2>&1 | tee run_online.log

#   --enforce_eager \ --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
#   --quantization ascend \
# --speculative-config '{"num_speculative_tokens": 2,"method": "deepseek_mtp"}' \
