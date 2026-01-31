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
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=8,9
nohup vllm serve /home/models/hello2026/ \
  --max_model_len 512 \
  --max-num-batched-tokens 512 \
  --served-model-name qwen \
  --gpu-memory-utilization 0.95 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --max-num-seqs 2 \
  --port 8666 \
  --block-size 128 \
  --enforce_eager \
  --additional_config '{"enable_cpu_binding": "True"}' \
  2>&1 | tee run_online.log

#   --async-scheduling \
#   --quantization ascend \

# nohup vllm serve /home/z00828031/weights/dummy_dsv4_layer4_es12/ \
#   --max_model_len 1024 \
#   --max-num-batched-tokens 1024 \
#   --served-model-name qwen \
#   --gpu-memory-utilization 0.80 \
#   --data-parallel-size 2 \
#   --enable-expert-parallel \
#   --port 8444 \
#   --block-size 128 \
#   --enforce_eager \
#   --additional_config '{"enable_cpu_binding": "True"}' \
#   2>&1 | tee run.log

# nohup vllm serve /home/z00828031/weights/dummy_dsv4_layer4_es12/ \
#   --max_model_len 1024 \
#   --max-num-batched-tokens 1024 \
#   --served-model-name qwen \
#   --gpu-memory-utilization 0.80 \
#   --enable-expert-parallel \
#   --port 8444 \
#   --block-size 128 \
#   --enforce_eager \
#   --additional_config '{"enable_cpu_binding": "True"}' \
#   2>&1 | tee run.log


