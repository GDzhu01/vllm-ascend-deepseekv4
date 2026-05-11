## two nodes

node0
```bash
# this obtained through ifconfig node0
# nic_name is the network interface name corresponding to local_ip of the current node
local_ip="172.21.100.74" # 当前节点ip，需要修改: hostname -I | awk '{print $1}'
nic_name="eth0"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="172.21.100.74" # master节点ip，需要修改

export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_BUFFSIZE=2048
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export TASK_QUEUE_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export USE_MULTI_GROUPS_KV_CACHE=1
export USE_MULTI_BLOCK_POOL=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

vllm serve /data01/public/download/DeepSeek-V4-Pro-W4a8-mtp-0505 \
  --safetensors-load-strategy 'prefetch' \
  --max_model_len 135000  \
  --max-num-batched-tokens 4096 \
  --served-model-name dsv4 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 32 \
  --data-parallel-size 2 \ #需要修改
  --data-parallel-size-local 1 \ #需要修改
  --data-parallel-start-rank 0 \ #需要修改
  --data-parallel-address $node0_ip \
  --data-parallel-rpc-port 13399 \
  --tensor-parallel-size 16 \ #需要修改
  --enable-expert-parallel \
  --quantization ascend \
  --port 8900 \
  --host 0.0.0.0 \
  --block-size 128 \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --no-enable-prefix-caching \    # 关闭prefix-caching功能。
  --reasoning-parser deepseek_v4 \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"enable_cpu_binding": "true", "ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}}' \
  2>&1 | tee ./_dsv4_node0.log
```

node1
```bash
############
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
local_ip="172.21.100.75" # use: hostname -I | awk '{print $1}'
nic_name="eth0"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="172.21.100.74"

export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_BUFFSIZE=2048
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export TASK_QUEUE_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export USE_MULTI_GROUPS_KV_CACHE=1
export USE_MULTI_BLOCK_POOL=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

vllm serve /data01/public/download/DeepSeek-V4-Pro-W4a8-mtp-0505 \
  --safetensors-load-strategy 'prefetch' \
  --headless \
  --max_model_len 135000  \
  --max-num-batched-tokens 4096 \
  --served-model-name dsv4 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 32 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address $node0_ip \
  --data-parallel-rpc-port 13399 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --quantization ascend \
  --port 8900 \
  --host 0.0.0.0 \
  --block-size 128 \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --no-enable-prefix-caching \    # 关闭prefix-caching功能
  --reasoning-parser deepseek_v4 \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"enable_cpu_binding": "true", "ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}}' \
  2>&1 | tee /data01/qinzhengda/code/_dsv4_node1.log
```

curl:
```bash
curl http://127.0.0.1:8900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dsv4",
    "messages": [
      {"role": "user", "content": "你好,请简单介绍一下你自己。"}
    ],
    "max_tokens": 128,
    "temperature": 0,
    "stream": false
  }'
```

bench serve:
```bash
vllm bench serve   --backend openai-chat   --base-url http://127.0.0.1:8900  \
 --endpoint /v1/chat/completions   --dataset-name random   --model dsv4  \
 --served-model-name dsv4  --random-prefix-len \
 --tokenizer /data00/public/download/DeepSeek-V4-Pro-W4a8-mtp-0505  \
 --num-prompts 4   --num-warmups 0   --random-input-len 8170  \
 --random-output-len 5   --request-rate inf   --max-concurrency 4 
```

举例模板：
910_c2 ip: 172.21.100.73
910c_4 ip: 172.21.100.76
使用2机910c_2, 910c4并行方式：dp16, tp2, ep32。
加上这个额外配置：
```
--additional-config '{
    "finegrained_tp_config": {
      "oproj_tensor_parallel_size": 16,
      "embedding_tensor_parallel_size": 16,
      "lmhead_tensor_parallel_size": 16
      }
    }'
```

