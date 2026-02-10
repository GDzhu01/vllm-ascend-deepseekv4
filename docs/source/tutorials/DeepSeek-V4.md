# DeepSeek-V4

## Introduction

DeepSeek-V4 is introducing several key upgrades over DeepSeek-V3. (1) The Manifold-Constrained Hyper-Connections (mHC)  to strengthen conventional residual connections; (2) A hybrid attention architecture, which greatly improves long-context efficiency through Compress-4-Attention and Compress-128-Attention. For the Mixture-of Experts (MoE) components, it still adopt the DeepSeekMoE architecture, with only minor adjustments.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## Environment Preparation

### Model Weight
- `DeepSeek-V4-w8a8`(Quantized version): require 1 Atlas 800 A3 (64G × 16) node or 1 Atlas 800 A2 (64G × 8) node. [Download model weight](https://modelers.cn/models/Modelers_Park/DeepSeek-V4)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../installation.md#verify-multi-node-communication).

### Installation

You can using our official docker image to run `DeepSeek-V4` directly. Currently, `DeepSeek-V4` is integrated in image `v0.13.0rc3-pre`.

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

Start the docker image on your each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:v0.13.0rc3-a3-pre
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::
::::{tab-item} A2 series
:sync: A2

Start the docker image on your each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:v0.13.0rc3-pre
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::
:::::

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../installation.md).
If you want to deploy multi-node environment, you need to set up environment on each node.

:::{note}
Please use the v0.13.0rc3 code to install vllm-ascend.
:::

## Deployment

:::{note}
In this tutorial, we suppose you downloaded the model weight to `/root/.cache/`. Feel free to change it to your own path.
:::

### Single-node Deployment

- `DeepSeek-V4-w8a8`: can be deployed on 1 Atlas 800 A3 (64G × 16) or 1 Atlas 800 A2 (64G × 8).

Run the following scripts on each node respectively. 

:::::{tab-set}
:sync-group: install

::::{tab-item} A2 series
:sync: A2

Run the following script to execute online inference.

```shell
export USE_MULTI_BLOCK_POOL=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export TRITON_ALL_BLOCKS_PARALLEL=1

vllm serve /mnt/weights/hello2026_w8a8_dynamic_m6_quarot_quant-mtp_real \
  --host 0.0.0.0 \
  --max_model_len 65536 \
  --max-num-batched-tokens 65536 \
  --served-model-name ds \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 16 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --port 8006 \
  --block-size 128 \
  --no-enable-chunked-prefill \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": true}' \
  --speculative-config '{"num_speculative_tokens": 2,"method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

::::

::::{tab-item} A3 series
:sync: A3

Run the following script to execute online inference.

```shell
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10  
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1
export HCCL_BUFFSIZE=4650
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-W8A8 \
    --host 0.0.0.0 \
    --max_model_len 10240 \
    --max-num-batched-tokens 10240 \
    --served-model-name deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 16 \
    --data-parallel-size 2 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp"}' \
    --port 8005 \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
    --no-enable-chunked-prefill \
    --async-scheduling \
    --additional-config '{"enable_cpu_binding": "true","multistream_overlap_shared_expert": true}'
```
::::
:::::


## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v4",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 256,
        "temperature": 0
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### Using Language Model Evaluation Harness

As an example, take the `gsm8k` dataset as a test dataset, and run accuracy evaluation of `DeepSeek-V4` in online mode.

1. Refer to [Using lm_eval](../developer_guide/evaluation/using_lm_eval.md) for `lm_eval` installation.

2. Run `lm_eval` to execute the accuracy evaluation.

```shell
lm_eval \
  --model local-completions \
  --model_args model=/root/.cache/Eco-Tech/DeepSeek-V4-w8a8,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

3. After execution, you can get the result.

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-V4-W8A8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model /root/.cache/Eco-Tech/DeepSeek-V4-w8a8  --dataset-name random --random-input 200 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```



