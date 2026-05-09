#!/bin/bash

ROOT_DIR=$1
SOC_VERSION=$2

log() {
    echo "[build_aclnn] $*"
}

find_op_dir() {
    local op_name=$1
    local dir
    for dir in \
        "${ROOT_DIR}/csrc/moe/${op_name}" \
        "${ROOT_DIR}/csrc/gmm/${op_name}" \
        "${ROOT_DIR}/csrc/attention/${op_name}" \
        "${ROOT_DIR}/csrc/mc2/${op_name}" \
        "${ROOT_DIR}/csrc/ffn/${op_name}" \
        "${ROOT_DIR}/csrc/posembedding/${op_name}"; do
        if [[ -d "${dir}" ]]; then
            echo "${dir}"
            return 0
        fi
    done
    find "${ROOT_DIR}/csrc" -maxdepth 3 -type d -name "${op_name}" -print -quit 2>/dev/null
}

dump_selected_ops() {
    local op_name
    local op_dir
    local kernel_cpp_count

    log "resolved SOC_ARG=${SOC_ARG}"
    log "resolved CUSTOM_OPS=${CUSTOM_OPS}"
    log "custom op count=${#CUSTOM_OPS_ARRAY[@]}"
    for op_name in "${CUSTOM_OPS_ARRAY[@]}"; do
        op_dir=$(find_op_dir "${op_name}")
        if [[ -z "${op_dir}" ]]; then
            log "op ${op_name}: dir=<missing>"
            continue
        fi
        kernel_cpp_count=0
        if [[ -d "${op_dir}/op_kernel" ]]; then
            kernel_cpp_count=$(find "${op_dir}/op_kernel" -maxdepth 1 -name '*.cpp' | wc -l | tr -d ' ')
        fi
        log "op ${op_name}: dir=${op_dir} cmake=$([[ -f "${op_dir}/CMakeLists.txt" ]] && echo yes || echo no) op_host_cmake=$([[ -f "${op_dir}/op_host/CMakeLists.txt" ]] && echo yes || echo no) op_kernel_cpp_count=${kernel_cpp_count}"
    done
}

log "start: ROOT_DIR=${ROOT_DIR:-<unset>} SOC_VERSION=${SOC_VERSION:-<unset>} cwd=$(pwd)"
log "env: ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>} ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"

if [[ "$SOC_VERSION" =~ ^ascend310 ]]; then
    log "matched SOC branch: ascend310"
    # ASCEND310P series
    CUSTOM_OPS_ARRAY=(
        "causal_conv1d_v310"
        "recurrent_gated_delta_rule_v310"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend310p"
elif [[ "$SOC_VERSION" =~ ^(ascend)?910b ]]; then
    log "matched SOC branch: ascend910b"
    # ASCEND910B (A2) series
    # dependency: catlass
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/csrc/third_party/catlass/include
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
    fi
    ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
    export CPATH=${ABSOLUTE_CATLASS_PATH}:${CPATH}
    log "catlass include=${ABSOLUTE_CATLASS_PATH}"

    CUSTOM_OPS_ARRAY=(
        "dequant_swiglu_quant"
        "scatter_nd_update_v2"
        "moe_grouped_matmul"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "lightning_indexer_vllm"
        "sparse_flash_attention"
        "matmul_allreduce_add_rmsnorm"
        "moe_init_routing_custom"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "apply_top_k_top_p_custom"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "lightning_indexer_quant"
        "compressor"
        "quant_lightning_indexer"
        "quant_lightning_indexer_metadata"
        "lightning_indexer_quant_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_post"
        "rms_norm_dynamic_quant"
        "inplace_partial_rotary_mul"
        "grouped_matmul_swiglu_quant"
    )

    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    log "matched SOC branch: ascend910_93"
    # ASCEND910C (A3) series
    # dependency: catlass
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/csrc/third_party/catlass/include
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
    fi
    # dependency: cann-toolkit file moe_distribute_base.h
    HCCL_STRUCT_FILE_PATH=$(find -L "${ASCEND_TOOLKIT_HOME}" -name "moe_distribute_base.h" 2>/dev/null | head -n1)
    if [ -z "$HCCL_STRUCT_FILE_PATH" ]; then
        echo "cannot find moe_distribute_base.h file in CANN env"
        exit 1
    fi
    # for dispatch_gmm_combine_decode
    yes | cp "${HCCL_STRUCT_FILE_PATH}" "${ROOT_DIR}/csrc/utils/inc/kernel"
    # for dispatch_ffn_combine
    SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
    TARGET_DIR="$SCRIPT_DIR/mc2/dispatch_ffn_combine/op_kernel/utils/"
    TARGET_FILE="$TARGET_DIR/$(basename "$HCCL_STRUCT_FILE_PATH")"

    echo "*************************************"
    echo $HCCL_STRUCT_FILE_PATH
    echo "$TARGET_DIR"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"

    sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE"
    sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE"

    TARGET_DIR="$SCRIPT_DIR/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/"
    TARGET_FILE="$TARGET_DIR/$(basename "$HCCL_STRUCT_FILE_PATH")"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"
    sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE"
    sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE"

    TARGET_DIR="$SCRIPT_DIR/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/"
    TARGET_FILE="$TARGET_DIR/$(basename "$HCCL_STRUCT_FILE_PATH")"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"
    sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE"
    sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE"

    # for dispatch_normal and combine_normal
    TARGET_DIR="$SCRIPT_DIR/mc2/moe_dispatch_normal/op_kernel/utils/"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"

    TARGET_DIR="$SCRIPT_DIR/mc2/moe_combine_normal/op_kernel/utils/"
    echo "$TARGET_DIR"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"
    
    CUSTOM_OPS_ARRAY=(
        "dequant_swiglu_quant"
        "scatter_nd_update_v2"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "lightning_indexer_vllm"
        "sparse_flash_attention"
        "dispatch_ffn_combine"
        "dispatch_ffn_combine_w4_a8"
        "dispatch_ffn_combine_bf16"
        "dispatch_gmm_combine_decode"
        "moe_combine_normal"
        "moe_dispatch_normal"
        "dispatch_layout"
        "notify_dispatch"
        "moe_init_routing_custom"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "apply_top_k_top_p_custom"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "moe_grouped_matmul"
        "lightning_indexer_quant"
        "compressor"
        "quant_lightning_indexer"
        "quant_lightning_indexer_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_post"
        "rms_norm_dynamic_quant"
        "inplace_partial_rotary_mul"
        "grouped_matmul_swiglu_quant"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_93"
else
    # others
    # currently, no custom aclnn ops for other series
    log "no custom ACLNN ops configured for SOC_VERSION=${SOC_VERSION}; skip build_aclnn"
    exit 0
fi

dump_selected_ops

# # build custom ops
# cd csrc
# rm -rf build output build_out
# echo "building custom ops $CUSTOM_OPS for $SOC_VERSION"
# bash build.sh --pkg --ops="$CUSTOM_OPS" --soc="$SOC_ARG"

# # install custom ops to vllm_ascend/_cann_ops_custom
# ./build/cann-ops-transformer*.run --install-path=$ROOT_DIR/vllm_ascend/_cann_ops_custom


(
  set -euo pipefail

  log "subshell cwd before cd=$(pwd)"
  cd csrc
  log "subshell cwd after cd=$(pwd)"
  log "cleaning csrc build dirs"
  rm -rf -- build output build_out

  : "${ROOT_DIR:?ROOT_DIR is not set}"
  : "${CUSTOM_OPS:?CUSTOM_OPS is not set}"
  : "${SOC_VERSION:?SOC_VERSION is not set}"
  : "${SOC_ARG:?SOC_ARG is not set}"

  log "build command: bash build.sh --pkg --ops=\"${CUSTOM_OPS}\" --soc=\"${SOC_ARG}\""
  log "building custom ops ${CUSTOM_OPS} for ${SOC_VERSION}"
  bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}"
  log "build.sh finished"

  install_dir="${ROOT_DIR}/vllm_ascend/_cann_ops_custom"
  log "install_dir=${install_dir}"

  mkdir -p -- "$install_dir"

  # 删除 install_dir 下除 .gitkeep 外的所有内容（包含隐藏文件/目录）
  find "$install_dir" -mindepth 1 -maxdepth 1 \
    ! -name '.gitkeep' \
    -exec rm -rf -- {} +

  shopt -s nullglob
  runs=(./build/cann-ops-transformer*.run)
  shopt -u nullglob

  log "installer candidate count=${#runs[@]}"
  for run_file in "${runs[@]}"; do
    log "installer candidate: $(ls -lh "${run_file}")"
  done

  (( ${#runs[@]} == 1 )) || { echo "ERROR: expected 1 installer, got ${#runs[@]}" >&2; exit 1; }

  chmod +x -- "${runs[0]}" || true
  log "running installer: ${runs[0]}"
  "${runs[0]}" --install-path="${install_dir}"
  log "installer finished"
  log "installed files under ${install_dir} (maxdepth=4, first 120 entries):"
  { find "${install_dir}" -mindepth 1 -maxdepth 4 -print | sort | head -n 120 | sed 's#^#[build_aclnn] install: #'; } || true
)
