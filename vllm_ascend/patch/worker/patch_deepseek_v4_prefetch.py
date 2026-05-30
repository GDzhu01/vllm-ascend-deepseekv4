import torch

from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2DecoderLayer,
    DeepseekV4Model,
)

import vllm_ascend.prefetch

_original_decoder_forward = DeepseekV2DecoderLayer.forward


def _patched_decoder_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    llama_4_scaling: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from vllm_ascend.envs import (
            VLLM_PREFETCH,
            VLLM_PREFETCH_LOG,
            VLLM_PREFETCH_WEIGHT_SIZE_LIMIT,
            VLLM_PREFETCH_WEIGHTS,
        )

        if VLLM_PREFETCH_LOG:
            print(
                f"[prefetch] ENTER forward layer={getattr(self, 'layer_idx', '?')}"
            )

        if VLLM_PREFETCH and hasattr(self, '_prefetch_enabled'):
            pw = set(
                w.strip() for w in VLLM_PREFETCH_WEIGHTS.split(",") if w.strip()
            )
            max_size = VLLM_PREFETCH_WEIGHT_SIZE_LIMIT

            residual = hidden_states.clone()
            hidden_states, post_attn, comb_attn = self.hc_pre(
                hidden_states, self.hc_attn_fn, self.hc_attn_scale,
                self.hc_attn_base
            )
            hidden_states = self.input_layernorm(hidden_states)

            if "gate" in pw:
                gate_w = self.mlp.gate.weight
                torch.ops.vllm.prefetch_after_attn(
                    hidden_states, gate_w, max_size,
                    self.layer_idx, f"layer.{self.layer_idx}.moe.gate",
                )

            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                llama_4_scaling=llama_4_scaling,
            )

            torch.ops.vllm.prefetch_sync(hidden_states, self.layer_idx)

            hidden_states = self.hc_post(
                hidden_states, residual, post_attn, comb_attn
            )
            residual = hidden_states.clone()
            hidden_states, post_ffn, comb_ffn = self.hc_pre(
                hidden_states, self.hc_ffn_fn, self.hc_ffn_scale,
                self.hc_ffn_base
            )
            hidden_states = self.post_attention_layernorm(hidden_states)

            hidden_states = self.mlp(hidden_states)

            if "next_qkv" in pw:
                next_weights = getattr(self, '_prefetch_next_qkv_weight', None)
                if next_weights is not None:
                    for wname, wt in next_weights:
                        torch.ops.vllm.prefetch_after_mlp(
                            hidden_states, wt, max_size,
                            self.layer_idx,
                            f"layer.{self.layer_idx}.next_{wname}",
                        )

            hidden_states = self.hc_post(
                hidden_states, residual, post_ffn, comb_ffn
            )

            if VLLM_PREFETCH_LOG:
                has_gate = "gate" in pw
                has_next_qkv = (
                    "next_qkv" in pw
                    and hasattr(self, "_prefetch_next_qkv_weight")
                )
                print(
                    f"[prefetch] layer={self.layer_idx} "
                    f"gate={has_gate} next_qkv={has_next_qkv}"
                )

            return hidden_states, residual

        return _original_decoder_forward(
            self, positions, hidden_states, residual, llama_4_scaling
        )
    except Exception as e:
        from vllm_ascend.envs import VLLM_PREFETCH_LOG
        if VLLM_PREFETCH_LOG:
            print(
                f"[prefetch] ERROR in patched forward "
                f"layer={getattr(self, 'layer_idx', '?')}: {e}"
            )
        return _original_decoder_forward(
            self, positions, hidden_states, residual, llama_4_scaling
        )


_original_model_init = DeepseekV4Model.__init__


def _patched_model_init(self, *, vllm_config, prefix=""):
    _original_model_init(self, vllm_config=vllm_config, prefix=prefix)

    from vllm_ascend.envs import VLLM_PREFETCH, VLLM_PREFETCH_LOG
    if not VLLM_PREFETCH:
        return

    if VLLM_PREFETCH_LOG:
        print(f"[prefetch] Initializing prefetch for {len(self.layers)} layers")

    for i, layer in enumerate(self.layers):
        layer._prefetch_enabled = True
        next_layer = self.layers[i + 1] if i + 1 < len(self.layers) else None
        if next_layer is not None:
            next_attn = next_layer.self_attn
            if VLLM_PREFETCH_LOG and i == 0:
                attn_attrs = [
                    a for a in dir(next_attn)
                    if not a.startswith("_") and not callable(getattr(next_attn, a))
                ]
                print(f"[prefetch] DIAG layer=0 attn_attrs={attn_attrs}")
            weights = []
            for attr_name in ("wq_b", "wkv", "wq_a", "wo_a", "wo_b"):
                linear = getattr(next_attn, attr_name, None)
                if VLLM_PREFETCH_LOG and i == 0:
                    print(
                        f"[prefetch] DIAG attr={attr_name} "
                        f"exists={linear is not None} "
                        f"has_weight={hasattr(linear, 'weight') if linear else 'N/A'}"
                    )
                if linear is not None and hasattr(linear, 'weight'):
                    weights.append((attr_name, linear.weight))
            if VLLM_PREFETCH_LOG and i == 0:
                print(f"[prefetch] DIAG layer=0 weights_count={len(weights)}")
            if weights:
                object.__setattr__(
                    layer, '_prefetch_next_qkv_weight', weights,
                )
                if VLLM_PREFETCH_LOG:
                    total = sum(
                        wt.element_size() * wt.numel() for _, wt in weights
                    )
                    names = [n for n, _ in weights]
                    print(
                        f"[prefetch] layer={i} next_attn "
                        f"weights={names} total_size={total} bytes"
                    )


DeepseekV2DecoderLayer.forward = _patched_decoder_forward
DeepseekV4Model.__init__ = _patched_model_init
