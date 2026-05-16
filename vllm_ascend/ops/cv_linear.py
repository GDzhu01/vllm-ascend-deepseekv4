#
# CVLinearWrapper - 将 Linear 层拆分为 quantize(Vector) + matmul(Cube)
#
# 支持条件: forward 时无 TP 通信操作的 Linear 层
#   - AscendReplicatedLinear: custom_op=CustomReplicatedOp 或 None，无通信
#   - AscendColumnParallelLinear: 仅 gather_output=False 时可拆分（如 wq_b）
#   - 其他情况自动回退到完整 forward
#
import torch
import torch_npu

from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod


class CVLinearWrapper:
    """
    将 Linear 层拆分为 quantize(Vector) + matmul(Cube)

    自动检测 TP 通信操作：
    - 无通信 (ReplicatedLinear): W8A8 拆分为独立 quantize + matmul
    - 有通信 (ColumnParallelLinear 含 custom_op): 自动回退到完整 forward

    使用示例：
        wrapper = CVLinearWrapper(linear)

        # Step 1: 量化 (Vector)
        q_quant, q_scale = wrapper.quantize(x)

        # Step 2: 矩阵乘 (Cube)
        result = wrapper.matmul(q_quant, q_scale)
    """

    def __init__(self, linear):
        self.linear = linear

        # 检测是否有 TP 通信操作
        self._has_communication = self._detect_communication(linear)

        # 检测量化方案
        # 处理两种情况：
        # 1. linear.quant_method 直接是 AscendW8A8DynamicLinearMethod
        # 2. linear.quant_method 是包装类，需要通过 .quant_method 获取真正的量化方法
        self._quant_method = linear.quant_method #AscendW8A8DynamicLinearMethod
        self._is_w8a8_dynamic = self._detect_w8a8_dynamic(linear.quant_method)

    @staticmethod
    def _detect_w8a8_dynamic(quant_method):
        """检测量化方法是否为 W8A8 Dynamic"""
        # 情况1: quant_method 直接是 AscendW8A8DynamicLinearMethod
        if isinstance(quant_method, AscendW8A8DynamicLinearMethod):
            return True
        # 情况2: quant_method 是包装类，需要通过 .quant_method 获取
        if hasattr(quant_method, 'quant_method') and isinstance(quant_method.quant_method, AscendW8A8DynamicLinearMethod):
            return True
        return False

    @staticmethod
    def _detect_communication(linear):
        """
        检测 Linear 层 forward 时是否有 TP 通信操作。

        判断依据：
        - custom_op 为 None 或 CustomReplicatedOp：无 TP 通信
        - 其他 custom_op（MLPColumnParallelOp 含 all_gather）：有 TP 通信
        - ColumnParallelLinear gather_output=True：有 all-gather 通信
        注意：ColumnParallelLinear 即使 custom_op=None，仅当 gather_output=True 时有通信。
              wq_b 使用默认 gather_output=False，故无通信，可以拆分。
        """
        custom_op = getattr(linear, 'custom_op', None)
        if custom_op is not None:
            from vllm_ascend.ops.linear_op import CustomReplicatedOp
            if not isinstance(custom_op, CustomReplicatedOp):
                return True

        if hasattr(linear, 'gather_output') and linear.gather_output:
            return True

        return False

    def quantize(self, x: torch.Tensor):
        """
        仅执行量化步骤 (Vector算子)

        Args:
            x: 输入张量

        Returns:
            (quantized_x, pertoken_scale): 量化后的张量和缩放因子
            对于有通信的 linear 或无需量化的方案，返回 (x, None)
        """
        if self._has_communication:
            return x, None

        if self._is_w8a8_dynamic:
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
            return quantized_x, pertoken_scale
        else:
            return x, None

    def matmul(self, quantized_x: torch.Tensor, pertoken_scale=None, bias=None):
        """
        仅执行矩阵乘步骤 (Cube算子)

        Args:
            quantized_x: 量化后的输入（有通信时直接传入原始输入）
            pertoken_scale: W8A8_DYNAMIC 的 per-token 缩放因子
            bias: 偏置

        Returns:
            矩阵乘结果
        """
        if self._has_communication:
            return self.linear.forward(quantized_x)

        if self._is_w8a8_dynamic:
            need_unsqz = False
            if pertoken_scale is not None and pertoken_scale.dim() == 2:
                need_unsqz = True
                quantized_x = quantized_x.squeeze(dim=1)
                pertoken_scale = pertoken_scale.squeeze(dim=1)

            output = torch_npu.npu_quant_matmul(
                quantized_x,
                self.linear.weight,
                self.linear.weight_scale,
                pertoken_scale=pertoken_scale,
                bias=bias,
                output_dtype=self.linear.weight_scale.dtype,
            )

            if need_unsqz:
                output = output.unsqueeze(dim=1)
            return output
        else:
            return self.linear.quant_method.apply(self.linear, quantized_x, bias)

    def forward(self, x: torch.Tensor, bias=None):
        """完整的 forward（等效于原始 Linear.forward）"""
        q_quant, q_scale = self.quantize(x)
        return self.matmul(q_quant, q_scale, bias)

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        self.linear.weight = value

    def __getattr__(self, name):
        """将未定义的属性委托给内部的 linear 对象"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.linear, name)
