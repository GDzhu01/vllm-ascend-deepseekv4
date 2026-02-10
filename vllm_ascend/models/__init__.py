from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model(
        "DeepseekXYZForCausalLM",
        "vllm_ascend.models.deepseek_xyz:AscendDeepseekXYZForCausalLM")

    ModelRegistry.register_model(
        "DeepSeekXYZMTPModel",
        "vllm_ascend.models.deepseek_xyz_mtp:DeepSeekXYZMTP")
