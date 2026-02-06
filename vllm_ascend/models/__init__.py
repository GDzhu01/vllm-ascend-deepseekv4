from vllm import ModelRegistry

def register_model():
    ModelRegistry.register_model(
    "DeepSeekNewForCausalLM",
    "vllm_ascend.models.deepseek_new:AscendDeepSeekNewForCausalLM")

    ModelRegistry.register_model(
    "DeepSeekNewMTPModel",
    "vllm_ascend.models.deepseek_new_mtp:DeepSeekNewMTP")
