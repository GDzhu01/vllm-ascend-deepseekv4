from vllm import ModelRegistry

def register_model():
    ModelRegistry.register_model(
    "DeepseekNewForCausalLM",
    "vllm_ascend.models.deepseek_new:AscendDeepseekNewForCausalLM")

    ModelRegistry.register_model(
    "DeepseekNewMTPModel",
    "vllm_ascend.models.deepseek_new_mtp:DeepseekNewMTP")
