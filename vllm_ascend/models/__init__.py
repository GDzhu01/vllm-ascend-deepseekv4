from vllm import ModelRegistry

def register_model():
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
    ModelRegistry.register_model(
    "DeepseekV4ForCausalLM",
    "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")
    print(f'ohhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh1')