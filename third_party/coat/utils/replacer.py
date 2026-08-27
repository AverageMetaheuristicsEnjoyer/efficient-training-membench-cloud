import torch
from ..activation.deepseek.fp8linear import FP8DeepSeekLinear

def fp8_deepseek_linear_replacer(model, linear_names: list = []):
    if linear_names == []:
        linear_names.extend(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        linear_names.extend(["attn_out", "ff_out", "att_proj", "ff_proj"])

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(key in name for key in linear_names):
            new_module = FP8DeepSeekLinear(module.in_features, module.out_features)
            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)
                
            setattr(model, name, new_module)
    return model