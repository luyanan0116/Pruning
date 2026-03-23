import torch

def register_llama_hooks(model, captured_grads):
    """为 Llama 2 的注意力头和 MLP 注册梯度钩子"""
    def hook_fn(module, grad_input, grad_output, name):
        captured_grads[name] = grad_output[0].detach()

    for i, layer in enumerate(model.model.layers):
        # 拦截注意力层 O 矩阵的梯度 (公式 1) [cite: 204]
        layer.self_attn.o_proj.register_full_backward_hook(
            lambda m, gi, go, i=i: hook_fn(m, gi, go, f"layer_{i}_attn")
        )
        # 拦截 MLP 层的梯度 (公式 3) [cite: 210]
        layer.mlp.down_proj.register_full_backward_hook(
            lambda m, gi, go, i=i: hook_fn(m, gi, go, f"layer_{i}_mlp")
        )

def split_llama_heads(grads, num_heads=32):
    """将隐藏层梯度切分为独立的注意力头单元"""
    # Llama 2-7B: 4096 -> 32 heads * 128 dim
    head_dim = grads.shape[-1] // num_heads
    return [grads[..., h*head_dim : (h+1)*head_dim] for h in range(num_heads)]