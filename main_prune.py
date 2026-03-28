import torch
import yaml
import gc
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm

# --- 1. 高性能 PPL 评估 (利用 A100 显存) ---
@torch.inference_mode()
def evaluate_perplexity(model, tokenizer, dataset_text, seq_len=2048, device='cuda'):
    model.eval()
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    nlls = []
    # A100 支持大 batch 或长序列，这里跳步增加
    for i in tqdm(range(0, input_ids.size(1), seq_len), desc="A100 高速评估"):
        j = min(i + seq_len, input_ids.size(1))
        input_ptr = input_ids[:, i:j]
        target_ids = input_ptr.clone()
        
        outputs = model(input_ptr, labels=target_ids)
        nlls.append(outputs.loss * (j - i))

    ppl = torch.exp(torch.stack(nlls).sum() / j)
    return ppl.item()

# --- 2. 物理权重置零 (针对 FP16 权重) ---
def apply_physical_masking(model, selected_ids):
    """
    在 A100 上直接操作权重数据，无需 Hook
    """
    num_layers = 32
    num_heads = 32
    head_dim = 128

    for i in range(num_layers):
        layer = model.model.layers[i]
        
        # 注意力头物理置零
        for h in range(num_heads):
            if f"layer_{i}_attn_head_{h}" not in selected_ids:
                # 找到 O 矩阵中对应头的列索引，直接清零权重
                start = h * head_dim
                end = (h + 1) * head_dim
                layer.self_attn.o_proj.weight.data[:, start:end] = 0.0
        
        # MLP 物理置零
        if f"layer_{i}_mlp" not in selected_ids:
            layer.mlp.down_proj.weight.data.fill_(0.0)
            layer.mlp.up_proj.weight.data.fill_(0.0)

# --- 3. 主程序 (无量化加载) ---
def main():
    model_path = "/path/to/your/llama-2-7b-hf" # 服务器上的绝对路径
    
    print(f"正在 A100 上以 bfloat16 加载原始模型...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # A100 原生支持 bfloat16，精度比 float16 更稳
    model = LlamaForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    # 加载得分并生成配置 (使用你之前修正的 generate_pruning_config)
    # ... (此处省略加载 lcb_scores.pt 的逻辑) ...

    # 执行物理剪枝
    apply_physical_masking(model, selected_ids)
    
    # 高精度评估
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_text = "\n\n".join(test_data["text"])
    ppl = evaluate_perplexity(model, tokenizer, test_text, seq_len=2048)
    print(f"A100 原始精度剪枝后 PPL: {ppl:.4f}")

if __name__ == "__main__":
    main()