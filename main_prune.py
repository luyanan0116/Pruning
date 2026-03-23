import torch
import yaml
import gc
import os
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
from pruner_core.pruner_engine import TaskContributionPruner

# --- 函数 1：WikiText 语料加载 ---
def get_wikitext_test_corpus():
    """加载 WikiText-2 测试集并合并为连续文本"""
    print("正在加载 WikiText-2 测试集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([line for line in dataset["text"] if len(line.strip()) > 0])
    return full_text

# --- 函数 2：自动化 PPL 评估 (针对 6G 显存优化) ---
@torch.inference_mode()
def evaluate_perplexity(model, tokenizer, dataset_text, seq_len=128, device='cuda'):
    """计算模型在给定文本上的困惑度 (PPL)"""
    model.eval()
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    nlls = []
    prev_end_loc = 0
    # 使用较小的 seq_len (128) 以防止 6G 显存 OOM
    for begin_loc in tqdm(range(0, input_ids.size(1), seq_len), desc="评估 PPL"):
        end_loc = min(begin_loc + seq_len, input_ids.size(1))
        trg_len = end_loc - prev_end_loc
        input_ptr = input_ids[:, begin_loc:end_loc]
        target_ids = input_ptr.clone()
        target_ids[:, :-trg_len] = -100 
        
        outputs = model(input_ptr, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == input_ids.size(1): break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

# --- 函数 3：物理剪枝执行 (基于 Masking) ---
def apply_structured_masking(model, selected_ids):
    """
    针对 4bit 量化模型的动态屏蔽方案 (修复 Hook 参数版本)
    """
    num_layers = 32
    num_heads = 32
    head_dim = 128

    def get_activation_mask_pre_hook(mask):
        # Pre-hook 只有 module 和 input 两个参数
        # input 是一个 tuple，input[0] 才是真正的张量
        def hook(module, input):
            # 将注意力机制输出的隐藏状态与掩码相乘，抹除被剪枝的头
            return (input[0] * mask,) # 必须返回 tuple
        return hook

    for i in range(num_layers):
        layer = model.model.layers[i]
        
        # 1. 构造注意力头掩码 [4096]
        attn_mask = torch.ones(num_heads * head_dim, device='cuda', dtype=torch.float16)
        for h in range(num_heads):
            if f"layer_{i}_attn_head_{h}" not in selected_ids:
                attn_mask[h * head_dim : (h + 1) * head_dim] = 0.0
        
        # 2. 挂载 Pre-hook 到 o_proj 
        # 当模型运行到 o_proj 时，会自动调用 hook 抹掉没被选中的头
        layer.self_attn.o_proj.register_forward_pre_hook(get_activation_mask_pre_hook(attn_mask))
        
        # 3. 处理 MLP (如果该层 MLP 被剪掉)
        if f"layer_{i}_mlp" not in selected_ids:
            layer.mlp.register_forward_pre_hook(lambda m, inp: (inp[0] * 0.0,))

    print("已修正 Hook 定义，物理屏蔽激活。")

# --- 主程序入口 ---
def main():
    # A. 配置与路径 (请根据实际情况修改)
    with open('configs/base_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_path = r"models\shakechen\Llama-2-7b-hf"
    
    # B. 加载模型 (4-bit 量化适配 6G 显存)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    print("正在加载模型...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, 
        quantization_config=bnb_config,
        device_map="auto"
    )

    # C. 加载分析结果 (main_analyze.py 生成的 .pt)
    if not os.path.exists('results/lcb_scores.pt'):
        print("错误：未找到分析结果。请先运行 main_analyze.py。")
        return
    units = torch.load('results/lcb_scores.pt')

    # D. 生成配置并执行剪枝
    pruner_engine = TaskContributionPruner(config, device='cuda')
    total_budget = len(units) * config['resource_budget']
    
    selected_units = pruner_engine.generate_pruning_config(
        units, total_budget, 
        gamma=config.get('gamma_threshold', 0.7), 
        alpha=config.get('alpha_comp', 1.0)
    )
    selected_ids = {u['id'] for u in selected_units}

    print(f"执行剪枝，保留单元: {len(selected_ids)}/{len(units)}")
    apply_structured_masking(model, selected_ids)

    # E. 验证 PPL
    test_text = get_wikitext_test_corpus()
    ppl = evaluate_perplexity(model, tokenizer, test_text[:30000], seq_len=128)
    print(f"剪枝后 WikiText-2 PPL: {ppl:.4f}")


if __name__ == "__main__":
    main()