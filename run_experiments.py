import torch
import json
import os
import gc
import yaml
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
# 确保从你的 main_prune.py 中导入核心函数
from main_prune import (
    get_wikitext_test_corpus, 
    evaluate_perplexity, 
    apply_structured_masking
)
from pruner_core.pruner_engine import TaskContributionPruner

def run_batch_experiments():
    # 1. 基础配置加载
    model_path = r"models\shakechen\Llama-2-7b-hf"
    with open('configs/base_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 实验范围：从 0.9 到 0.7，步长 0.05
    retention_rates = [0.9, 0.85, 0.8, 0.75, 0.7]
    results = []
    
    # 加载分析数据 (main_analyze 生成的得分)
    if not os.path.exists('results/lcb_scores.pt'):
        print("错误：未找到 lcb_scores.pt，请先运行 main_analyze.py")
        return
    units = torch.load('results/lcb_scores.pt')
    
    # 加载验证集 (只加载一次)
    test_text = get_wikitext_test_corpus()
    # 为了加快速度，测试前 20000 字符即可
    val_text = test_text[:20000]

    for rate in retention_rates:
        print(f"\n" + "="*50)
        print(f"开始实验：保留率 {rate}")
        print("="*50)
        
        # A. 4-bit 模型加载 (确保每轮都是干净的模型)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path, 
            quantization_config=bnb_config,
            device_map="auto"
        )

        # B. 生成剪枝配置
        pruner = TaskContributionPruner(config, device='cuda')
        total_units = len(units)
        target_budget = total_units * rate
        
        selected_units = pruner.generate_pruning_config(
            units, 
            target_budget, 
            gamma=config.get('gamma_threshold', 0.5),
            alpha=config.get('alpha_comp', 1.0)
        )
        selected_ids = {u['id'] for u in selected_units}
        
        # C. 执行物理剪枝 (Hooks)
        apply_structured_masking(model, selected_ids)
        
        # D. 评估 PPL
        try:
            ppl = evaluate_perplexity(model, tokenizer, val_text, seq_len=128)
            print(f"保留率 {rate} -> 得到 PPL: {ppl:.4f}")
            
            results.append({
                "retention_rate": rate,
                "num_units": len(selected_ids),
                "ppl": ppl
            })
        except Exception as e:
            print(f"实验 {rate} 出错: {e}")
        
        # E. 关键：显存彻底清理
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"已清理显存，准备下一轮...")

    # 2. 保存最终结果
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_history.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n所有实验完成！结果已保存至 results/experiment_history.json")
    
    # 打印对比简报
    print("\n" + "对比简报".center(30, "-"))
    print("保留率 | 保留单元 | PPL")
    for r in results:
        print(f"{r['retention_rate']:>6.2f} | {r['num_units']:>8} | {r['ppl']:>8.2f}")

if __name__ == "__main__":
    run_batch_experiments()