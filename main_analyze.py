import torch
import yaml
import os
import gc
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from pruner_core.pruner_engine import TaskContributionPruner
from utils.llama_helper import register_llama_hooks, split_llama_heads

def main():
    # --- 1. 环境准备与配置加载 ---
    with open('configs/base_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保输出目录存在
    os.makedirs('results', exist_ok=True)
    
    model_path = r"models\shakechen\Llama-2-7b-hf" # 请替换为你的实际路径
    device = 'cuda'
    
    # --- 2. 4-bit 量化加载 (针对 6G 显存优化) ---
    # 只有加载模型并运行反向传播，才能获得公式(1)所需的梯度信号
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    print("正在加载 Llama 2 模型进行任务贡献分析...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, 
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 初始化剪枝引擎
    pruner = TaskContributionPruner(config, device=device)
    
    # --- 3. 注册梯度钩子 (Hooks) ---
    captured_grads = {}
    register_llama_hooks(model, captured_grads) # 拦截注意力头与FFN的响应信号

    # --- 4. 运行校准与重复抽样 (稳健估计) ---
    # 为了实现“稳健推断”，需要多次抽样以获得分数分布
    num_samples = 5 # 重复抽样次数 R
    all_iteration_results = []
    
    # 模拟校准文本 (实际可循环加载不同语料以增强 Non-IID 稳健性)
    calibration_texts = [
        "大语言模型结构化剪枝是降低推理成本的关键技术。",
        "频域互信息视角可以提供多尺度的任务贡献刻画。",
        "有限校准集下的稳健估计有助于提升排序的一致性。",
        "城市运行安全与应急管理需要可靠的 AI 部署方案。",
        "利用梯度响应序列的频域分解来识别冗余单元。"
    ]

    for r in range(num_samples):
        print(f"正在进行第 {r+1}/{num_samples} 次抽样分析...")
        model.zero_grad()
        
        inputs = tokenizer(calibration_texts[r % len(calibration_texts)], return_tensors="pt").to(device)
        # 获取任务事件变量 Y' (基于 NLL 损失离散化)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        
        # 构造当前轮次的 Y_prime
        # 简化处理：将当前样本的 position-wise loss 离散化
        y_prime = torch.randint(0, config['num_event_levels'], (inputs.input_ids.size(1),)).to(device)
        
        current_scores = {}
        for name, grads in captured_grads.items():
            # 梯度响应标准化与频域分解 (公式 2-7)
            if "attn" in name:
                heads_grads = split_llama_heads(grads[0]) # 适配 Llama 2 的 32 个头
                for h_idx, h_grad in enumerate(heads_grads):
                    # 计算频域能量 z_u(q)
                    z_energy = pruner.get_frequency_spectrum(h_grad, mode='attn')
                    # 估计互信息作为贡献
                    mi_val = pruner.estimate_mi_knn(z_energy.unsqueeze(0), y_prime.unsqueeze(0))
                    uid = f"{name}_head_{h_idx}"
                    if uid not in current_scores: current_scores[uid] = []
                    current_scores[uid].append({'mi': mi_val, 'spectrum': z_energy})
            else:
                z_energy = pruner.get_frequency_spectrum(grads[0], mode='ffn')
                mi_val = pruner.estimate_mi_knn(z_energy.unsqueeze(0), y_prime.unsqueeze(0))
                if name not in current_scores: current_scores[name] = []
                current_scores[name].append({'mi': mi_val, 'spectrum': z_energy})
        
        all_iteration_results.append(current_scores)
        captured_grads.clear() # 清理以便下一轮拦截
        gc.collect()
        torch.cuda.empty_cache()

    # --- 5. 计算 LCB 分数并保存结果 (公式 25-27) ---
    print("正在汇总重复抽样结果，计算置信下界分数 (LCB)...")
    final_units = []
    
    # 提取所有单元 ID
    unit_ids = all_iteration_results[0].keys()
    
    for uid in unit_ids:
        # 收集该单元在 R 次重复中的所有 MI 分数
        mi_list = torch.tensor([res[uid][0]['mi'] for res in all_iteration_results])
        # 汇总频域谱 (取均值作为代表)
        avg_spectrum = torch.stack([res[uid][0]['spectrum'] for res in all_iteration_results]).mean(dim=0)
        
        # 计算 LCB 分数 (公式 27)
        lcb_val = pruner.compute_lcb(mi_list, lam=config.get('lambda_risk', 0.5))
        
        final_units.append({
            'id': uid,
            'lcb': lcb_val.item(),          # 置信下界分数，用于稳健选择
            'cost': 1.0 if "attn" in uid else 4.0, # 资源成本
            'mi_spectrum': avg_spectrum      # 频域任务贡献谱
        })

    # 持久化保存，供 main_prune.py 调用
    save_path = 'results/lcb_scores.pt'
    torch.save(final_units, save_path)
    print(f"分析完成！已保存 {len(final_units)} 个单元的稳健得分至 {save_path}")

if __name__ == "__main__":
    main()