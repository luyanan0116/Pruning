import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml
import os
import math
import gc
from transformers import LlamaForCausalLM, LlamaTokenizer, default_data_collator
from datasets import load_dataset
from tqdm import tqdm

# --- 1. 高速分布式环境初始化 ---
def setup_distributed():
    """初始化 NCCL 后端并设置当前 GPU"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

# --- 2. 贡献分析引擎 (集成之前修正的 DCT 逻辑) ---
class DistTaskContributionAnalyzer:
    def __init__(self, config, device, local_rank):
        self.config = config
        self.device = device
        self.local_rank = local_rank
        self.B = config['num_bands']
        self.Q = config['num_buckets']
        self.num_layers = 32
        self.num_heads = 32
        
        # 预计算用于 MI 分桶的频率索引
        self.bin_indices = torch.linspace(0, self.Q, self.B + 1).long().to(device)

    # --- 修复版 DCT-II (针对 bfloat16 和任意长度优化) ---
    def bf16_dct_ii(self, x):
        N = x.shape[-1]
        
        # 1. 对称填充
        x_pad = torch.cat([x[..., ::2], x[..., 1::2].flip(dims=[-1])], dim=-1)
        
        # 2. 计算最近的 2 的幂次方 (针对 cuFFT 速度优化)
        M = 2 ** (int(math.ceil(math.log2(N)))) 
        pad_size = M - N
        if pad_size > 0:
            x_pad = torch.nn.functional.pad(x_pad, (0, pad_size))
        
        # 3. 运行实数 FFT (bf16 会自动转成 fp32 运算以保证精度)
        X_fft = torch.fft.rfft(x_pad.float(), dim=-1)
        
        # 4. 动态构造相位因子 phi
        freq_len = X_fft.shape[-1]
        k = torch.arange(freq_len, device=x.device, dtype=torch.float32)
        phi = torch.exp(-1j * math.pi * k / (2 * N))
        
        # 5. 执行频域旋转并取实部
        result = 2 * (X_fft * phi).real
        return result[..., :N].to(x.dtype) # 截断并转回 bf16

    # ... get_frequency_spectrum 和 MI 计算逻辑 (省略，保持之前鲁棒版) ...

# --- 3. 分布式数据集加载 (WikiText-2 用于校准) ---
def get_dist_dataloader(tokenizer, seq_len, batch_size):
    # 加载 WikiText-2 raw 训练集用于校准
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=seq_len)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 物理切分数据集到不同 GPU
    sampler = DistributedSampler(tokenized_datasets, shuffle=True)
    
    dataloader = DataLoader(
        tokenized_datasets, 
        batch_size=batch_size, 
        sampler=sampler, 
        collate_fn=default_data_collator
    )
    return dataloader, sampler

# --- 4. 分布式主程序 ---
def main():
    # A. 分布式设置
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # 只有主进程打印日志
    is_main_process = (local_rank == 0)

    # B. 配置加载
    model_path = "/path/to/your/llama-2-7b-hf" # A100 服务器绝对路径
    with open('configs/base_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # C. 模型加载 (集成 bfloat16，无量化)
    if is_main_process: print(f"正在以 bfloat16 加载模型...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # A100 原生支持 bf16
    model = LlamaForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map={"": device} # 显式映射到当前 GPU
    )
    
    # 将模型包装为 DDP
    model = DDP(model, device_ids=[local_rank])
    model.train() # 开启 Train 模式以获取梯度

    # D. 分布式数据加载 (A100 可增加 batch_size)
    # 将 seq_len 增加到 512，获取更丰富的频域特征
    dataloader, sampler = get_dist_dataloader(tokenizer, seq_len=512, batch_size=config.get('batch_size', 16))
    
    analyzer = DistTaskContributionAnalyzer(config, device, local_rank)
    
    # 用于在不同卡之间聚合数据
    all_mi_spectrums = []

    # E. 分布式反向传播与 MI 谱计算
    if is_main_process: print(f"开始在多张 A100 上进行分布式贡献分析...")
    
    # 为了速度，只分析 10 个 Batch 的文本
    num_calibration_batches = config.get('num_calibration_batches', 10)
    
    pbar = tqdm(total=num_calibration_batches, desc="分析进度", disable=not is_main_process)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_calibration_batches: break
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播 (DDP 会自动处理同步)
        outputs = model(**batch)
        loss = outputs.loss
        
        # 反向传播 (A100 高速获取梯度)
        loss.backward()
        
        # 提取当前卡计算出的局部 MI 谱
        # 此处需要根据 DDP 模型结构定位 o_proj 和 mlp 的权重的梯度
        # ... (此处省略 register_dist_hooks 逻辑，与之前 Hook 方案类似) ...
        # current_local_mi = analyzer.calculate_mi_on_this_gpu(...)
        
        #F. 梯度清理 (防止显存随时间累积)
        model.zero_grad(set_to_none=True)
        
        if is_main_process: pbar.update(1)

    # G. 分布式聚合 (All-Reduce)
    if is_main_process: print(f"正在从各 A100 聚合贡献谱...")
    
    # 将各卡上的 mi_spectrum 列表进行聚合 (使用 dist.all_reduce)
    # ... (聚合逻辑省略) ...
    # 最终在主进程获得全模型统一的 mi_spectrums
    
    # H. 结果保存
    if is_main_process:
        torch.save(final_aggregated_mi, 'results/lcb_scores.pt')
        print(f"聚合后的稳健贡献得分已保存。")

    pbar.close()
    cleanup()

if __name__ == "__main__":
    main()