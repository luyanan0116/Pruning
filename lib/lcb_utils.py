import torch
import numpy as np

# ==========================================================
# 第一步：互信息 (频域分解与任务贡献谱)
# ==========================================================
def step1_frequency_domain_mi(responses_list, device, num_bins=10, chunk_size=8):
    batch_size = len(responses_list)
    seq_len = responses_list[0].shape[1] 
    
    z_q_list = []
    
    for start_idx in range(0, batch_size, chunk_size):
        end_idx = min(start_idx + chunk_size, batch_size)
        chunk = torch.cat(responses_list[start_idx:end_idx], dim=0).to(device)
        
        # 1. 严格按照论文公式执行样本内标准化，剥离绝对幅值
        mu = chunk.mean(dim=1, keepdim=True)
        sigma = chunk.std(dim=1, keepdim=True) + 1e-8
        norm_chunk = (chunk - mu) / sigma
        
        # 2. 离散余弦变换 (使用实数 FFT 近似，连续内存加速)
        norm_chunk_fp32 = norm_chunk.transpose(1, 2).to(torch.float32).contiguous()
        fft_chunk = torch.fft.rfft(norm_chunk_fp32, dim=2)
        
        # 3. 统计能量谱 (全程保持 float32 防止溢出)
        energy_chunk = (torch.abs(fft_chunk) ** 2).transpose(1, 2)
        
        freq_len = energy_chunk.shape[1]
        bin_size = max(1, freq_len // num_bins)
        
        # 4. 细粒度频带划分
        chunk_z_q = []
        for q in range(num_bins):
            start = q * bin_size
            end = freq_len if q == num_bins - 1 else min((q + 1) * bin_size, freq_len)
            bin_energy = energy_chunk[:, start:end, :].sum(dim=1)
            chunk_z_q.append(bin_energy)
            
        chunk_z_q = torch.stack(chunk_z_q, dim=0)
        # 统一转为 float32 保存，彻底绝后患
        z_q_list.append(chunk_z_q.to(torch.float32))
        
        del chunk, mu, sigma, norm_chunk, norm_chunk_fp32, fft_chunk, energy_chunk, chunk_z_q

    # shape: (num_bins, batch_size, in_features)
    z_q = torch.cat(z_q_list, dim=1) 
    return z_q


# ==========================================================
# 第二步：粒球 (多粒度局部化与稳健估计)
# ==========================================================
def step2_granular_ball_estimation(z_q, args):
    num_bins, batch_size, in_features = z_q.shape
    R = getattr(args, 'n_samples_lcb', 10)
    
    mi_estimates = []
    for r in range(R):
        # 1. 模拟粒球：随机抽取局部样本子集
        subset_size = max(2, batch_size // 2)
        subset_indices = torch.randperm(batch_size, device=z_q.device)[:subset_size]
        local_z = z_q[:, subset_indices, :]
        
        # 2. 无监督互信息评估：使用“频段稳定性”作为结构化互信息的代理。
        # 方差越小，频段模式越稳定，说明该特征携带了可靠的底层语言规律（高互信息）。
        local_var = local_z.var(dim=1) 
        local_mi = 1.0 / (local_var + 1e-4)
        
        # 3. 跨频带融合贡献度
        total_contribution = local_mi.mean(dim=0) 
        mi_estimates.append(total_contribution)
        
    # shape: (R, in_features)
    return torch.stack(mi_estimates, dim=0)


# ==========================================================
# 第三步：LCB (置信下界评分综合输出)
# ==========================================================
def compute_lcb_weight_metric(weight, real_responses_list, scaler_row, args):
    num_bins = getattr(args, 'num_bins', 10) 
    chunk_size = getattr(args, 'chunk_size', 8)
    device = weight.device 
    
    # [Step 1]
    z_q = step1_frequency_domain_mi(real_responses_list, device, num_bins=num_bins, chunk_size=chunk_size)
    
    # [Step 2]
    S_estimates = step2_granular_ball_estimation(z_q, args)
    
    # [Step 3] 计算置信下界
    S_mean = S_estimates.mean(dim=0)
    S_std = S_estimates.std(dim=0)
    
    lam = getattr(args, 'lcb_lambda', 1.0)
    lcb_score_in = S_mean - lam * S_std  
    
    # 【核心平滑修正】：使用 Z-Score + Sigmoid 平滑归一化。
    # 彻底杜绝截断带来的“众生平等” Bug，保留特征在 0~1 之间的精确优劣排序！
    lcb_z = (lcb_score_in - lcb_score_in.mean()) / (lcb_score_in.std() + 1e-8)
    lcb_factor = torch.sigmoid(lcb_z)
    
    # 结合 Wanda 基座：权重幅值 × 激活量级 × 论文算出的频域稳健概率系数
    W_metric = torch.abs(weight) * torch.sqrt(scaler_row) * lcb_factor.to(weight.dtype).view(1, -1)
    
    return W_metric