import torch
import math

class TaskContributionPruner:
    def __init__(self, config, device='cuda'):
        self.device = device
        self.B = config.get('num_bands', 5)       # 频带数 B [cite: 240]
        self.Q = config.get('num_buckets', 20)    # 细粒度桶 Q [cite: 223]
        self.eps = 1e-10

    def torch_dct_ii(self, x):
        """
        公式 (5): 修复版 DCT-II
        解决实数 FFT (rfft) 长度对齐与 FP16 幂次方限制
        """
        N = x.shape[-1]
        
        # 1. 镜像填充 (Mirroring)
        x_pad = torch.cat([x[..., ::2], x[..., 1::2].flip(dims=[-1])], dim=-1)
        
        # 2. 补齐到 2 的幂次方 (针对 6G 显存 FP16 环境)
        M = 2 ** (int(math.ceil(math.log2(N)))) 
        pad_size = M - N
        if pad_size > 0:
            x_pad = torch.nn.functional.pad(x_pad, (0, pad_size))
        
        # 3. 运行实数 FFT
        # 对于长度 M 的输入，rfft 输出长度为 (M // 2) + 1
        X_fft = torch.fft.rfft(x_pad, dim=-1)
        
        # 4. 构造相位因子 phi
        # 注意：phi 必须与 X_fft 的输出长度 (M // 2 + 1) 对齐
        freq_len = X_fft.shape[-1]
        k = torch.arange(freq_len, device=x.device, dtype=x.dtype)
        phi = torch.exp(-1j * math.pi * k / (2 * N))
        
        # 5. 执行频域旋转并取实部
        # 只取前 N 个有效分量以对齐后续的分桶逻辑
        result = 2 * (X_fft * phi).real
        return result[..., :N]

    def get_frequency_spectrum(self, grads, mode='attn'):
        g = torch.norm(grads, p=2, dim=-1) if mode == 'attn' else torch.abs(grads).mean(dim=-1)
        g_tilde = (g - g.mean()) / (g.std() + self.eps)
        e = torch.abs(self.torch_dct_ii(g_tilde))**2
        e_reshaped = e.view(1, 1, -1)
        z = torch.nn.functional.adaptive_avg_pool1d(e_reshaped, self.Q)
        return z.view(-1) 
    
    def estimate_mi_knn(self, Z, Y, k=3):
        """公式 (28-29): 基于近邻搜索的互信息估计"""
        n = Z.shape[0]
        if n <= k: return torch.tensor(0.0, device=self.device)

        def get_h(data):
            d = 1 if data.ndim == 1 else data.shape[1]
            dist_mat = torch.cdist(data.view(-1, d), data.view(-1, d), p=2)
            eps_k = torch.topk(dist_mat, k + 1, largest=False, dim=-1).values[:, -1]
            v_d = math.pi**(d/2) / torch.exp(torch.lgamma(torch.tensor(d/2 + 1)))
            return torch.digamma(torch.tensor(float(n))) - torch.digamma(torch.tensor(float(k))) + \
                   torch.log(v_d) + (d/n) * torch.sum(torch.log(eps_k + self.eps))

        h_total = get_h(Z)
        h_cond = torch.tensor(0.0, device=self.device)
        for c in torch.unique(Y):
            mask = (Y == c)
            if mask.sum() > k:
                h_cond += (mask.sum().float() / n) * get_h(Z[mask])
        return torch.clamp(h_total - h_cond, min=0.0)

    def compute_lcb(self, scores, lam=0.5):
        """
        实现公式 (27): 计算风险厌恶的置信下界分数 (LCB) [cite: 138, 154]
        用于在有限校准集下抑制抽样扰动，量化排序可信度 [cite: 39, 145]
        """
        # scores: 重复抽样下获得的贡献分数经验分布 [cite: 138]
        # lam: 风险控制系数 lambda [cite: 291, 292]
        return scores.mean() - lam * scores.std()
    def generate_pruning_config(self, units, total_budget, gamma=0.5, alpha=1.0):
        """
        基于稳健贡献谱的多维资源预算剪枝配置生成 (集成层级保护逻辑)
        对应研究内容 (3): 公式 33-39
        """
        import torch
        
        selected_K = []
        current_cost = 0
        is_selected = [False] * len(units)
        
        # --- 1. 预处理：设备对齐与频带聚合 ---
        for u in units:
            # 确保数据在 GPU 上以进行快速矩阵运算
            u['mi_spectrum'] = u['mi_spectrum'].to(self.device)
            # 将 Q=20 维原始谱聚合为 B=5 维频带贡献 (公式 33)
            if u['mi_spectrum'].shape[0] != self.B:
                u['mi_spectrum'] = torch.nn.functional.adaptive_avg_pool1d(
                    u['mi_spectrum'].view(1, 1, -1), self.B
                ).view(-1)

        # 计算全模型总贡献基准 C_U (公式 34)
        C_U = torch.stack([u['mi_spectrum'] for u in units]).sum(dim=0)

        # --- 2. 核心干预：基础层级保护 (防止 PPL 崩溃) ---
        # 强制保留前 3 层 (Layer 0, 1, 2) 和 最后一层 (Layer 31)
        # 理由：底层负责特征提取，顶层负责输出对齐，是模型的“生命线”
        protected_layer_indices = [0, 1, 2, 31]
        
        for i, u in enumerate(units):
            # 从 ID 中解析层号 (例如 "layer_5_mlp" -> 5)
            layer_idx = -1
            parts = u['id'].replace('.', '_').split('_')
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    break
            
            if layer_idx in protected_layer_indices:
                is_selected[i] = True
                selected_K.append(u)
                current_cost += u['cost']

        # --- 3. 贪心搜索阶段 (基于 LCB 与 频带补偿) ---
        while True:
            # 计算当前已选集合在各频带上的覆盖度 C_K (公式 33)
            C_K = torch.zeros(self.B, device=self.device)
            for idx, selected in enumerate(is_selected):
                if selected:
                    C_K += units[idx]['mi_spectrum']
            
            # 计算欠覆盖权重 w (公式 38): 显式保护长程依赖和局部模式的平衡
            w = torch.clamp(gamma * C_U - C_K, min=0.0)
            
            best_delta = -1.0
            best_idx = -1
            
            for i, u in enumerate(units):
                # 跳过已选单元或超出剩余预算的单元
                if is_selected[i] or (current_cost + u['cost'] > total_budget):
                    continue
                
                # 计算候选单元的边际增益 Delta(u|K) (公式 39)
                # 包含：1. 稳健分数贡献 (LCB)  2. 频带覆盖补偿 (w * spectrum)
                term_lcb = u['lcb'] / u['cost']
                term_cover = alpha * torch.sum(w * u['mi_spectrum']) / u['cost']
                delta = term_lcb + term_cover
                
                if delta > best_delta:
                    best_delta = delta
                    best_idx = i
            
            # 如果没有更多单元可加入或预算耗尽，退出循环
            if best_idx == -1:
                break
            
            # 正式加入选择集合
            is_selected[best_idx] = True
            selected_K.append(units[best_idx])
            current_cost += units[best_idx]['cost']

        print(f"配置生成完毕：硬性保护层单元已锁定，总计保留 {len(selected_K)} 个单元。")
        return selected_K