import torch
from lib.lcb_core import calculate_mi_vectorized

class GranularBall:
    def __init__(self, samples_idx, macro_space, Y_discrete):
        self.samples_idx = samples_idx 
        self.center = torch.mean(macro_space[samples_idx], dim=0) 
        distances = torch.norm(macro_space[samples_idx] - self.center, p=2, dim=1)
        self.radius = torch.max(distances).item() if len(distances) > 0 else 0.0 
        
        y_sub = Y_discrete[samples_idx]
        if len(y_sub) == 0:
            self.purity = 0.0
        else:
            counts = torch.bincount(y_sub)
            self.purity = torch.max(counts).item() / len(y_sub) 

def build_granular_balls(macro_space, Y_discrete, n_min=4, eta=0.85, max_depth=3):
    N = macro_space.size(0)
    initial_idx = torch.arange(N, device=macro_space.device)
    root_ball = GranularBall(initial_idx, macro_space, Y_discrete)
    balls_by_level = {0: [root_ball]}
    
    for depth in range(max_depth):
        current_balls = balls_by_level[depth]
        next_balls = []
        for ball in current_balls:
            if len(ball.samples_idx) >= n_min and ball.purity < eta:
                sub_indices = ball.samples_idx
                if len(sub_indices) > 2:
                    perm = torch.randperm(len(sub_indices))
                    c1, c2 = macro_space[sub_indices[perm[0]]], macro_space[sub_indices[perm[1]]]
                    dist1 = torch.norm(macro_space[sub_indices] - c1, dim=1)
                    dist2 = torch.norm(macro_space[sub_indices] - c2, dim=1)
                    mask1 = dist1 <= dist2
                    idx1, idx2 = sub_indices[mask1], sub_indices[~mask1]
                    if len(idx1) > 0 and len(idx2) > 0:
                        next_balls.append(GranularBall(idx1, macro_space, Y_discrete))
                        next_balls.append(GranularBall(idx2, macro_space, Y_discrete))
                        continue
            next_balls.append(ball)
        balls_by_level[depth + 1] = next_balls
    return balls_by_level

def estimate_lcb_scores(v_space, Y_discrete, macro_space, num_bands=4, num_channels=11008, lambda_rf=0.5):
    # 使用降维后的 macro_space 聚类粒球，避免在极高维特征通道中失效
    levels = build_granular_balls(macro_space, Y_discrete)
    M = len(levels)
    N = v_space.size(0)
    R_runs = 3
    S_runs = []
    
    for r in range(R_runs):
        S_u = torch.zeros(num_channels, device=v_space.device)
        for m, balls in levels.items():
            weight_m = 1.0 / M 
            for ball in balls:
                if len(ball.samples_idx) <= 1:
                    continue
                p_g = len(ball.samples_idx) / N 
                for b in range(num_bands):
                    # 抽取隶属于粒球的样本，在具体通道频带上计算互信息
                    Z_subset = v_space[ball.samples_idx, :, b] 
                    Y_subset = Y_discrete[ball.samples_idx]
                    mi_local = calculate_mi_vectorized(Z_subset, Y_subset)
                    S_u += weight_m * p_g * mi_local  
        S_runs.append(S_u.unsqueeze(0))
        
    S_tensor = torch.cat(S_runs, dim=0) 
    S_mean = torch.mean(S_tensor, dim=0) 
    S_std = torch.std(S_tensor, dim=0) + 1e-8 
    
    LCB = S_mean - lambda_rf * S_std 
    # 【极度关键】：阻断方差惩罚造成的负数开根号崩溃
    return torch.clamp(LCB, min=1e-8)