import torch
import torch.nn as nn
import numpy as np
import math

def dct1d(tensor):
    tensor_f32 = tensor.to(torch.float32)
    N = tensor_f32.size(-1)
    x_pad = torch.cat([tensor_f32, tensor_f32.flip(dims=[-1])], dim=-1)
    fft_res = torch.fft.fft(x_pad, dim=-1)
    idx = torch.arange(N, device=tensor.device)
    weights = torch.exp(-1j * math.pi * idx / (2 * N))
    dct_out = torch.real(fft_res[..., :N] * weights)
    dct_out[..., 0] = dct_out[..., 0] * (1.0 / math.sqrt(2))
    return (dct_out * math.sqrt(2.0 / N)).to(tensor.dtype)

def calculate_mi_vectorized(Z_band, Y_discrete, num_classes=8):
    # 支持输入 Z_band 形状为 [N] 或 [N, C] 
    N = Z_band.size(0)
    if N <= 1:
        return torch.zeros(Z_band.shape[1:], device=Z_band.device) if Z_band.dim() > 1 else 0.0
    
    var_global = torch.var(Z_band, dim=0, unbiased=True) + 1e-8
    H_Z = 0.5 * torch.log(2 * math.pi * math.e * var_global)
    
    H_Z_given_Y = torch.zeros_like(H_Z)
    for c in range(1, num_classes + 1):
        mask = (Y_discrete == c)
        n_c = mask.sum().item()
        if n_c > 1:
            p_c = n_c / N
            var_c = torch.var(Z_band[mask], dim=0, unbiased=True) + 1e-8
            H_c = 0.5 * torch.log(2 * math.pi * math.e * var_c)
            H_Z_given_Y += p_c * H_c
            
    mi = H_Z - H_Z_given_Y
    return torch.clamp(mi, min=0.0)

def adaptive_band_merging(z_bar, Y_discrete, num_bands=4):
    # z_bar 形状: [N, Q_bins]
    Q = z_bar.shape[-1]
    bands = [[q] for q in range(Q)]
    
    while len(bands) > num_bands:
        min_loss = float('inf')
        merge_idx = 0
        for i in range(len(bands) - 1):
            idx_curr = bands[i]
            idx_next = bands[i+1]
            
            z_curr = z_bar[..., idx_curr].sum(dim=-1)
            z_next = z_bar[..., idx_next].sum(dim=-1)
            z_merged = z_bar[..., idx_curr + idx_next].sum(dim=-1)
            
            mi_curr = calculate_mi_vectorized(z_curr, Y_discrete).sum().item()
            mi_next = calculate_mi_vectorized(z_next, Y_discrete).sum().item()
            mi_merged = calculate_mi_vectorized(z_merged, Y_discrete).sum().item()
            
            loss = (mi_curr + mi_next) - mi_merged 
            if loss < min_loss:
                min_loss = loss
                merge_idx = i
                
        bands[merge_idx] = bands[merge_idx] + bands[merge_idx+1]
        bands.pop(merge_idx + 1)
    return bands