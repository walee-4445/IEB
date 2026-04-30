

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class FourierFeatures(nn.Module):
    """傅里叶特征编码"""
    
    def __init__(self, num_bands: int = 16, max_freq: float = 10.0):
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        freqs = torch.logspace(0, math.log10(max_freq), steps=num_bands)
        self.register_buffer("freqs", freqs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [F] -> [F, 2*num_bands]"""
        xb = x.unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * math.pi #(F, num_bands=16)
        return torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)


class ConditionalINR(nn.Module):
    """条件隐式神经表示（FiLM调制）"""
    
    def __init__(self, context_dim: int, out_dim: int, num_bands: int = 16,
                 hidden: int = 128, depth: int = 3, max_freq: float = 10.0):
        super().__init__()
        self.ff = FourierFeatures(num_bands=num_bands, max_freq=max_freq)
        ff_dim = 2 * num_bands
        
        # FiLM调制参数
        self.gamma = nn.Linear(context_dim, ff_dim)
        self.beta = nn.Linear(context_dim, ff_dim)
        
        # MLP网络
        layers = []
        in_dim = ff_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, out_dim)]
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, freq_coord: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        freq_coord: [F] in [0,1]
        context: [B, C]
        return: [B, F, out_dim]
        """
        B = context.size(0)
        Freq = freq_coord.size(0)
        
        base = self.ff(freq_coord)  # [F, ff_dim]
        base = base.unsqueeze(0).expand(B, Freq, -1)  # [B, F, ff_dim]
        
        gamma = self.gamma(context).unsqueeze(1)  # [B, 1, ff_dim]
        beta = self.beta(context).unsqueeze(1)    # [B, 1, ff_dim]


        x = base * (1.0 + gamma) + beta   # [B, F, ff_dim]
        
        y = self.mlp(x)  # [B, F, out_dim]
        return y


class PVNFMMagnitudeFixedV2(nn.Module):
    
    def __init__(self, F_bins: int, context_dim: int = 128,
                 num_bands: int = 16, hidden_inr: int = 128,
                 use_progressive_range: bool = False,
                 output_clip_range: float = 9.0):
        super().__init__()
        self.F_bins = F_bins
        self.context_dim = context_dim
        self.use_progressive_range = use_progressive_range
        self.output_clip_range = output_clip_range
        
        # 共享编码器：幅度谱 -> 上下文向量
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, context_dim),
            nn.LayerNorm(context_dim),
        )
        
        # P2V方向的独立INR（压力→振动）
        self.inr_T_p2v = ConditionalINR(
            context_dim=context_dim,
            out_dim=1,  # α参数
            num_bands=num_bands,
            hidden=hidden_inr,
            depth=3,
            max_freq=10.0,
        )
        self.inr_R_p2v = ConditionalINR(
            context_dim=context_dim,
            out_dim=1,  # β参数
            num_bands=num_bands,
            hidden=hidden_inr,
            depth=3,
            max_freq=10.0,
        )
        
        # V2P方向的独立INR（振动→压力）
        self.inr_T_v2p = ConditionalINR(
            context_dim=context_dim,
            out_dim=1,  # α参数
            num_bands=num_bands,
            hidden=hidden_inr,
            depth=3,
            max_freq=10.0,
        )
        self.inr_R_v2p = ConditionalINR(
            context_dim=context_dim,
            out_dim=1,  # β参数
            num_bands=num_bands,
            hidden=hidden_inr,
            depth=3,
            max_freq=10.0,
        )
        
        # 固定的归一化频率坐标 [0,1]
        freq = torch.linspace(0.0, 1.0, steps=F_bins)
        self.register_buffer("freq_coord", freq)
    
    def forward(self, source_mag: torch.Tensor, direction: str = 'p2v',
                training_progress: float = 1.0) -> torch.Tensor:

        B, F_dim, _ = source_mag.shape
        

        x_in = source_mag.permute(0, 2, 1)  # (B, 1, F)
        z0 = self.encoder(x_in)  # (B, context_dim)
        

        if direction == 'p2v':
            # 使用P2V专用INR
            T_hat = self.inr_T_p2v(self.freq_coord, z0)  # (B, F, 1)
            R_hat = self.inr_R_p2v(self.freq_coord, z0)  # (B, F, 1)
        else:  # v2p
            # 使用V2P专用INR
            T_hat = self.inr_T_v2p(self.freq_coord, z0)  # (B, F, 1)
            R_hat = self.inr_R_v2p(self.freq_coord, z0)  # (B, F, 1)

        if direction == 'v2p':
            # V→P
            if self.use_progressive_range:

                alpha_range = 0.4 + training_progress * 0.4  # [0.4, 0.8]
                beta_range = 0.3 + training_progress * 0.2   # [0.3, 0.5]
                alpha = torch.sigmoid(T_hat[..., 0]) * alpha_range + (1.0 - alpha_range/2)
                beta = torch.tanh(R_hat[..., 0]) * beta_range
            else:

                alpha = torch.sigmoid(T_hat[..., 0]) * 0.8 + 0.6  # [0.6, 1.4]
                beta = torch.tanh(R_hat[..., 0]) * 0.5  # [-0.5, 0.5]
        else:  # p2v
            #  P→V
            if self.use_progressive_range:
                alpha_range = 0.8 + training_progress * 0.7  # [0.8, 1.5]
                beta_range = 0.8 + training_progress * 0.7   # [0.8, 1.5]
                alpha = torch.sigmoid(T_hat[..., 0]) * alpha_range + (1.0 - alpha_range/2)
                beta = torch.tanh(R_hat[..., 0]) * beta_range
            else:

                alpha = torch.sigmoid(T_hat[..., 0]) * 1.5 + 0.5  # [0.5, 2.0]
                beta = torch.tanh(R_hat[..., 0]) * 1.5  # [-1.5, 1.5]
        
        source_mag_2d = source_mag.squeeze(-1)  # (B, F)


        target_mag_2d = alpha * source_mag_2d + beta


        target_mag = target_mag_2d.unsqueeze(-1)  # (B, F, 1)
        
        return target_mag


