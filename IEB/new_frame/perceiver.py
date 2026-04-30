"""
Perceiver跨模态融合模块
基于CMAD的Perceiver实现，适配双模态时序数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# 架构：
#     输入特征 [(B,D), (B,D)]  # 压力特征 + 振动特征
#       ↓ Stack
#     Context (B, 2, D)
#       ↓ 初始化Latents
#     Latents (B, M, D)  # M=32个可学习向量
#       ↓ PerceiverBlock × 4
#         ├─ CrossAttention: Latents ← Context
#         ├─ SelfAttention: Latents ← Latents
#         └─ FeedForward
#       ↓ GlobalAvgPool
#     Output (B, D)

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):
    """交叉注意力机制"""

    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or dim

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        """
        x: queries (B, N, dim)
        context: keys and values (B, M, context_dim)
        """
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PerceiverBlock(nn.Module):
    """Perceiver基本块"""

    def __init__(self, dim, context_dim, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
        super().__init__()

        # 交叉注意力（latent attend to input）
        self.cross_attn = CrossAttention(
            dim=dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        self.cross_attn_norm = nn.LayerNorm(dim)

        # 自注意力（latent attend to latent）
        self.self_attn = MultiHeadAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        self.self_attn_norm = nn.LayerNorm(dim)

        # 前馈网络
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, latent, context):
        """
        latent: (B, num_latents, dim)
        context: (B, num_context, context_dim)
        """
        # 交叉注意力
        latent = self.cross_attn(self.cross_attn_norm(latent), context) + latent

        # 自注意力
        latent = self.self_attn(self.self_attn_norm(latent)) + latent

        # 前馈网络
        latent = self.ff(self.ff_norm(latent)) + latent

        return latent


class PerceiverFusion(nn.Module):
    """
    Perceiver融合模块
    用于融合压力和振动特征
    """

    def __init__(self,
                 input_dim=128,
                 latent_dim=256,
                 num_latents=32,
                 depth=4,
                 heads=8,## 注意力头数
                 dim_head=32, # 每个头的维度
                 mlp_dim=512, # FFN隐藏层维度
                 dropout=0.1):
        super(PerceiverFusion, self).__init__()

        self.num_latents = num_latents
        self.latent_dim = latent_dim
        # 可学习的潜在查询向量（核心组件）
        # 形状：(1, num_latents, latent_dim)
        # 作用：作为"问题"向输入"提问"

        # 可学习的潜在查询向量
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, latent_dim))

        # Perceiver块
        self.blocks = nn.ModuleList([
            PerceiverBlock(
                dim=latent_dim,
                context_dim=input_dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout
            ) for _ in range(depth)
        ])

        # 输出层归一化
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, features):
        """
        features: List of feature tensors
            - 单模态: [(B, input_dim)]
            - 双模态: [(B, input_dim), (B, input_dim)]

        return: (B, latent_dim) 融合后的特征



          Args:
            features: List of feature tensors
                - 单模态: [(B, input_dim)]
                - 双模态: [(B, input_dim), (B, input_dim)]

        Returns:
            output: (B, latent_dim) 融合后的特征

        处理流程：
            1. 拼接输入特征为context
            2. 复制latent queries到batch
            3. 通过多层Perceiver块处理
            4. 全局平均池化得到最终特征
        """
        B = features[0].size(0)

        # 拼接所有模态特征
        if len(features) == 1:
            # 单模态
            context = features[0].unsqueeze(1)  # (B, 1, input_dim)
        else:
            # 多模态：拼接
            context = torch.stack(features, dim=1)  # (B, M, input_dim)

        # 初始化潜在向量
        latent = repeat(self.latent_queries, '1 n d -> b n d', b=B)

        # 通过Perceiver块
        for block in self.blocks:
            latent = block(latent, context)

        # 归一化
        latent = self.norm(latent)

        # 全局平均池化
        output = latent.mean(dim=1)  # (B, latent_dim)

        return output


if __name__ == '__main__':
    # 测试代码
    print("Testing PerceiverFusion...")

    # 双模态测试
    perceiver = PerceiverFusion(
        input_dim=128,
        latent_dim=256,
        num_latents=32,
        depth=4
    )

    # 模拟压力和振动特征
    pressure_feat = torch.randn(4, 128)
    vibration_feat = torch.randn(4, 128)

    # 双模态融合
    output = perceiver([pressure_feat, vibration_feat])
    print(f"Dual-modal output shape: {output.shape}")
    assert output.shape == (4, 256), "Output shape mismatch!"

    # 单模态测试
    output_single = perceiver([pressure_feat])
    print(f"Single-modal output shape: {output_single.shape}")
    assert output_single.shape == (4, 256), "Single-modal output shape mismatch!"

    print("\nAll tests passed! ✓")