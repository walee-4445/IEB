# 99"
# 幅度谱编码器
# 直接处理频域幅度谱，提取频域特征
#
# 输入：(B, F, 1) 幅度谱
# 输出：(B, feature_dim) 特征向量
# """

import torch
import torch.nn as nn


class MagnitudeEncoder(nn.Module):
    """
    # 幅度谱编码器：直接处理频域幅度谱
    #
    # 架构：1D CNN + 全局池化 + 线性投影
    #
    # 输入：(B, F, 1) 幅度谱
    # 输出：(B, feature_dim) 特征向量
    #
    # 优点：
    # - 直接处理频域信息
    # - 保留频域物理意义
    # - 参数高效
    """
    
    def __init__(self, F_bins: int, channels=None, feature_dim=256):
        """
        Args:
            F_bins: 频域维度（频点数）
            channels: 各层通道数列表
            feature_dim: 输出特征维度
        """
        super().__init__()
        
        self.F_bins = F_bins
        self.feature_dim = feature_dim
        
        # 设置默认通道数
        if channels is None:
            channels = [32, 64, 128]
        
        # 构建1D CNN层
        layers = []
        in_ch = 1  # 输入1通道：仅幅度谱
        
        for out_ch in channels:
            layers.extend([
                # 卷积层
                nn.Conv1d(in_ch, out_ch, kernel_size=5, 
                         stride=2, padding=2),
                # 批归一化
                nn.BatchNorm1d(out_ch),
                # 激活函数
                nn.ReLU(inplace=True),
                nn.Dropout1d(0.1), #频域特征dropout
                # 最大池化
                nn.MaxPool1d(kernel_size=2)
            ])
            in_ch = out_ch
        
        # 全局平均池化
        layers.append(nn.AdaptiveAvgPool1d(1))
        
        self.conv_blocks = nn.Sequential(*layers)
        
        # 输出投影
        self.fc = nn.Linear(channels[-1], feature_dim)
    
    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Args:
            magnitude: (B, F, 1) 幅度谱
        
        Returns:
            features: (B, feature_dim) 特征向量
        """
        # (B, F, 1) -> (B, 1, F)
        x = magnitude.permute(0, 2, 1)
        
        # CNN提取频域特征
        x = self.conv_blocks(x)  # (B, C, 1)
        x = x.squeeze(-1)        # (B, C)
        
        # 输出投影
        x = self.fc(x)  # (B, feature_dim)
        
        return x

#
# if __name__ == '__main__':
#     # 测试幅度谱编码器
#     print("=" * 60)
#     print("Testing Magnitude Encoder")
#     print("=" * 60)
#
#     batch_size = 4
#     seq_len = 5120
#     F_bins = seq_len // 2 + 1  # 2561
#     feature_dim = 256
#
#     # 创建模拟幅度谱数据
#     magnitude = torch.randn(batch_size, F_bins, 1).abs()
#
#     # 创建编码器
#     encoder = MagnitudeEncoder(F_bins=F_bins, feature_dim=feature_dim)
#
#     print(f"\nEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
#
#     # 测试前向传播
#     print("\nTesting forward pass...")
#     features = encoder(magnitude)
#     print(f"  Input shape: {magnitude.shape}")
#     print(f"  Output shape: {features.shape}")
#     assert features.shape == (batch_size, feature_dim), "Shape mismatch!"
#     print("  ✓ Forward pass successful!")
#
#     # 测试反向传播
#     print("\nTesting backward pass...")
#     loss = features.mean()
#     loss.backward()
#     print("  ✓ Backward pass successful!")
#
#     # 测试不同序列长度
#     print("\nTesting different sequence lengths...")
#     for test_len in [2560, 5120, 10240]:
#         test_F = test_len // 2 + 1
#         test_mag = torch.randn(2, test_F, 1).abs()
#         test_encoder = MagnitudeEncoder(F_bins=test_F, feature_dim=feature_dim)
#         test_out = test_encoder(test_mag)
#         print(f"  seq_len={test_len}, F_bins={test_F}, output={test_out.shape}")
#         assert test_out.shape == (2, feature_dim), f"Shape mismatch for seq_len={test_len}!"
#     print("  ✓ All sequence lengths passed!")
#
#     print("\n" + "=" * 60)
#     print("All tests passed! ✓")
#     print("=" * 60)