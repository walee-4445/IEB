
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

        if channels is None:
            channels = [32, 64, 128]
        
        # 构建1D
        layers = []
        in_ch = 1  # 输入1通道
        
        for out_ch in channels:
            layers.extend([

                nn.Conv1d(in_ch, out_ch, kernel_size=5, 

                nn.BatchNorm1d(out_ch),

                nn.ReLU(inplace=True),
                nn.Dropout1d(0.1),

                nn.MaxPool1d(kernel_size=2)
            ])
            in_ch = out_ch
        

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
