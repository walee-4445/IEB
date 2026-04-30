"""
幅度谱数据集
纯频域架构：输入/输出都是幅度谱
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class MagnitudeSpectrumDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 mean: np.ndarray, std: np.ndarray,
                 compute_mag_stats: bool = True):
        """
        Args:
            compute_mag_stats: 是否计算幅度谱统计（训练集为True，验证/测试集为False）
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.mean = mean.reshape(1, 2).astype(np.float32)
        self.std = std.reshape(1, 2).astype(np.float32)

        #  计算幅度谱的全局统计
        if compute_mag_stats:
            print("正在计算幅度谱全局统计...")
            self.mag_mean, self.mag_std = self._compute_magnitude_stats()
            print(f"  压力幅度谱: mean={self.mag_mean[0]:.6f}, std={self.mag_std[0]:.6f}")
            print(f"  振动幅度谱: mean={self.mag_mean[1]:.6f}, std={self.mag_std[1]:.6f}")
        else:
            # 验证/测试集需要传入训练集的统计量
            self.mag_mean = np.zeros(2, dtype=np.float32)
            self.mag_std = np.ones(2, dtype=np.float32)

    def _compute_magnitude_stats(self):
        """计算整个数据集的幅度谱统计"""
        p_mags = []
        v_mags = []

        # 采样部分数据计算统计（避免内存溢出）
        sample_size = min(1000, len(self.X))
        indices = np.random.choice(len(self.X), sample_size, replace=False)

        for idx in indices:
            seg = (self.X[idx] - self.mean) / self.std
            p_fft = np.fft.rfft(seg[:, 0])
            v_fft = np.fft.rfft(seg[:, 1])

            # 先Log变换，再计算统计
            p_mag = np.log(np.abs(p_fft) + 1e-8)
            v_mag = np.log(np.abs(v_fft) + 1e-8)

            p_mags.append(p_mag)
            v_mags.append(v_mag)

        p_mags = np.array(p_mags)  # (sample_size, F)
        v_mags = np.array(v_mags)  # (sample_size, F)

        mag_mean = np.array([p_mags.mean(), v_mags.mean()], dtype=np.float32)
        mag_std = np.array([p_mags.std(), v_mags.std()], dtype=np.float32)

        return mag_mean, mag_std

    def __getitem__(self, idx):
        # 1. 归一化时域信号
        seg = (self.X[idx] - self.mean) / self.std

        # 2. FFT转换
        p_fft = np.fft.rfft(seg[:, 0])
        v_fft = np.fft.rfft(seg[:, 1])

        # 3. 提取幅度谱
        p_mag = np.abs(p_fft)
        v_mag = np.abs(v_fft)

        # 4. Log变换
        p_mag = np.log(p_mag + 1e-8)
        v_mag = np.log(v_mag + 1e-8)

        # 5. 使用全局统计归一化（更稳定）
        p_mag = (p_mag - self.mag_mean[0]) / (self.mag_std[0] + 1e-8)
        v_mag = (v_mag - self.mag_mean[1]) / (self.mag_std[1] + 1e-8)


        # 6. 添加通道维度
        pressure_mag = p_mag[:, np.newaxis]
        vibration_mag = v_mag[:, np.newaxis]

        return {
            'pressure': torch.from_numpy(pressure_mag.astype(np.float32)),
            'vibration': torch.from_numpy(vibration_mag.astype(np.float32)),
            'label': torch.tensor(self.y[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.X)
    
    def set_mag_stats(self, mag_mean, mag_std):
        """设置幅度谱统计（用于验证/测试集）"""
        self.mag_mean = mag_mean
        self.mag_std = mag_std
        print(f"✓ 已设置幅度谱统计: P_mean={mag_mean[0]:.6f}, V_mean={mag_mean[1]:.6f}")

