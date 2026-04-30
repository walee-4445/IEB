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

        # 🆕 计算幅度谱的全局统计
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

            # 🆕 先Log变换，再计算统计
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

        # 🔧 5. 在幅度谱阶段重新归一化（修复范围爆炸问题）
        # 先用样本内统计归一化，再用全局统计
        # p_mag = (p_mag - p_mag.mean()) / (p_mag.std() + 1e-8)
        # v_mag = (v_mag - v_mag.mean()) / (v_mag.std() + 1e-8)

        # 然后使用全局统计进行二次归一化（可选，用于对齐不同样本）
        # p_mag = (p_mag - self.mag_mean[0]) / (self.mag_std[0] + 1e-8)
        # v_mag = (v_mag - self.mag_mean[1]) / (self.mag_std[1] + 1e-8)

        # 🔧 5. 使用全局统计归一化（更稳定）
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


#原始代码

# class MagnitudeSpectrumDataset(Dataset):
#     def __init__(self, X: np.ndarray, y: np.ndarray,
#                  mean: np.ndarray, std: np.ndarray,
#                  compute_mag_stats: bool = True):
#         """
#         Args:
#             X: (N, L, 2) 时域信号
#             y: (N,) 标签
#             mean: (2,) 时域归一化均值
#             std: (2,) 时域归一化标准差
#             compute_mag_stats: 是否计算幅度谱统计（首次创建时为True）
#         """
#         assert isinstance(X, np.ndarray) and X.ndim == 3 and X.shape[2] == 2
#         assert isinstance(y, np.ndarray) and y.ndim == 1 and len(y) == len(X)
#
#         self.X = X.astype(np.float32)
#         self.y = y.astype(np.int64)
#         self.mean = mean.reshape(1, 2).astype(np.float32)
#         self.std = std.reshape(1, 2).astype(np.float32)
#
#         # 🆕 计算幅度谱的全局统计
#         if compute_mag_stats:
#             print("正在计算幅度谱全局统计...")
#             self.mag_mean, self.mag_std = self._compute_magnitude_stats()
#             print(f"  压力幅度谱: mean={self.mag_mean[0]:.6f}, std={self.mag_std[0]:.6f}")
#             print(f"  振动幅度谱: mean={self.mag_mean[1]:.6f}, std={self.mag_std[1]:.6f}")
#         else:
#             self.mag_mean = np.zeros(2, dtype=np.float32)
#             self.mag_std = np.ones(2, dtype=np.float32)
#
#     def _compute_magnitude_stats(self):
#         """计算整个数据集的幅度谱统计"""
#         p_mags = []
#         v_mags = []
#
#         # 采样部分数据计算统计（避免内存溢出）
#         sample_size = min(1000, len(self.X))
#         indices = np.random.choice(len(self.X), sample_size, replace=False)
#
#         for idx in indices:
#             seg = (self.X[idx] - self.mean) / self.std
#             p_fft = np.fft.rfft(seg[:, 0])
#             v_fft = np.fft.rfft(seg[:, 1])
#             p_mags.append(np.abs(p_fft))
#             v_mags.append(np.abs(v_fft))
#
#         p_mags = np.array(p_mags)  # (sample_size, F)
#         v_mags = np.array(v_mags)  # (sample_size, F)
#
#         mag_mean = np.array([p_mags.mean(), v_mags.mean()], dtype=np.float32)
#         mag_std = np.array([p_mags.std(), v_mags.std()], dtype=np.float32)
#
#         return mag_mean, mag_std
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         # 1. 归一化时域信号
#         seg = (self.X[idx] - self.mean) / self.std  # (L, 2)
#
#         # 2. 分离模态
#         pressure_time = seg[:, 0]
#         vibration_time = seg[:, 1]
#
#         # 3. FFT转换
#         p_fft = np.fft.rfft(pressure_time)
#         v_fft = np.fft.rfft(vibration_time)
#
#         # 4. 提取幅度谱
#         p_mag = np.abs(p_fft)
#         v_mag = np.abs(v_fft)
#
#         # 🆕 5. 使用全局统计归一化幅度谱
#         p_mag = (p_mag - self.mag_mean[0]) / (self.mag_std[0] + 1e-8)
#         v_mag = (v_mag - self.mag_mean[1]) / (self.mag_std[1] + 1e-8)
#
#         # 6. 添加通道维度
#         pressure_mag = p_mag[:, np.newaxis]
#         vibration_mag = v_mag[:, np.newaxis]
#
#         return {
#             'pressure': torch.from_numpy(pressure_mag.astype(np.float32)),
#             'vibration': torch.from_numpy(vibration_mag.astype(np.float32)),
#             'label': torch.tensor(self.y[idx], dtype=torch.long)
#         }


if __name__ == '__main__':
    # 测试幅度谱数据集
    print("Testing MagnitudeSpectrumDataset...")
    
    # 创建模拟数据
    N, L = 100, 5120
    X = np.random.randn(N, L, 2).astype(np.float32)
    y = np.random.randint(0, 10, N).astype(np.int64)
    mean = np.array([0.0, 0.0], dtype=np.float32)
    std = np.array([1.0, 1.0], dtype=np.float32)
    
    # 创建数据集
    dataset = MagnitudeSpectrumDataset(X, y, mean, std)
    
    # 测试__getitem__
    sample = dataset[0]
    print(f"Pressure magnitude shape: {sample['pressure'].shape}")
    print(f"Vibration magnitude shape: {sample['vibration'].shape}")
    print(f"Label: {sample['label']}")
    
    # 验证频域维度
    F_bins = L // 2 + 1
    assert sample['pressure'].shape == (F_bins, 1), f"Expected ({F_bins}, 1), got {sample['pressure'].shape}"
    assert sample['vibration'].shape == (F_bins, 1), f"Expected ({F_bins}, 1), got {sample['vibration'].shape}"
    
    print("\n✓ All tests passed!")
    print(f"Frequency bins: {F_bins}")
    print(f"Frequency resolution: {10000 / L:.2f} Hz (assuming 10kHz sampling rate)")

    #
    # class MagnitudeSpectrumDataset(Dataset):
    #     """
    #     幅度谱数据集：在数据加载阶段进行FFT，仅保留幅度谱
    #
    #     输入：时域信号 (N, L, 2) [pressure, vibration]
    #     输出：幅度谱 (F, 1) 每个模态
    #
    #     优点：
    #     - 避免相位建模复杂性
    #     - 纯频域计算，无需IFFT
    #     - 幅度谱包含主要能量信息
    #     """
    #
    #     def __init__(self, X: np.ndarray, y: np.ndarray,
    #                  mean: np.ndarray, std: np.ndarray):
    #         """
    #         Args:
    #             X: (N, L, 2) 时域信号 [pressure, vibration]
    #             y: (N,) 标签
    #             mean: (2,) 归一化均值
    #             std: (2,) 归一化标准差
    #         """
    #         assert isinstance(X, np.ndarray) and X.ndim == 3 and X.shape[2] == 2
    #         assert isinstance(y, np.ndarray) and y.ndim == 1 and len(y) == len(X)
    #         assert isinstance(mean, np.ndarray) and mean.shape == (2,)
    #         assert isinstance(std, np.ndarray) and std.shape == (2,)
    #
    #         self.X = X.astype(np.float32)
    #         self.y = y.astype(np.int64)
    #         self.mean = mean.reshape(1, 2).astype(np.float32)
    #         self.std = std.reshape(1, 2).astype(np.float32)
    #
    #     def __len__(self):
    #         return len(self.X)
    #
    #     def __getitem__(self, idx):
    #         # 1. 归一化时域信号
    #         seg = (self.X[idx] - self.mean) / self.std  # (L, 2)
    #
    #         # 2. 分离模态
    #         pressure_time = seg[:, 0]  # (L,)
    #         vibration_time = seg[:, 1]  # (L,)
    #
    #         # 3. FFT转换（仅保留幅度谱）
    #         p_fft = np.fft.rfft(pressure_time)  # (F,) 复数
    #         v_fft = np.fft.rfft(vibration_time)  # (F,) 复数
    #
    #         # 4. 提取幅度谱
    #         p_mag = np.abs(p_fft)  # (F,)
    #         v_mag = np.abs(v_fft)  # (F,)
    #
    #         # 5. Log-scale归一化（可选，提升训练稳定性）
    #         p_mag = np.log(p_mag + 1e-8)
    #         v_mag = np.log(v_mag + 1e-8)
    #         # 🆕 5. 对幅度谱进行归一化（关键修复）
    #         # 方法A：Z-score归一化（推荐）
    #         p_mag = (p_mag - p_mag.mean()) / (p_mag.std() + 1e-8)
    #         v_mag = (v_mag - v_mag.mean()) / (v_mag.std() + 1e-8)
    #         # 6. 添加通道维度
    #         pressure_mag = p_mag[:, np.newaxis]  # (F, 1)
    #         vibration_mag = v_mag[:, np.newaxis]  # (F, 1)
    #
    #         return {
    #             'pressure': torch.from_numpy(pressure_mag.astype(np.float32)),  # (F, 1)
    #             'vibration': torch.from_numpy(vibration_mag.astype(np.float32)),  # (F, 1)
    #             'label': torch.tensor(self.y[idx], dtype=torch.long)
    #         }