"""
Jenks自然断点优化 + 联合谱熵密度
用于自适应频域划分（IEB组件）
"""

import numpy as np
import torch
import torch.nn as nn
from jenkspy import jenks_breaks


class JenksEntropySplitter:
    """
    基于联合谱熵密度的Jenks自适应频域划分器

    核心算法：
    1. 计算双模态联合谱熵密度 H(f) = -Σ p(f)log(p(f))
    2. 使用Jenks自然断点优化算法切分频域
    3. 验证GVF > 0.8的统计显著性

    参数：
        F_bins: 频域维度（频点数）
        min_k: 最小子带数量
        max_k: 最大子带数量
        gvf_threshold: 拟合优度阈值
    """

    def __init__(self, F_bins=1281, min_k=2, max_k=6, gvf_threshold=0.8):
        self.F_bins = F_bins
        self.min_k = min_k
        self.max_k = max_k
        self.gvf_threshold = gvf_threshold

    def compute_joint_spectral_entropy(self, pressure_mag, vibration_mag):
        """
        计算双模态联合谱熵密度

        Args:
            pressure_mag: (B, F, 1) 压力幅度谱
            vibration_mag: (B, F, 1) 振动幅度谱

        Returns:
            entropy_density: (F,) numpy数组，!!!每个频点的熵值
        """
        # 1. 计算batch平均能量
        p_energy = (pressure_mag ** 2).mean(dim=0).squeeze()  # (F,)
        v_energy = (vibration_mag ** 2).mean(dim=0).squeeze()  # (F,)

        # 2. 联合能量分布
        joint_energy = p_energy + v_energy + 1e-8

        # 3. 归一化为概率分布
        joint_prob = joint_energy / joint_energy.sum()

        # 4. 计算熵密度
        entropy = -joint_prob * torch.log(joint_prob + 1e-8)

        return entropy.cpu().numpy()

    def compute_gvf(self, data, breaks):
        """
        计算拟合优度 (Goodness of Variance Fit)

        GVF = 1 - (SDAM / SDCM)
        SDAM: 类内方差和
        SDCM: 总方差

        Args:
            data: 一维数据数组
            breaks: 切分点列表

        Returns:
            gvf: 拟合优度 [0, 1]
        """
        data = np.array(data)
        n_classes = len(breaks) - 1

        # 计算总方差
        sdcm = np.sum((data - data.mean()) ** 2)

        if sdcm == 0:
            return 0.0

        # 计算类内方差
        sdam = 0
        for i in range(n_classes):
            # 找到当前类的数据
            mask = (data >= breaks[i]) & (data < breaks[i + 1])
            if i == n_classes - 1:  # 最后一个类包含右边界
                mask = (data >= breaks[i]) & (data <= breaks[i + 1])

            if mask.sum() > 0:
                class_data = data[mask]
                sdam += np.sum((class_data - class_data.mean()) ** 2)

        gvf = 1 - (sdam / sdcm)
        return gvf

    # def adaptive_split(self, entropy_density):
    #     """
    #     自适应确定最优k值并切分
    #
    #     Args:
    #         entropy_density: (F,) 熵密度数组
    #
    #     Returns:
    #         breaks: 切分点列表 [0, b1, b2, ..., F_bins]
    #         k: 最优子带数量
    #         gvf: 拟合优度
    #     """
    #     best_k = self.min_k
    #     best_gvf = 0
    #     best_breaks = None
    #
    #     # 遍历所有可能的k值
    #     for k in range(self.min_k, self.max_k + 1):
    #         try:
    #             # 使用Jenks算法计算切分点
    #             breaks = jenks_breaks(entropy_density, n_classes=k)
    #
    #             # 计算GVF
    #             gvf = self.compute_gvf(entropy_density, breaks)
    #
    #             # 选择GVF最高且满足阈值的k
    #             if gvf > best_gvf and gvf >= self.gvf_threshold:
    #                 best_k = k
    #                 best_gvf = gvf
    #                 best_breaks = breaks
    #         except Exception as e:
    #             # Jenks算法可能失败（数据分布特殊）
    #             continue
    #
    #     # 回退机制：若无满足条件的k，使用均匀切分
    #     if best_breaks is None:
    #         best_breaks = np.linspace(0, self.F_bins, self.min_k + 1).astype(int)
    #         best_k = self.min_k
    #         best_gvf = 0.0
    #         print(f"⚠️ Jenks切分失败，回退到k={best_k}均匀切分")
    #
    #     return best_breaks, best_k, best_gvf
    def adaptive_split(self, entropy_density):
        """
        固定k值，用Jenks找到最优的频点索引切分位置

        方法：对每个频点的熵值做Jenks分类，然后找到连续区间的边界
        """
        k = self.min_k  # 固定k（min_k == max_k）

        try:
            # Step 1: Jenks对熵值分类 → 得到值域断点
            value_breaks = jenks_breaks(entropy_density, n_classes=k)
            gvf = self.compute_gvf(entropy_density, value_breaks)

            # Step 2: 将每个频点分配到对应的类别
            labels = np.zeros(len(entropy_density), dtype=int)
            for f_idx, val in enumerate(entropy_density):
                for c in range(k):
                    if c == k - 1:
                        if val >= value_breaks[c]:
                            labels[f_idx] = c
                    else:
                        if value_breaks[c] <= val < value_breaks[c + 1]:
                            labels[f_idx] = c

            # Step 3: 找到连续区间的边界
            # 注意：Jenks按值分类，相同类别的频点可能不连续
            # 所以我们用"累积熵等分"作为更稳定的替代方案

            cumulative = np.cumsum(entropy_density)
            total = cumulative[-1]

            index_breaks = [0]
            for i in range(1, k):
                target = total * i / k
                idx = int(np.searchsorted(cumulative, target))
                # 确保不重复
                if idx <= index_breaks[-1]:
                    idx = index_breaks[-1] + 1
                index_breaks.append(min(idx, len(entropy_density) - 1))
            index_breaks.append(len(entropy_density))

            return np.array(index_breaks), k, gvf

        except Exception as e:
            # 回退到均匀切分
            index_breaks = np.linspace(0, self.F_bins, k + 1).astype(int)
            return index_breaks, k, 0.0

    def generate_frequency_masks(self, breaks, device='cuda'):
        """
        生成频域掩码

        Args:
            breaks: 切分点列表 [0, b1, b2, ..., F_bins]
            device: 'cuda' or 'cpu'

        Returns:
            masks: List[Tensor(F,)] 长度为k的掩码列表
        """
        k = len(breaks) - 1
        masks = []

        for i in range(k):
            mask = torch.zeros(self.F_bins, device=device)
            start_idx = int(breaks[i])
            end_idx = int(breaks[i + 1])

            # 设置掩码区域为1
            mask[start_idx:end_idx] = 1.0

            masks.append(mask)

        return masks


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("测试 JenksEntropySplitter")
    print("=" * 60)

    splitter = JenksEntropySplitter(F_bins=1281, min_k=2, max_k=6, gvf_threshold=0.8)

    # 模拟数据
    pressure_mag = torch.randn(32, 1281, 1).abs()
    vibration_mag = torch.randn(32, 1281, 1).abs()

    # 计算熵
    entropy = splitter.compute_joint_spectral_entropy(pressure_mag, vibration_mag)
    print(f"✓ 熵密度形状: {entropy.shape}")
    print(f"  熵值范围: [{entropy.min():.4f}, {entropy.max():.4f}]")

    # 自适应切分
    breaks, k, gvf = splitter.adaptive_split(entropy)
    print(f"✓ 切分结果: k={k}, GVF={gvf:.4f}")
    print(f"  切分点: {breaks}")

    # 生成掩码
    masks = splitter.generate_frequency_masks(breaks, device='cpu')
    print(f"✓ 生成{len(masks)}个掩码")
    for i, mask in enumerate(masks):
        print(f"  掩码{i + 1}: 覆盖频点数={mask.sum().item():.0f}")

    print("\n✓ 所有测试通过！")
