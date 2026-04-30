"""
Scale Decoupled Distillation Loss
尺度解耦蒸馏损失（SDD组件）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleDecoupledDistillationLoss(nn.Module):
    """
    SDD损失：L_sdd = α·L_global + (1-α)·L_local

    参数：
        temperature: 温度参数（软化概率分布）
        alpha: 全局-局部权重平衡系数
        local_weight_strategy: 局部头权重策略
            - 'uniform': 均匀权重（推荐）
            - 'adaptive': 自适应权重（基于熵值）
    """

    def __init__(self, temperature=4.0, alpha=0.5, local_weight_strategy='uniform'):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.local_weight_strategy = local_weight_strategy
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_global, student_locals,
                teacher_global, teacher_locals, entropy_weights=None):
        """
        计算尺度解耦蒸馏损失

        Args:
            student_global: (B, C) 学生全局logits
            student_locals: List[(B, C)] 学生局部logits，长度为k
            teacher_global: (B, C) 教师全局logits
            teacher_locals: List[(B, C)] 教师局部logits，长度为k
            entropy_weights: (k,) 可选，自适应权重（基于熵值）

        Returns:
            total_loss: 总蒸馏损失
            loss_global: 全局蒸馏损失（用于日志）
            loss_local: 局部蒸馏损失（用于日志）
        """
        T = self.temperature

        # 1. 全局KL散度
        loss_global = self.kl_div(
            F.log_softmax(student_global / T, dim=1),
            F.softmax(teacher_global / T, dim=1)
        ) * (T ** 2)

        # 2. 局部KL散度
        k = len(student_locals)

        if self.local_weight_strategy == 'uniform':
            # 均匀权重策略（推荐）
            loss_local = 0
            for s_local, t_local in zip(student_locals, teacher_locals):
                loss_local += self.kl_div(
                    F.log_softmax(s_local / T, dim=1),
                    F.softmax(t_local / T, dim=1)
                )
            loss_local = loss_local / k * (T ** 2)

        elif self.local_weight_strategy == 'adaptive':
            # 自适应权重策略（基于熵值）
            if entropy_weights is None:
                # 回退到均匀权重
                entropy_weights = torch.ones(k) / k
            else:
                # 归一化权重
                entropy_weights = entropy_weights / entropy_weights.sum()

            loss_local = 0
            for i, (s_local, t_local) in enumerate(zip(student_locals, teacher_locals)):
                loss_local += entropy_weights[i] * self.kl_div(
                    F.log_softmax(s_local / T, dim=1),
                    F.softmax(t_local / T, dim=1)
                )
            loss_local = loss_local * (T ** 2)

        else:
            raise ValueError(f"Unknown local_weight_strategy: {self.local_weight_strategy}")

        # 3. 加权组合
        total_loss = self.alpha * loss_global + (1 - self.alpha) * loss_local

        return total_loss, loss_global, loss_local


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("测试 ScaleDecoupledDistillationLoss")
    print("=" * 60)

    loss_fn = ScaleDecoupledDistillationLoss(temperature=4.0, alpha=0.5)

    # 模拟数据
    B, C, k = 32, 10, 4
    s_global = torch.randn(B, C)
    t_global = torch.randn(B, C)
    s_locals = [torch.randn(B, C) for _ in range(k)]
    t_locals = [torch.randn(B, C) for _ in range(k)]

    # 计算损失
    total_loss, loss_g, loss_l = loss_fn(s_global, s_locals, t_global, t_locals)

    print(f"✓ 总损失: {total_loss:.4f}")
    print(f"  全局损失: {loss_g:.4f}")
    print(f"  局部损失: {loss_l:.4f}")
    print(f"  加权组合: {0.5}*{loss_g:.4f} + {0.5}*{loss_l:.4f} = {total_loss:.4f}")

    # 测试反向传播
    total_loss.backward()
    print("✓ 反向传播成功")

    # 测试自适应权重
    print("\n测试自适应权重策略...")
    loss_fn_adaptive = ScaleDecoupledDistillationLoss(
        temperature=4.0, alpha=0.5, local_weight_strategy='adaptive'
    )
    entropy_weights = torch.tensor([0.4, 0.3, 0.2, 0.1])  # 高熵子带权重更大
    total_loss_adp, _, _ = loss_fn_adaptive(
        s_global, s_locals, t_global, t_locals, entropy_weights
    )
    print(f"✓ 自适应权重损失: {total_loss_adp:.4f}")

    print("\n✓ 所有测试通过！")
