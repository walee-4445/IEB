
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleDecoupledDistillationLoss(nn.Module):


    def __init__(self, temperature=4.0, alpha=0.5, local_weight_strategy='uniform'):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.local_weight_strategy = local_weight_strategy
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_global, student_locals,
                teacher_global, teacher_locals, entropy_weights=None):

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

