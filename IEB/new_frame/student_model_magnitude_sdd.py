"""
Multi-Head Student Model for Magnitude Spectrum with SDD
多头学生网络（支持尺度解耦蒸馏）
"""

import torch
import torch.nn as nn
import numpy as np
import math
import sys
import os
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magnitude_encoder import MagnitudeEncoder
from perceiver import PerceiverFusion

# 导入PV-NFM
try:
    from new_frame.pvnfm_magnitude_fixed_v2 import PVNFMMagnitudeFixedV2 as PVNFMMagnitude
except ImportError:
    PVNFMMagnitude = None


class ModalityDropoutMagnitude(nn.Module):
    """模态缺失模拟模块（频域版本）"""

    def __init__(self):
        super().__init__()
        self.modality_combinations = [
            [1, 0],  # 仅压力
            [0, 1],  # 仅振动
            [1, 1],  # 完整
        ]
        self.train_probs = np.array([0.33, 0.33, 0.34])

    def forward(self, pressure_mag, vibration_mag, mode='train', fixed_mask=None,
            enable_pvnfm_inference=False, pvnfm_inference_weight=1.0):
        batch_size = pressure_mag.size(0)
        device = pressure_mag.device

        if mode == 'train' and fixed_mask is None:
            masks = []
            for _ in range(batch_size):
                idx = np.random.choice(len(self.modality_combinations), p=self.train_probs)
                masks.append(self.modality_combinations[idx])
            masks = torch.tensor(masks, dtype=torch.float32, device=device)
        else:
            if fixed_mask is None:
                masks = torch.ones(batch_size, 2, device=device)
            else:
                masks = fixed_mask.float().to(device)

        p_mask = masks[:, 0].view(-1, 1, 1)
        v_mask = masks[:, 1].view(-1, 1, 1)

        masked_pressure = pressure_mag * p_mask
        masked_vibration = vibration_mag * v_mask

        return masked_pressure, masked_vibration, masks


class StudentModelMagnitudeSDD(nn.Module):
    """
    多头学生网络：1个全局头 + k个局部头

    参数：
        F_bins: 频域维度（1281）
        num_classes: 分类类别数（10）
        encoder_channels: CNN通道数（[32, 64]）
        latent_dim: 特征维度（128）
        num_latents: Perceiver latent数量（16）
        perceiver_depth: Perceiver深度（3）
        perceiver_heads: Perceiver注意力头数（8）
        max_k: 最大局部头数量（6）
        use_pvnfm: 是否使用PV-NFM（True）
        use_dual_perceiver: 是否使用双Perceiver（False）
        use_transformer_fusion: 是否使用Transformer融合（False）
        decay_schedule: NFM权重衰减策略（'piecewise'）

        nfm_initial_weight: NFM初始权重（0.4）
        nfm_final_weight: NFM最终权重（0.0）
        nfm_warmup_ratio: NFM warmup比例（0.7）
        nfm_decay_ratio: NFM decay比例（0.3）

        pvnfm_context_dim: PV-NFM上下文维度（128）
        pvnfm_num_bands: PV-NFM傅里叶频带数（16）
        pvnfm_hidden_inr: PV-NFM INR隐藏层维度（128）
    """

    def __init__(self,
                 F_bins=1281,
                 num_classes=9,
                 encoder_channels=[32, 64],
                 latent_dim=128,
                 num_latents=16,
                 perceiver_depth=3,
                 perceiver_heads=8,
                 max_k=6,
                 use_pvnfm=False,
                 use_dual_perceiver=False,
                 use_transformer_fusion=False,
                 decay_schedule='piecewise',
                 nfm_initial_weight=0.4,
                 nfm_final_weight=0.0,
                 nfm_warmup_ratio=0.7,
                 nfm_decay_ratio=0.3,
                 pvnfm_context_dim=128,
                 pvnfm_num_bands=16,
                 pvnfm_hidden_inr=128):
        super().__init__()

        self.F_bins = F_bins
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.max_k = max_k
        self.use_pvnfm = use_pvnfm
        self.use_dual_perceiver = use_dual_perceiver
        self.use_transformer_fusion = use_transformer_fusion

        # NFM权重衰减参数
        self.decay_schedule = decay_schedule
        self.nfm_initial_weight = nfm_initial_weight
        self.nfm_final_weight = nfm_final_weight
        self.nfm_warmup_ratio = nfm_warmup_ratio
        self.nfm_decay_ratio = nfm_decay_ratio

        # 模态缺失模块
        self.modality_dropout = ModalityDropoutMagnitude()

        # 频域编码器（共享）
        self.pressure_encoder = MagnitudeEncoder(
            F_bins=F_bins,
            channels=encoder_channels,
            feature_dim=latent_dim
        )
        self.vibration_encoder = MagnitudeEncoder(
            F_bins=F_bins,
            channels=encoder_channels,
            feature_dim=latent_dim
        )

        # PV-NFM生成器
        if use_pvnfm and PVNFMMagnitude is not None:
            self.modality_generator = PVNFMMagnitude(
                F_bins=F_bins,
                context_dim=pvnfm_context_dim,
                num_bands=pvnfm_num_bands,
                hidden_inr=pvnfm_hidden_inr
            )
        else:
            self.modality_generator = None

        # Perceiver融合模块
        if use_dual_perceiver:
            self.perceiver_p = PerceiverFusion(
                input_dim=latent_dim,
                latent_dim=latent_dim,
                num_latents=num_latents,
                depth=perceiver_depth,
                heads=perceiver_heads
            )
            self.perceiver_v = PerceiverFusion(
                input_dim=latent_dim,
                latent_dim=latent_dim,
                num_latents=num_latents,
                depth=perceiver_depth,
                heads=perceiver_heads
            )
        else:
            self.perceiver = PerceiverFusion(
                input_dim=latent_dim,
                latent_dim=latent_dim,
                num_latents=num_latents,
                depth=perceiver_depth,
                heads=perceiver_heads
            )

        # 跨模态融合
        if use_transformer_fusion:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=perceiver_heads,
                dim_feedforward=latent_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fusion = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        # 全局分类头
        # 这个池化层的作用是：把截取出来的任意长度频段（比如320个频点、321
        # 个频点）统一插值回
        # F_bins = 1281
        # 的长度，这样就能复用现有的共享编码器，无需创建新的编码器。
        self.local_pool = nn.AdaptiveAvgPool1d(F_bins)  # 统一回 F_bins 长度
        self.classifier_dropout = nn.Dropout(0.3)  # 新增
        self.global_head = nn.Linear(latent_dim, num_classes)

        # 局部分类头（预分配max_k个）
        self.local_heads = nn.ModuleList([
            nn.Linear(latent_dim, num_classes) for _ in range(max_k)
        ])

        # 当前激活的局部头数量
        self.active_k = 0

    def compute_nfm_weight(self, current_epoch, total_epochs):
        """计算NFM权重（渐进式衰减）"""
        if total_epochs <= 0:
            return self.nfm_final_weight

        progress = current_epoch / total_epochs

        if self.decay_schedule == 'linear':
            weight = self.nfm_initial_weight - \
                     (self.nfm_initial_weight - self.nfm_final_weight) * progress
        elif self.decay_schedule == 'cosine':
            weight = self.nfm_final_weight + \
                     (self.nfm_initial_weight - self.nfm_final_weight) * \
                     0.5 * (1 + math.cos(math.pi * progress))
        elif self.decay_schedule == 'piecewise':
            if progress < self.nfm_warmup_ratio:
                weight = self.nfm_initial_weight
            else:
                decay_progress = (progress - self.nfm_warmup_ratio) / self.nfm_decay_ratio
                decay_progress = min(1.0, decay_progress)
                weight = self.nfm_initial_weight - \
                         (self.nfm_initial_weight - self.nfm_final_weight) * decay_progress
        else:
            raise ValueError(f"Unknown decay schedule: {self.decay_schedule}")

        weight = max(self.nfm_final_weight, min(self.nfm_initial_weight, weight))
        return weight

    def forward(self, pressure_mag, vibration_mag, freq_masks=None,
                mode='train', fixed_mask=None, current_epoch=0, total_epochs=30,
            enable_pvnfm_inference=False, pvnfm_inference_weight=1.0):
        """
        前向传播

        Args:
            pressure_mag: (B, F, 1) 压力幅度谱
            vibration_mag: (B, F, 1) 振动幅度谱
            freq_masks: List[Tensor(F,)] k个频域掩码（可选）
            mode: 'train' or 'test'
            fixed_mask: (B, 2) 固定的模态mask
            current_epoch: 当前轮次
            total_epochs: 总轮次

        Returns:
            global_logits: (B, C) 全局分类logits
            local_logits: List[(B, C)] 局部分类logits（如果提供freq_masks）
        """
        batch_size = pressure_mag.size(0)

        # 1. 模态缺失模拟
        masked_p, masked_v, masks = self.modality_dropout(
            pressure_mag, vibration_mag, mode, fixed_mask
        )

        # 2. PV-NFM跨模态补全（渐进式权重衰减）
        # 2. PV-NFM跨模态补全（训练 / 测试统一支持）
        if self.modality_generator is not None:
            if mode == 'train':
                nfm_weight = self.compute_nfm_weight(current_epoch, total_epochs)
            elif enable_pvnfm_inference:
                nfm_weight = pvnfm_inference_weight
            else:
                nfm_weight = 0.0

            if nfm_weight > 0:
                p_missing = (masks[:, 0] == 0)
                v_missing = (masks[:, 1] == 0)

                if p_missing.any():
                    gen_p = self.modality_generator(masked_v[p_missing], direction='v2p')
                    masked_p[p_missing] = nfm_weight * gen_p + \
                                          (1 - nfm_weight) * masked_p[p_missing]

                if v_missing.any():
                    gen_v = self.modality_generator(masked_p[v_missing], direction='p2v')
                    masked_v[v_missing] = nfm_weight * gen_v + \
                                          (1 - nfm_weight) * masked_v[v_missing]
        # 3. 全局特征提取
        feat_p = self.pressure_encoder(masked_p)
        feat_v = self.vibration_encoder(masked_v)

        if self.use_dual_perceiver:
            latent_p = self.perceiver_p([feat_p])
            latent_v = self.perceiver_v([feat_v])

            if self.use_transformer_fusion:
                latent_p = latent_p.unsqueeze(1).expand(-1, self.num_latents, -1)
                latent_v = latent_v.unsqueeze(1).expand(-1, self.num_latents, -1)
                stacked = torch.stack([latent_p, latent_v], dim=0)
                stacked = stacked.permute(1, 0, 2, 3).reshape(batch_size, -1, self.latent_dim)
                fused = self.fusion_transformer(stacked)
                fused_p = fused[:, :fused.size(1) // 2].mean(dim=1)
                fused_v = fused[:, fused.size(1) // 2:].mean(dim=1)
            else:
                fused_p = latent_p
                fused_v = latent_v

            fused_all = torch.cat([fused_p, fused_v], dim=1)
            fused_global = self.fusion(fused_all)
        else:
            fused_global = self.perceiver([feat_p, feat_v])

        # 4. 全局分类
        global_logits = self.global_head(self.classifier_dropout(fused_global))
        # 第313-339行，替换为：
        # 5. 局部特征提取（如果提供freq_masks）
        # 5. 局部特征提取（方案C：频段截取 + 自适应池化）
        local_logits = []
        if freq_masks is not None:
            self.active_k = len(freq_masks)

            for i, mask in enumerate(freq_masks):
                # ① 找到掩码的有效频段范围 [start, end)
                mask_cpu = mask.cpu()
                nonzero_indices = torch.nonzero(mask_cpu, as_tuple=True)[0]

                if len(nonzero_indices) == 0:
                    # 空掩码 → 跳过或用零logits
                    local_logits.append(torch.zeros(batch_size, self.num_classes,
                                                    device=masked_p.device))
                    continue

                start_idx = nonzero_indices[0].item()
                end_idx = nonzero_indices[-1].item() + 1  # 左闭右开

                # ② 截取有效频段（而非全谱掩零）
                seg_p = masked_p[:, start_idx:end_idx, :]  # (B, F_local, 1)
                seg_v = masked_v[:, start_idx:end_idx, :]  # (B, F_local, 1)
                seg_p = F.interpolate(seg_p.permute(0, 2, 1), size=self.F_bins,
                                      mode='linear', align_corners=False).permute(0, 2, 1)
                seg_v = F.interpolate(seg_v.permute(0, 2, 1), size=self.F_bins,
                                      mode='linear', align_corners=False).permute(0, 2, 1)

                # ③ 自适应池化：统一到 F_bins 长度，复用共享编码器
                #    (B, F_local, 1) → permute → (B, 1, F_local) → pool → (B, 1, F_bins) → permute
                # seg_p = self.local_pool(seg_p.permute(0, 2, 1)).permute(0, 2, 1)  # (B, F_bins, 1)
                # seg_v = self.local_pool(seg_v.permute(0, 2, 1)).permute(0, 2, 1)  # (B, F_bins, 1)

                # 暂时冻结BN的running_stats更新
                self.pressure_encoder.eval()
                self.vibration_encoder.eval()
                if not self.use_dual_perceiver:
                    self.perceiver.eval()
                # ④ 编码（stop-gradient，不污染共享编码器）
                with torch.no_grad():
                    local_feat_p = self.pressure_encoder(seg_p)
                    local_feat_v = self.vibration_encoder(seg_v)
                    if self.use_dual_perceiver:
                        local_latent_p = self.perceiver_p([local_feat_p])
                        local_latent_v = self.perceiver_v([local_feat_v])
                        local_fused_all = torch.cat([local_latent_p, local_latent_v], dim=1)
                        local_fused = self.fusion(local_fused_all)
                    else:
                        local_fused = self.perceiver([local_feat_p, local_feat_v])
                self.pressure_encoder.train()
                self.vibration_encoder.train()
                if not self.use_dual_perceiver:
                    self.perceiver.train()
                # ⑤ 分类（只有 local_heads 接收梯度）
                local_logit = self.local_heads[i](local_fused)
                local_logits.append(local_logit)

        return global_logits, local_logits
#
#         # local_logits = []
#         # # if freq_masks is not None:
#         # #     self.active_k = len(freq_masks)
#         # #
#         # #     for i, mask in enumerate(freq_masks):
#         # #         # 应用频域掩码
#         # #         mask_3d = mask.to(masked_p.device).view(1, -1, 1).expand(batch_size, -1, 1)
#         # #         masked_p_i = masked_p * mask_3d
#         # #         masked_v_i = masked_v * mask_3d
#         # #
#         # #         # 编码（共享编码器，保留梯度！）
#         # #         local_feat_p = self.pressure_encoder(masked_p_i)
#         # #         local_feat_v = self.vibration_encoder(masked_v_i)
#         # #
#         # #         # 融合（保留梯度，但用较小的学习率缩放）
#         # #         if self.use_dual_perceiver:
#         # #             local_latent_p = self.perceiver_p([local_feat_p])
#         # #             local_latent_v = self.perceiver_v([local_feat_v])
#         # #             local_fused_all = torch.cat([local_latent_p, local_latent_v], dim=1)
#         # #             local_fused = self.fusion(local_fused_all)
#         # #         else:
#         # #             local_fused = self.perceiver([local_feat_p, local_feat_v])
#         # #
#         # #         # 梯度缩放：局部分支对编码器的梯度贡献缩小为 0.1x
#         # #         # 避免局部掩码的噪声梯度干扰全局分支
#         # #         local_fused_scaled = fused_global.detach() + 0.1 * (local_fused - local_fused.detach())
#         # #
#         # #         # 分类（独立头）
#         # #         local_logit = self.local_heads[i](local_fused_scaled)
#         # #         local_logits.append(local_logit)
#         # # student_model_magnitude_sdd.py 第313-343行，修改为：
#         #
#         # if freq_masks is not None:
#         #     self.active_k = len(freq_masks)
#         #
#         #     for i, mask in enumerate(freq_masks):
#         #         mask_3d = mask.to(masked_p.device).view(1, -1, 1).expand(batch_size, -1, 1)
#         #         masked_p_i = masked_p * mask_3d
#         #         masked_v_i = masked_v * mask_3d
#         #
#         #         # ✅ 关键修改：完全阻断局部分支对编码器的梯度
#         #         with torch.no_grad():
#         #             local_feat_p = self.pressure_encoder(masked_p_i)
#         #             local_feat_v = self.vibration_encoder(masked_v_i)
#         #             if self.use_dual_perceiver:
#         #                 local_latent_p = self.perceiver_p([local_feat_p])
#         #                 local_latent_v = self.perceiver_v([local_feat_v])
#         #                 local_fused_all = torch.cat([local_latent_p, local_latent_v], dim=1)
#         #                 local_fused = self.fusion(local_fused_all)
#         #             else:
#         #                 local_fused = self.perceiver([local_feat_p, local_feat_v])
#         #
#         #         # ✅ 只有 local_heads[i] 接收梯度
#         #         local_logit = self.local_heads[i](local_fused)
#         #         local_logits.append(local_logit)
#         #
#         # return global_logits, local_logits
#
#         # 5. 局部特征提取（如果提供freq_masks）
#         # local_logits = []
#         # if freq_masks is not None:
#         #     self.active_k = len(freq_masks)
#         #
#         #     for i, mask in enumerate(freq_masks):
#         #         # 应用频域掩码
#         #         mask_3d = mask.to(masked_p.device).view(1, -1, 1).expand(batch_size, -1, 1)  # 添加.to(masked_p.device)
#         #         masked_p_i = masked_p * mask_3d
#         #         masked_v_i = masked_v * mask_3d
#         #
#         #         # 编码（复用编码器）
#         #         local_feat_p = self.pressure_encoder(masked_p_i).detach()
#         #         local_feat_v = self.vibration_encoder(masked_v_i).detach()
#         #
#         #         # 融合
#         #         if self.use_dual_perceiver:
#         #             local_latent_p = self.perceiver_p([local_feat_p])
#         #             local_latent_v = self.perceiver_v([local_feat_v])
#         #             local_fused_all = torch.cat([local_latent_p, local_latent_v], dim=1)
#         #             local_fused = self.fusion(local_fused_all)
#         #         else:
#         #             local_fused = self.perceiver([local_feat_p, local_feat_v]).detach()
#         #
#         #         # 分类（独立头）
#         #         local_logit = self.local_heads[i](local_fused)
#         #         local_logits.append(local_logit)
#         #
#         # return global_logits, local_logits
#
#
# if __name__ == '__main__':
#     print("=" * 60)
#     print("测试 StudentModelMagnitudeSDD")
#     print("=" * 60)
#
#     student = StudentModelMagnitudeSDD(
#         F_bins=1281, num_classes=9,
#         encoder_channels=[32, 64],
#         latent_dim=128, max_k=6,
#         use_pvnfm=False
#     )
#
#     print(f"✓ 模型参数量: {sum(p.numel() for p in student.parameters()):,}")
#
#     # 模拟数据
#     B = 4
#     pressure_mag = torch.randn(B, 1281, 1)
#     vibration_mag = torch.randn(B, 1281, 1)
#
#     # 生成频域掩码
#     freq_masks = [torch.zeros(1281) for _ in range(4)]
#     freq_masks[0][0:300] = 1.0
#     freq_masks[1][300:600] = 1.0
#     freq_masks[2][600:900] = 1.0
#     freq_masks[3][900:1281] = 1.0
#
#     # 前向传播
#     global_logits, local_logits = student(
#         pressure_mag, vibration_mag,
#         freq_masks=freq_masks,
#         mode='test'
#     )
#
#     print(f"✓ 全局logits: {global_logits.shape}")
#     print(f"✓ 局部logits: {len(local_logits)}个, 每个形状={local_logits[0].shape}")
#
#     # 测试反向传播
#     loss = global_logits.mean() + sum([l.mean() for l in local_logits])
#     loss.backward()
#     print("✓ 反向传播成功")
#
#     print("\n✓ 所有测试通过！")
