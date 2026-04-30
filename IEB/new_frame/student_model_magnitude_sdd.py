

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

        batch_size = pressure_mag.size(0)


        masked_p, masked_v, masks = self.modality_dropout(
            pressure_mag, vibration_mag, mode, fixed_mask
        )

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

        # 全局分类
        global_logits = self.global_head(self.classifier_dropout(fused_global))

        # 5. 局部特征提取（如果提供freq_masks）

        local_logits = []
        if freq_masks is not None:
            self.active_k = len(freq_masks)

            for i, mask in enumerate(freq_masks):

                mask_cpu = mask.cpu()
                nonzero_indices = torch.nonzero(mask_cpu, as_tuple=True)[0]

                if len(nonzero_indices) == 0:

                    local_logits.append(torch.zeros(batch_size, self.num_classes,
                                                    device=masked_p.device))
                    continue

                start_idx = nonzero_indices[0].item()
                end_idx = nonzero_indices[-1].item() + 1  # 左闭右开


                seg_p = masked_p[:, start_idx:end_idx, :]  # (B, F_local, 1)
                seg_v = masked_v[:, start_idx:end_idx, :]  # (B, F_local, 1)
                seg_p = F.interpolate(seg_p.permute(0, 2, 1), size=self.F_bins,
                                      mode='linear', align_corners=False).permute(0, 2, 1)
                seg_v = F.interpolate(seg_v.permute(0, 2, 1), size=self.F_bins,
                                      mode='linear', align_corners=False).permute(0, 2, 1)


                # 暂时冻结BN的running_stats更新
                self.pressure_encoder.eval()
                self.vibration_encoder.eval()
                if not self.use_dual_perceiver:
                    self.perceiver.eval()

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
