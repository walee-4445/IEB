"""
IEB Teacher Model (FD-MVLLM Architecture)
基于FD-MVLLM的IEB教师网络 - 使用冻结的LLM编码器
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Any
import inspect  # 🔧 修复：用于检查函数签名

class IEBTeacherMagnitude(nn.Module):
    """IEB教师网络（支持任意d_model维度）"""

    def __init__(self,
                 F_bins: int = 1281, #FFT 后的频点数量
                 num_classes: int = 9,
                 d_model: int = 768,#每个 patch token 的 embedding 维度
                 patch_len: int = 16, #一个 patch 包含多少频点
                 stride: int = 8, #patch 滑动步长
                 dropout: float = 0.1,
                 llm_model: Optional[nn.Module] = None,
                 freeze_llm: bool = True):
        super().__init__()

        self.F_bins = F_bins
        self.num_classes = num_classes
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride

        # 1. Patch Embedding  修改1：这里不再使用
        # self.patch_embedding = PatchEmbedding(
        #     d_model=d_model,
        #     seq_len=F_bins,
        #     patch_len=patch_len,
        #     stride=stride,
        #     dropout=dropout
        # )
        # ❌ 删除原代码:
        # self.patch_embedding = PatchEmbedding(
        #     d_model=d_model,
        #     seq_len=F_bins,
        #     patch_len=patch_len,
        #     stride=stride,
        #     dropout=dropout
        # )

        # ✅ 添加新代码:
        # 1. CNN特征提取器 (双模态输入)
        self.cnn_encoder = nn.Sequential(
            # 第1层: 提取低频特征
            nn.Conv1d(2, 64, kernel_size=7, padding=3),  # 2通道(压力+振动) → 64通道
            nn.BatchNorm1d(64),
            nn.GELU(),

            # 第2层: 提取中频特征
            nn.Conv1d(64, 128, kernel_size=5, padding=2),  # 64 → 128通道
            nn.BatchNorm1d(128),
            nn.GELU(),

            # 第3层: 提取高频特征
            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # 128 → 256通道
            nn.BatchNorm1d(256),
            nn.GELU(),

            # 自适应池化: 1281频点 → 160个特征向量
            nn.AdaptiveAvgPool1d(160)
        )

        # 2. 投影层: CNN特征 → LLM维度
        self.cnn_projection = nn.Sequential(
            nn.Linear(256, d_model),  # 256 → d_model (可能是256或768)
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        print(f"✓ 使用CNN编码器替代Patch Embedding")
        print(f"  CNN输出: (B, 256, 160) → 投影后: (B, 160, {d_model})")

        # 2. LLM
        self.use_huggingface_llm = False
        if llm_model is None:
            try:
                from transformers import GPT2Model
                self.llm_model = GPT2Model.from_pretrained('gpt2')
                self.use_huggingface_llm = True
                print("✓ 使用预训练GPT-2作为LLM编码器")
            except (ImportError, OSError) as e:
                print(f"⚠ 无法加载GPT-2，使用Transformer替代")
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True
                )
                self.llm_model = nn.TransformerEncoder(encoder_layer, num_layers=6)
                self.use_huggingface_llm = False
        else:
            self.llm_model = llm_model
            self.use_huggingface_llm = hasattr(llm_model, 'config')

        # 🔧 维度投影层
        if self.use_huggingface_llm:
            llm_hidden_size = 768
            if d_model != llm_hidden_size:
                self.input_projection = nn.Linear(d_model, llm_hidden_size)
                self.output_projection = nn.Linear(llm_hidden_size, d_model)
                print(f"✓ 添加投影层: {d_model} <-> {llm_hidden_size}")
            else:
                self.input_projection = None
                self.output_projection = None
        else:
            self.input_projection = None
            self.output_projection = None

        # 冻结LLM
        # if freeze_llm:
        #     for param in self.llm_model.parameters():
        #         param.requires_grad = False
        #     print("✓ LLM参数已冻结")

        # 3. 分类头
        self.act = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, num_classes)

    def forward(self, pressure_mag, vibration_mag, freq_mask=None):
        batch_size = pressure_mag.size(0)

        # ✅ 修复：在CNN之前应用频域掩码（与学生对齐）
        if freq_mask is not None:
            freq_mask = freq_mask.to(pressure_mag.device)  # ← 加这一行
            if freq_mask.dim() == 1:
                freq_mask_3d = freq_mask.view(1, -1, 1)  # (1, F, 1)
            else:
                freq_mask_3d = freq_mask.unsqueeze(-1)  # (B, F, 1)
            pressure_mag = pressure_mag * freq_mask_3d
            vibration_mag = vibration_mag * freq_mask_3d

        # 1. 拼接双模态
        x_enc = torch.cat([pressure_mag, vibration_mag], dim=-1)

        # 2. 转置适配Conv1d
        x_enc = x_enc.permute(0, 2, 1)

        # 3. CNN特征提取（现在处理的是掩码后的输入）
        enc_out = self.cnn_encoder(x_enc)
        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.cnn_projection(enc_out)

        # 4. 投影到GPT-2维度
        if self.input_projection is not None:
            enc_out = self.input_projection(enc_out)

        # 5. LLM编码
        if self.use_huggingface_llm:
            outputs = self.llm_model(inputs_embeds=enc_out)
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = self.llm_model(enc_out)

        # 6. 投影回原始维度
        if self.output_projection is not None:
            hidden_states = self.output_projection(hidden_states)

        # 7. 全局平均池化（不再需要feature_mask）
        teacher_features = hidden_states.mean(dim=1)

        # 8. 分类
        output = self.act(teacher_features)
        output = self.dropout_layer(output)
        logits = self.projection(output)

        return logits, teacher_features


def generate_jenks_freq_masks(F_bins: int, jenks_cuts: list, device='cuda') -> list:
    """
    将Jenks切分点转换为频域掩码

    Args:
        F_bins: 频域维度（1281）
        jenks_cuts: Jenks切分点列表，如 [0, 300, 600, 900, 1281]
        device: 设备

    Returns:
        freq_masks: List[Tensor(F,)] k个频域掩码
    """
    masks = []

    for i in range(len(jenks_cuts) - 1):
        start_freq = jenks_cuts[i]
        end_freq = jenks_cuts[i + 1]

        # 创建频域掩码
        mask = torch.zeros(F_bins, device=device)
        mask[start_freq:end_freq] = 1.0

        masks.append(mask)

    return masks

# if __name__ == '__main__':
#     print("=" * 60)
#     print("测试 IEBTeacherMagnitude (FD-MVLLM架构)")
#     print("=" * 60)
#
#     # 创建教师网络（使用GPT-2）
#     teacher = IEBTeacherMagnitude(
#         F_bins=1281,
#         num_classes=10,
#         d_model=768,  # GPT-2隐藏层维度
#         patch_len=16,
#         stride=8,
#         dropout=0.1,
#         llm_model=None,  # 自动加载GPT-2
#         freeze_llm=True
#     )
#
#     print(f"\n✓ 教师参数量: {sum(p.numel() for p in teacher.parameters()):,}")
#     print(f"✓ 可训练参数量: {sum(p.numel() for p in teacher.parameters() if p.requires_grad):,}")
#
#     # 模拟数据
#     B = 4
#     pressure = torch.randn(B, 1281, 1)
#     vibration = torch.randn(B, 1281, 1)
#
#     # 测试1：全局推理（无掩码）
#     print("\n" + "=" * 60)
#     print("测试1：全局推理（无掩码）")
#     print("=" * 60)
#     logits, features = teacher(pressure, vibration, freq_mask=None)
#     print(f"✓ Logits: {logits.shape}")
#     print(f"✓ Features: {features.shape}")
#
#     # 测试2：局部推理（带Jenks掩码）
#     print("\n" + "=" * 60)
#     print("测试2：局部推理（带Jenks掩码）")
#     print("=" * 60)
#
#     # 生成Jenks掩码
#     jenks_cuts = [0, 300, 600, 900, 1281]
#     freq_masks = generate_jenks_freq_masks(1281, jenks_cuts, device='cpu')
#
#     print(f"✓ 生成{len(freq_masks)}个频域掩码")
#
#     for i, mask in enumerate(freq_masks):
#         logits_local, features_local = teacher(
#             pressure, vibration, freq_mask=mask
#         )
#         active_bins = mask.sum().item()
#         print(f"  Mask {i + 1}: {active_bins:.0f}个激活频点, logits={logits_local.shape}")
#
#     # 测试3：反向传播
#     print("\n" + "=" * 60)
#     print("测试3：反向传播")
#     print("=" * 60)
#     loss = logits.mean()
#     loss.backward()
#     print("✓ 反向传播成功")
#
#     print("\n" + "=" * 60)
#     print("✓ 所有测试通过！FD-MVLLM架构正确实现")
#     print("=" * 60)

# """
# IEB Teacher Model for Magnitude Spectrum
# 支持频域掩码的教师网络（幅度谱输入）
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class IEBTeacherMagnitude(nn.Module):
#     """
#     支持IEB的教师网络（幅度谱输入）
#
#     架构：
#         输入：双模态幅度谱 (B, F, 2)
#         编码器：1D-CNN（替代Patch Embedding）
#         Backbone：Transformer Encoder
#         输出：全局logits + 局部logits
#
#     参数：
#         F_bins: 频域维度（1281）
#         num_classes: 分类类别数（10）
#         d_model: Transformer维度（256）
#         nhead: 注意力头数（8）
#         num_layers: Transformer层数（4）
#         dropout: Dropout概率（0.1）
#     """
#
#     def __init__(self, F_bins, num_classes, d_model=256,
#                  nhead=8, num_layers=4, dropout=0.1):
#         super().__init__()
#
#         self.F_bins = F_bins
#         self.num_classes = num_classes
#         self.d_model = d_model
#
#         # 1D-CNN编码器（替代Patch Embedding）
#         self.freq_encoder = nn.Sequential(
#             nn.Conv1d(2, 64, kernel_size=7, padding=3),
#             nn.BatchNorm1d(64),
#             nn.GELU(),
#             nn.Conv1d(64, d_model, kernel_size=5, padding=2),
#             nn.BatchNorm1d(d_model),
#             nn.GELU()
#         )
#
#         # Transformer编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=d_model * 4,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         # 分类头
#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, num_classes)
#         )
#
#     def forward(self, pressure_mag, vibration_mag, freq_mask=None):
#         """
#         前向传播
#
#         Args:
#             pressure_mag: (B, F, 1) 压力幅度谱
#             vibration_mag: (B, F, 1) 振动幅度谱
#             freq_mask: (F,) 频域掩码（可选）
#
#         Returns:
#             logits: (B, C) 分类logits
#             features: (B, D) 用于蒸馏的特征
#         """
#         B, F, _ = pressure_mag.shape
#
#         # 1. 拼接双模态
#         x = torch.cat([pressure_mag, vibration_mag], dim=2)  # (B, F, 2)
#
#         # 2. 应用频域掩码（如果提供）
#         if freq_mask is not None:
#             # 扩展掩码到双模态
#             mask_3d = freq_mask.view(1, -1, 1).expand(B, -1, 2)
#             x = x * mask_3d
#
#         # 3. CNN编码
#         x = x.permute(0, 2, 1)  # (B, 2, F)
#         x = self.freq_encoder(x)  # (B, D, F)
#         x = x.permute(0, 2, 1)  # (B, F, D)
#
#         # 4. Transformer编码
#         features_seq = self.transformer(x)  # (B, F, D)
#
#         # 5. 全局池化
#         if freq_mask is not None:
#             # 仅对有效频点池化
#             mask_expanded = freq_mask.view(1, -1, 1).expand(B, -1, self.d_model)
#             sum_features = torch.sum(features_seq * mask_expanded, dim=1)
#             count_valid = torch.sum(mask_expanded, dim=1) + 1e-9
#             features = sum_features / count_valid  # (B, D)
#         else:
#             # 全局平均池化
#             features = features_seq.mean(dim=1)  # (B, D)
#
#         # 6. 分类
#         logits = self.classifier(features)
#
#         return logits, features
#
#
# if __name__ == '__main__':
#     # 测试代码
#     print("=" * 60)
#     print("测试 IEBTeacherMagnitude")
#     print("=" * 60)
#
#     teacher = IEBTeacherMagnitude(F_bins=1281, num_classes=10)
#
#     print(f"✓ 模型参数量: {sum(p.numel() for p in teacher.parameters()):,}")
#
#     # 模拟数据
#     B = 4
#     pressure_mag = torch.randn(B, 1281, 1)
#     vibration_mag = torch.randn(B, 1281, 1)
#
#     # 全局推断
#     logits, features = teacher(pressure_mag, vibration_mag)
#     print(f"✓ 全局推断: logits={logits.shape}, features={features.shape}")
#     assert logits.shape == (B, 10)
#     assert features.shape == (B, 256)
#
#     # 局部推断（带掩码）
#     freq_mask = torch.zeros(1281)
#     freq_mask[0:400] = 1.0  # 仅保留前400个频点
#     logits_local, features_local = teacher(
#         pressure_mag, vibration_mag, freq_mask=freq_mask
#     )
#     print(f"✓ 局部推断: logits={logits_local.shape}, features={features_local.shape}")
#     assert logits_local.shape == (B, 10)
#
#     # 测试反向传播
#     loss = logits.mean()
#     loss.backward()
#     print("✓ 反向传播成功")
#
#     print("\n✓ 所有测试通过！")
###
###
### class PatchEmbedding(nn.Module):
#     """Patch嵌入层（来自FD-MVLLM）"""
#
#     def __init__(self, d_model, seq_len, patch_len, stride, dropout):
#         super(PatchEmbedding, self).__init__()
#         self.patch_len = patch_len
#         self.stride = stride
#         self.padding_patch_layer = ReplicationPad1d((0, stride))
#
#         # Token嵌入
#         self.value_embedding = TokenEmbedding(patch_len, d_model)
#
#         # Dropout
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         """
#         Args:
#             x: (B, F, C) 幅度谱
#         Returns:
#             (B, N_patches, D_model) Patch嵌入
#         """
#         n_vars = x.shape[1]
#         x = self.padding_patch_layer(x)
#         x = self.value_embedding(x)
#         return self.dropout(x), n_vars

# class PatchEmbedding(nn.Module):
#     """Patch嵌入层（修复版：适配幅度谱输入）"""
#
#     def __init__(self, d_model, seq_len, patch_len, stride, dropout):
#         super(PatchEmbedding, self).__init__()
#         self.patch_len = patch_len
#         self.stride = stride
#         self.seq_len = seq_len
#
#         # 填充层
#         self.padding_patch_layer = ReplicationPad1d((0, stride))
#
#         # Token嵌入：将patch_len个频点映射到d_model维
#         self.value_embedding = TokenEmbedding(patch_len, d_model)
#
#         # Dropout
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         """
#         Args:
#             x: (B, L, C) 幅度谱输入，L=频点数，C=模态数
#         Returns:
#             enc_out: (B, N, D) Patch序列，N=Patch数，D=d_model
#             n_vars: 模态数C
#         """
#         B, L, C = x.shape
#         n_vars = C
#
#         # ✅ 修复：逐模态处理
#         patches_list = []
#         for c in range(C):
#             # 提取单模态：(B, L)
#             x_c = x[:, :, c]
#
#             # 填充：(B, L) -> (B, L+stride)
#             x_padded = self.padding_patch_layer(x_c)
#
#             # Unfold切片：(B, L+stride) -> (B, N, patch_len)
#             x_unfolded = x_padded.unfold(dimension=1, size=self.patch_len, step=self.stride)
#
#             # Token嵌入：(B, N, patch_len) -> (B, N, d_model)
#             x_embedded = self.value_embedding(x_unfolded)
#
#             patches_list.append(x_embedded)
#
#         # 拼接所有模态：(B, N, d_model) * C -> (B, N*C, d_model)
#         enc_out = torch.cat(patches_list, dim=1)
#
#         # Dropout
#         enc_out = self.dropout(enc_out)
#
#         return enc_out, n_vars




    # def forward(self,
    #             pressure_mag: torch.Tensor,
    #             vibration_mag: torch.Tensor,
    #             freq_mask: Optional[torch.Tensor] = None):
    #     """
    #     前向传播 (CNN版本)
    #
    #     Args:
    #         pressure_mag: (B, F, 1) 压力幅度谱, F=1281
    #         vibration_mag: (B, F, 1) 振动幅度谱
    #         freq_mask: (F,) 频域掩码 (可选)
    #
    #     Returns:
    #         logits: (B, num_classes) 分类预测
    #         teacher_features: (B, d_model) 蒸馏特征
    #     """
    #     batch_size = pressure_mag.size(0)
    #     # ✅ 修复：在CNN之前应用频域掩码（与学生对齐）
    #     if freq_mask is not None:
    #         if freq_mask.dim() == 1:
    #             freq_mask_3d = freq_mask.view(1, -1, 1)  # (1, F, 1)
    #         else:
    #             freq_mask_3d = freq_mask.unsqueeze(-1)  # (B, F, 1)
    #         pressure_mag = pressure_mag * freq_mask_3d
    #         vibration_mag = vibration_mag * freq_mask_3d
    #
    #     # 1. 拼接双模态: (B, F, 1) + (B, F, 1) → (B, F, 2)
    #     x_enc = torch.cat([pressure_mag, vibration_mag], dim=-1)
    #
    #     # 2. 转置: (B, F, 2) → (B, 2, F) 适配Conv1d
    #     x_enc = x_enc.permute(0, 2, 1)
    #
    #     # # 3. CNN特征提取: (B, 2, 1281) → (B, 256, 160)
    #     # enc_out = self.cnn_encoder(x_enc)
    #     # 3. CNN特征提取（现在处理的是掩码后的输入）
    #     enc_out = self.cnn_encoder(x_enc)
    #     enc_out = enc_out.permute(0, 2, 1)
    #     enc_out = self.cnn_projection(enc_out)
    #
    #     # 4. 投影到GPT-2维度
    #     if self.input_projection is not None:
    #         enc_out = self.input_projection(enc_out)
    #
    #     # 5. LLM编码
    #     if self.use_huggingface_llm:
    #         outputs = self.llm_model(inputs_embeds=enc_out)
    #         hidden_states = outputs.last_hidden_state
    #     else:
    #         hidden_states = self.llm_model(enc_out)
    #
    #     # 6. 投影回原始维度
    #     if self.output_projection is not None:
    #         hidden_states = self.output_projection(hidden_states)
    #
    #     # 7. 全局平均池化（不再需要feature_mask）
    #     teacher_features = hidden_states.mean(dim=1)
    #
    #     # 8. 分类
    #     output = self.act(teacher_features)
    #     output = self.dropout_layer(output)
    #     logits = self.projection(output)
    #
    #     return logits, teacher_features
    #     # # 4. 转置: (B, 256, 160) → (B, 160, 256)
    #     # enc_out = enc_out.permute(0, 2, 1)
    #     #
    #     # # 5. 投影到LLM维度: (B, 160, 256) → (B, 160, d_model)
    #     # enc_out = self.cnn_projection(enc_out)
    #     #
    #     # num_features = enc_out.size(1)  # 160
    #     #
    #     # # 6. 应用频域掩码 (如果提供)
    #     # feature_mask = None
    #     # if freq_mask is not None:
    #     #     # 将频域掩码转换为特征级掩码
    #     #     # 每个特征对应 F_bins/num_features ≈ 1281/160 ≈ 8个频点
    #     #     feature_mask = torch.zeros(batch_size, num_features, 1, device=enc_out.device)
    #     #     freq_per_feature = self.F_bins // num_features
    #     #
    #     #     if freq_mask.dim() == 1:
    #     #         freq_mask = freq_mask.unsqueeze(0)
    #     #
    #     #     for f_idx in range(num_features):
    #     #         f_start = f_idx * freq_per_feature
    #     #         f_end = min(f_start + freq_per_feature, self.F_bins)
    #     #         f_center = (f_start + f_end) // 2
    #     #
    #     #         if f_center < freq_mask.size(1) and freq_mask[0, f_center] > 0:
    #     #             feature_mask[:, f_idx, :] = 1.0
    #     #
    #     #     # 应用掩码
    #     #     enc_out = enc_out * feature_mask
    #     #
    #     # # 7. 投影到GPT-2维度 (如果d_model≠768)
    #     # if self.input_projection is not None:
    #     #     enc_out = self.input_projection(enc_out)
    #     #
    #     # # 8. LLM编码
    #     # if self.use_huggingface_llm:
    #     #     outputs = self.llm_model(inputs_embeds=enc_out)
    #     #     hidden_states = outputs.last_hidden_state
    #     # else:
    #     #     hidden_states = self.llm_model(enc_out)
    #     #
    #     # # 9. 投影回原始维度
    #     # if self.output_projection is not None:
    #     #     hidden_states = self.output_projection(hidden_states)
    #     #
    #     # # 10. 特征聚合
    #     # if feature_mask is not None:
    #     #     # 掩码平均池化
    #     #     sum_features = torch.sum(hidden_states * feature_mask, dim=1)
    #     #     count_valid = torch.sum(feature_mask, dim=1) + 1e-9
    #     #     teacher_features = sum_features / count_valid
    #     # else:
    #     #     # 全局平均池化 (改进: 比取最后一个token更稳定)
    #     #     teacher_features = hidden_states.mean(dim=1)
    #     #
    #     # # 11. 分类
    #     # output = self.act(teacher_features)
    #     # output = self.dropout_layer(output)
    #     # logits = self.projection(output)
    #     #
    #     # return logits, teacher_features


#修改1
# def forward(self,
    #             pressure_mag: torch.Tensor,
    #             vibration_mag: torch.Tensor,
    #             freq_mask: Optional[torch.Tensor] = None):
    #
    #     batch_size = pressure_mag.size(0)
    #
    #     # 拼接
    #     x_enc = torch.cat([pressure_mag, vibration_mag], dim=-1)
    #
    #     # Patch Embedding
    #     enc_out, n_vars = self.patch_embedding(x_enc)
    #     # Patch化：把长信号变成一个个向量块（Tokens）
    #     num_patches = enc_out.size(1)
    #
    #     # 应用掩码
    #     patch_mask = None
    #     if freq_mask is not None:
    #         patch_mask = torch.zeros(batch_size, num_patches, 1, device=enc_out.device)
    #         if freq_mask.dim() == 1:
    #             freq_mask = freq_mask.unsqueeze(0)
    #
    #         for p_idx in range(num_patches):
    #             p_start = p_idx * self.stride
    #             p_end = min(p_start + self.patch_len, self.F_bins)
    #             p_center = (p_start + p_end) // 2
    #
    #             if p_center < freq_mask.size(1) and freq_mask[0, p_center] > 0:
    #                 patch_mask[:, p_idx, :] = 1.0
    #
    #         enc_out = enc_out * patch_mask
    #
    #     # 🔧 投影到GPT-2维度
    #     if self.input_projection is not None:
    #         enc_out = self.input_projection(enc_out)
    #
    #     # LLM编码
    #     if self.use_huggingface_llm:
    #         outputs = self.llm_model(inputs_embeds=enc_out)
    #         hidden_states = outputs.last_hidden_state
    #     else:
    #         hidden_states = self.llm_model(enc_out)
    #
    #     # 🔧 投影回原始维度
    #     if self.output_projection is not None:
    #         hidden_states = self.output_projection(hidden_states)
    #
    #     # 特征聚合
    #     if patch_mask is not None:
    #         sum_features = torch.sum(hidden_states * patch_mask, dim=1)
    #         count_valid = torch.sum(patch_mask, dim=1) + 1e-9
    #         teacher_features = sum_features / count_valid
    #     else:
    #         teacher_features = hidden_states[:, -1, :]
    #
    #     # 分类
    #     output = self.act(teacher_features)
    #     output = self.dropout_layer(output)
    #     logits = self.projection(output)
    #
    #     return logits, teacher_features

# class IEBTeacherMagnitude(nn.Module):
#     """
#     IEB教师网络（基于FD-MVLLM架构）
#
#     核心特性：
#     1. Patch Embedding：将频域幅度谱切片
#     2. 冻结的LLM编码器：使用预训练大模型（如GPT-2）
#     3. Jenks Mask支持：频域局部解耦
#     4. 特征聚合：Masked Average Pooling
#
#     参数：
#         F_bins: 频域维度（1281）
#         num_classes: 分类类别数（10）
#         d_model: LLM隐藏层维度（768 for GPT-2）
#         patch_len: Patch长度（16）
#         stride: Patch步长（8）
#         dropout: Dropout率（0.1）
#         llm_model: 预训练LLM模型（如GPT-2）
#         freeze_llm: 是否冻结LLM参数（True）
#     """
#
#     def __init__(self,
#                  F_bins: int = 1281,
#                  num_classes: int = 10,
#                  d_model: int = 768,
#                  patch_len: int = 16,
#                  stride: int = 8,
#                  dropout: float = 0.1,
#                  llm_model: Optional[nn.Module] = None,
#                  freeze_llm: bool = True):
#         super().__init__()
#
#         self.F_bins = F_bins
#         self.num_classes = num_classes
#         self.d_model = d_model
#         self.patch_len = patch_len
#         self.stride = stride
#
#         # ============================================================
#         # 1. Patch Embedding（将频域幅度谱切片）
#         # ============================================================
#         self.patch_embedding = PatchEmbedding(
#             d_model=d_model,
#             seq_len=F_bins,
#             patch_len=patch_len,
#             stride=stride,
#             dropout=dropout
#         )
#
#         # ============================================================
#         # 2. 冻结的LLM基座（大模型编码器）
#         # ============================================================
#         # 🔧 修复警告1：将GPT2Model导入移到try块外
#         self.use_huggingface_llm = False  # 标记是否使用HF模型
#
#         if llm_model is None:
#             # 如果未提供LLM，尝试使用GPT-2作为默认
#             try:
#                 from transformers import GPT2Model  # 🔧 修复：在try块内导入
#                 self.llm_model = GPT2Model.from_pretrained('gpt2')
#                 self.use_huggingface_llm = True
#                 print("✓ 使用预训练GPT-2作为LLM编码器")
#             except (ImportError, OSError) as e:
#                 # 🔧 修复：捕获ImportError和OSError（网络问题）
#                 print(f"⚠ 无法加载GPT-2 ({e})，使用简化的Transformer替代")
#                 # Fallback：使用标准Transformer
#                 encoder_layer = nn.TransformerEncoderLayer(
#                     d_model=d_model,
#                     nhead=8,
#                     dim_feedforward=d_model * 4,
#                     dropout=dropout,
#                     batch_first=True
#                 )
#                 self.llm_model = nn.TransformerEncoder(encoder_layer, num_layers=6)
#                 self.use_huggingface_llm = False
#         else:
#             self.llm_model = llm_model
#             # 检测是否为HuggingFace模型
#             self.use_huggingface_llm = hasattr(llm_model, 'config') and \
#                                        hasattr(llm_model, 'forward')
#
#         # 冻结LLM参数
#         if freeze_llm:
#             for param in self.llm_model.parameters():
#                 param.requires_grad = False
#             print("✓ LLM参数已冻结")
#
#         # ============================================================
#         # 3. 分类头
#         # ============================================================
#         self.act = nn.GELU()
#         self.dropout_layer = nn.Dropout(dropout)
#         self.projection = nn.Linear(d_model, num_classes)
#
#     def forward(self,
#                 pressure_mag: torch.Tensor,
#                 vibration_mag: torch.Tensor,
#                 freq_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         前向传播
#
#         Args:
#             pressure_mag: (B, F, 1) 压力幅度谱
#             vibration_mag: (B, F, 1) 振动幅度谱
#             freq_mask: (B, F) 或 (1, F) 频域掩码（可选）
#                       1=保留，0=屏蔽
#
#         Returns:
#             logits: (B, num_classes) 分类logits
#             features: (B, d_model) 聚合特征（用于蒸馏）
#         """
#         batch_size = pressure_mag.size(0)
#
#         # ============================================================
#         # 1. 拼接双模态幅度谱
#         # ============================================================
#         # (B, F, 1) + (B, F, 1) -> (B, F, 2)
#         x_enc = torch.cat([pressure_mag, vibration_mag], dim=-1)
#
#         # ============================================================
#         # 2. Patch Embedding
#         # ============================================================
#         # (B, F, 2) -> (B, N_patches, d_model)
#         enc_out, n_vars = self.patch_embedding(x_enc)
#         num_patches = enc_out.size(1)
#
#         # ============================================================
#         # 3. 应用Jenks频域掩码（核心：Zero-out Strategy）
#         # ============================================================
#         # 🔧 修复警告3：初始化patch_mask，避免未赋值引用
#         patch_mask = None
#
#         if freq_mask is not None:
#             # 将频域掩码转换为Patch级掩码
#             # freq_mask: (B, F) or (1, F)
#             # 需要转换为: (B, N_patches, 1)
#
#             # 🔧 修复：确保patch_mask被正确初始化
#             patch_mask = torch.zeros(batch_size, num_patches, 1, device=enc_out.device)
#
#             # 确保freq_mask是2D的
#             if freq_mask.dim() == 1:
#                 freq_mask = freq_mask.unsqueeze(0)
#
#             # 对每个Patch，检查其覆盖的频点是否被激活
#             for p_idx in range(num_patches):
#                 p_start = p_idx * self.stride
#                 p_end = min(p_start + self.patch_len, self.F_bins)
#                 p_center = (p_start + p_end) // 2
#
#                 # 如果Patch中心点被激活，则保留该Patch
#                 if p_center < freq_mask.size(1) and freq_mask[0, p_center] > 0:
#                     patch_mask[:, p_idx, :] = 1.0
#
#             # 应用掩码（硬屏蔽）
#             enc_out = enc_out * patch_mask
#
#         # ============================================================
#         # 4. 送入LLM编码器
#         # ============================================================
#         # 🔧 修复警告2：使用inspect检查函数签名，避免__code__警告
#         if self.use_huggingface_llm:
#             # HuggingFace模型（如GPT-2）
#             outputs = self.llm_model(inputs_embeds=enc_out)
#             hidden_states = outputs.last_hidden_state  # (B, N_patches, d_model)
#         else:
#             # 标准Transformer
#             hidden_states = self.llm_model(enc_out)  # (B, N_patches, d_model)
#
#         # ============================================================
#         # 5. 特征聚合（Masked Average Pooling）
#         # ============================================================
#         if patch_mask is not None:
#             # 仅对未被Mask的有效Patch进行平均池化
#             sum_features = torch.sum(hidden_states * patch_mask, dim=1)
#             count_valid = torch.sum(patch_mask, dim=1) + 1e-9
#             teacher_features = sum_features / count_valid  # (B, d_model)
#         else:
#             # 无Mask时使用最后一个Token（或全局平均）
#             teacher_features = hidden_states[:, -1, :]  # (B, d_model)
#
#         # ============================================================
#         # 6. 分类头
#         # ============================================================
#         output = self.act(teacher_features)
#         output = self.dropout_layer(output)
#         logits = self.projection(output)  # (B, num_classes)
#
#         return logits, teacher_features

# from new_frame.magnitude_encoder import MagnitudeEncoder


# class TokenEmbedding(nn.Module):
#     """Token嵌入层（来自FD-MVLLM）"""
#
#     def __init__(self, c_in, d_model):
#         super(TokenEmbedding, self).__init__()
#         padding = 1 if torch.__version__ >= '1.5.0' else 2
#         self.tokenConv = nn.Conv1d(
#             in_channels=c_in,
#             out_channels=d_model,
#             kernel_size=3,
#             padding=padding,
#             padding_mode='circular',
#             bias=False
#         )
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_in', nonlinearity='leaky_relu'
#                 )
#
#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1))
#         return x.transpose(1, 2)


# class ReplicationPad1d(nn.Module):
#     """复制填充层（来自FD-MVLLM）"""
#
#     def __init__(self, padding) -> None:
#         super(ReplicationPad1d, self).__init__()
#         self.padding = padding
#
#     def forward(self, input: Tensor) -> Tensor:
#         replicate_padding = input[:, -1].unsqueeze(-1).repeat(1, self.padding[-1])
#         output = torch.cat([input, replicate_padding], dim=-1)
#         return output
# class TokenEmbedding(nn.Module):
#     """Token嵌入层（修复版）"""
#     def __init__(self, patch_len, d_model):
#         super(TokenEmbedding, self).__init__()
#         self.projection = nn.Linear(patch_len, d_model)
#         nn.init.kaiming_normal_(self.projection.weight, mode='fan_in', nonlinearity='leaky_relu')
#
#     def forward(self, x):
#         return self.projection(x)
# class ReplicationPad1d(nn.Module):
#     """复制填充层（修复版：支持3D输入）"""
#
#     def __init__(self, padding) -> None:
#         super(ReplicationPad1d, self).__init__()
#         self.padding = padding
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         """R
#         Args:
#             input: (B, L) 或 (B, L, C)
#         Returns:
#             output: (B, L+padding) 或 (B, L+padding, C)
#         """
#         # ✅ 修复：支持2D和3D输入
#         if input.dim() == 2:
#             # 原始逻辑：(B, L)
#             replicate_padding = input[:, -1].unsqueeze(-1).repeat(1, self.padding[-1])
#             output = torch.cat([input, replicate_padding], dim=-1)
#         elif input.dim() == 3:
#             # 新增逻辑：(B, L, C)
#             replicate_padding = input[:, -1:, :].repeat(1, self.padding[-1], 1)
#             output = torch.cat([input, replicate_padding], dim=1)
#         else:
#             raise ValueError(f"Unsupported input dimension: {input.dim()}")
#
#         return output
