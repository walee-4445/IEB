
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
                print(f"无法加载GPT-2，使用Transformer替代")
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

        #  维度投影层
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

        # 3. 分类头
        self.act = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, num_classes)

    def forward(self, pressure_mag, vibration_mag, freq_mask=None):
        batch_size = pressure_mag.size(0)

        if freq_mask is not None:
            freq_mask = freq_mask.to(pressure_mag.device)
            if freq_mask.dim() == 1:
                freq_mask_3d = freq_mask.view(1, -1, 1)  # (1, F, 1)
            else:
                freq_mask_3d = freq_mask.unsqueeze(-1)  # (B, F, 1)
            pressure_mag = pressure_mag * freq_mask_3d
            vibration_mag = vibration_mag * freq_mask_3d


        x_enc = torch.cat([pressure_mag, vibration_mag], dim=-1)


        x_enc = x_enc.permute(0, 2, 1)


        enc_out = self.cnn_encoder(x_enc)
        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.cnn_projection(enc_out)


        if self.input_projection is not None:
            enc_out = self.input_projection(enc_out)


        if self.use_huggingface_llm:
            outputs = self.llm_model(inputs_embeds=enc_out)
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = self.llm_model(enc_out)


        if self.output_projection is not None:
            hidden_states = self.output_projection(hidden_states)


        teacher_features = hidden_states.mean(dim=1)

        # 8. 分类
        output = self.act(teacher_features)
        output = self.dropout_layer(output)
        logits = self.projection(output)

        return logits, teacher_features


def generate_jenks_freq_masks(F_bins: int, jenks_cuts: list, device='cuda') -> list:

    masks = []

    for i in range(len(jenks_cuts) - 1):
        start_freq = jenks_cuts[i]
        end_freq = jenks_cuts[i + 1]

        # 创建频域掩码
        mask = torch.zeros(F_bins, device=device)
        mask[start_freq:end_freq] = 1.0

        masks.append(mask)

    return masks

