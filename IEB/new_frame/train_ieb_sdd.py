
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from jenks_entropy_splitter import JenksEntropySplitter
from sdd_loss import ScaleDecoupledDistillationLoss

class IEBSDDTrainer:


    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 device: str = 'cuda',
                 # IEB参数
                 use_ieb: bool = True,
                 ieb_start_epoch: int = 10,
                 ieb_update_interval: int = 5,
                 min_k: int = 3,
                 max_k: int = 6,
                 gvf_threshold: float = 0.8,
                 # SDD参数
                 use_sdd: bool = True,
                 sdd_alpha: float = 0.5,
                 sdd_temperature: float = 4.0,
                 local_weight_strategy: str = 'uniform',
                 # 损失权重
                 ce_weight: float = 1.0,
                 distill_weight: float = 0.7,
                 recon_weight: float = 0.1,
                 # 梯度控制
                 gradient_clip_norm: float = 1.0,
                 # 验证/测试时PV-NFM补全
                 use_pvnfm_inference: bool = False,
                 pvnfm_inference_weight: float = 1.0,
                 # 日志
                 log_interval: int = 10):

        self.teacher = teacher_model
        self.student = student_model
        self.device = device

        self.use_pvnfm_inference = use_pvnfm_inference
        self.pvnfm_inference_weight = pvnfm_inference_weight

        # IEB配置
        self.use_ieb = use_ieb
        self.ieb_start_epoch = ieb_start_epoch
        self.ieb_update_interval = ieb_update_interval
        self.min_k = min_k
        self.max_k = max_k
        self.gvf_threshold = gvf_threshold

        # SDD配置
        self.use_sdd = use_sdd
        self.sdd_alpha = sdd_alpha
        self.sdd_temperature = sdd_temperature
        self.local_weight_strategy = local_weight_strategy

        # 损失权重
        self.ce_weight = ce_weight
        self.distill_weight = distill_weight
        self.recon_weight = recon_weight

        # 梯度控制
        self.gradient_clip_norm = gradient_clip_norm

        # 日志
        self.log_interval = log_interval

        # 初始化组件
        self.jenks_splitter = JenksEntropySplitter(
            min_k=min_k, max_k=max_k, gvf_threshold=gvf_threshold
        )

        self.sdd_loss = ScaleDecoupledDistillationLoss(
            alpha=sdd_alpha,
            temperature=sdd_temperature,
            local_weight_strategy=local_weight_strategy
            #weight_strategy=local_weight_strategy
        )

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 当前频域划分
        self.current_freq_masks = None
        self.current_k = min_k
        self.previous_freq_masks = None
        self.mask_transition_progress = 1.0
        self.mask_transition_epochs = 5
        self.mask_transition_start_epoch = -1

        # 移动模型到设备
        self.teacher.to(device)
        self.student.to(device)
        self.teacher.eval()

    def update_frequency_split(self, dataloader: DataLoader, epoch: int) -> bool:
        """更新频域划分（IEB）- 固定k版本"""

        #no_ieb模式：固定均匀掩码
        if not self.use_ieb:
            if self.current_freq_masks is None:
                F_bins = self.student.F_bins
                self.current_freq_masks = self._generate_uniform_masks(F_bins, self.min_k)
                self.current_k = self.min_k
                logging.info(f"[IEB] 固定划分模式（no_ieb）: k={self.min_k}")
            return False

        # ---- 延迟启动阶段：均匀掩码 ----
        if epoch < self.ieb_start_epoch:
            if self.current_freq_masks is None:
                F_bins = self.student.F_bins
                self.current_freq_masks = self._generate_uniform_masks(F_bins, self.min_k)
                self.current_k = self.min_k
                logging.info(f"[IEB] 延迟启动阶段: 使用均匀k={self.min_k}")
            return False

        # ---- 检查更新间隔 ----
        if (epoch - self.ieb_start_epoch) % self.ieb_update_interval != 0:
            return False

        logging.info(f"[IEB] Epoch {epoch}: 开始更新频域划分...")

        pressure_samples = []
        vibration_samples = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 50:
                    break
                if isinstance(batch, dict):
                    pressure_mag = batch['pressure'].to(self.device)
                    vibration_mag = batch['vibration'].to(self.device)
                else:
                    pressure_mag, vibration_mag, _ = batch
                    pressure_mag = pressure_mag.to(self.device)
                    vibration_mag = vibration_mag.to(self.device)
                pressure_samples.append(pressure_mag.cpu())
                vibration_samples.append(vibration_mag.cpu())

        pressure_samples = torch.cat(pressure_samples, dim=0)
        vibration_samples = torch.cat(vibration_samples, dim=0)


        try:
            entropy_density = self.jenks_splitter.compute_joint_spectral_entropy(
                pressure_samples, vibration_samples
            )
            breaks, k, gvf = self.jenks_splitter.adaptive_split(entropy_density)


            freq_masks = self.jenks_splitter.generate_frequency_masks(
                breaks, device=self.device
            )

            # 直接替换掩码（k不变，无需过渡）
            old_breaks = "均匀" if self.current_freq_masks is None else "Jenks"
            self.current_freq_masks = freq_masks
            self.current_k = k

            logging.info(f"[IEB] ✓ 切分位置更新: k={k}, GVF={gvf:.4f}")
            logging.info(f"[IEB]   切分点: {breaks}")
            return True

        except Exception as e:
            logging.warning(f"[IEB] ✗ 更新失败: {e}, 保持当前划分")
            return False

    def _generate_uniform_masks(self, F_bins: int, k: int) -> List[torch.Tensor]:
        """生成均匀频域掩码（fallback）"""
        masks = []
        step = F_bins // k

        for i in range(k):
            mask = torch.zeros(F_bins)
            start = i * step
            end = (i + 1) * step if i < k - 1 else F_bins
            mask[start:end] = 1.0
            masks.append(mask)

        return masks

    def train_epoch(self,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    total_epochs: int,
                    logger: Optional[logging.Logger] = None) -> Dict[str, float]:
        """训练一个epoch"""
        self.student.train()

        # 更新频域划分
        self.update_frequency_split(dataloader, epoch)

        # 统计
        total_loss = 0.0
        total_ce_loss = 0.0
        total_distill_loss = 0.0
        total_recon_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs}')
        pbar.close()
        print(end='')  # 消除多余换行
        for batch_idx, batch in enumerate(pbar):
            # 解析batch
            if isinstance(batch, dict):
                pressure_mag = batch['pressure'].to(self.device)
                vibration_mag = batch['vibration'].to(self.device)
                labels = batch['label'].to(self.device)
            else:
                pressure_mag, vibration_mag, labels = batch
                pressure_mag = pressure_mag.to(self.device)
                vibration_mag = vibration_mag.to(self.device)
                labels = labels.to(self.device)

            batch_size = pressure_mag.size(0)

            # 教师推理
            with torch.no_grad():
                teacher_global_logits, _ = self.teacher(
                    pressure_mag, vibration_mag, freq_mask=None
                )

                teacher_local_logits = []
                if self.use_sdd and self.current_freq_masks is not None:
                    for mask in self.current_freq_masks:
                        # 找到掩码的有效范围
                        nonzero = torch.nonzero(mask, as_tuple=True)[0]
                        if len(nonzero) == 0:
                            teacher_local_logits.append(teacher_global_logits)  # fallback
                            continue
                        start = nonzero[0].item()
                        end = nonzero[-1].item() + 1

                        # 截取有效频段
                        p_seg = pressure_mag[:, start:end, :]
                        v_seg = vibration_mag[:, start:end, :]

                        # 池化到完整F_bins长度
                        p_seg = F.interpolate(p_seg.permute(0, 2, 1), size=self.student.F_bins,
                                              mode='linear', align_corners=False).permute(0, 2, 1)
                        v_seg = F.interpolate(v_seg.permute(0, 2, 1), size=self.student.F_bins,
                                              mode='linear', align_corners=False).permute(0, 2, 1)

                        # 教师用截取+插值后的数据推理（与学生一致）
                        local_logits, _ = self.teacher(p_seg, v_seg, freq_mask=None)
                        teacher_local_logits.append(local_logits)

            # 学生推理
            student_global_logits, student_local_logits = self.student(
                pressure_mag, vibration_mag,
                freq_masks=self.current_freq_masks if self.use_sdd else None,
                mode='train',
                current_epoch=epoch,
                total_epochs=total_epochs
            )

            # 计算损失
            ce_loss = self.ce_loss(student_global_logits, labels)

            if self.use_sdd and len(teacher_local_logits) > 0:
                distill_loss, loss_global, loss_local = self.sdd_loss(
                    student_global=student_global_logits,
                    student_locals=student_local_logits,
                    teacher_global=teacher_global_logits,
                    teacher_locals=teacher_local_logits
                )
            else:
                distill_loss = F.kl_div(
                    F.log_softmax(student_global_logits / self.sdd_temperature, dim=1),
                    F.softmax(teacher_global_logits / self.sdd_temperature, dim=1),
                    reduction='batchmean'
                ) * (self.sdd_temperature ** 2)

            recon_loss = torch.tensor(0.0, device=self.device)
            if self.recon_weight > 0 and self.student.modality_generator is not None:
                with torch.no_grad():
                    gen_v = self.student.modality_generator(pressure_mag, direction='p2v')
                recon_p = self.student.modality_generator(gen_v, direction='v2p')
                recon_loss += F.mse_loss(recon_p, pressure_mag)

                with torch.no_grad():
                    gen_p = self.student.modality_generator(vibration_mag, direction='v2p')
                recon_v = self.student.modality_generator(gen_p, direction='p2v')
                recon_loss += F.mse_loss(recon_v, vibration_mag)

            #  过渡期蒸馏降权：掩码刚切换时降低蒸馏权重，避免随机头干扰
            effective_distill_weight = self.distill_weight

            loss = (self.ce_weight * ce_loss +
                    effective_distill_weight * distill_loss +
                    self.recon_weight * recon_loss)


            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    max_norm=self.gradient_clip_norm
                )

            optimizer.step()

            # 统计
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_distill_loss += distill_loss.item()
            total_recon_loss += recon_loss.item()

            _, predicted = student_global_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * correct / total:.2f}%',
                'k': self.current_k
            })

        num_batches = len(dataloader)
        metrics = {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'distill_loss': total_distill_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'accuracy': 100.0 * correct / total,
            'k': self.current_k
        }

        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        拆分验证结果：
        1. complete : 双模态 [1,1]
        2. p_only   : 仅压力 [1,0]
        3. v_only   : 仅振动 [0,1]
        4. mixed    : 与训练分布一致的加权结果 0.34/0.33/0.33
        """
        self.student.eval()

        stats = {
            'complete': {'loss': 0.0, 'correct': 0, 'total': 0},
            'p_only': {'loss': 0.0, 'correct': 0, 'total': 0},
            'v_only': {'loss': 0.0, 'correct': 0, 'total': 0},
            'mixed': {'loss': 0.0, 'correct': 0, 'total': 0},
        }

        for batch in tqdm(dataloader, desc='Evaluating'):
            if isinstance(batch, dict):
                pressure_mag = batch['pressure'].to(self.device)
                vibration_mag = batch['vibration'].to(self.device)
                labels = batch['label'].to(self.device)
            else:
                pressure_mag, vibration_mag, labels = batch
                pressure_mag = pressure_mag.to(self.device)
                vibration_mag = vibration_mag.to(self.device)
                labels = labels.to(self.device)

            B = pressure_mag.size(0)

            fixed_mask_complete = torch.tensor([[1, 1]] * B, device=self.device).float()
            fixed_mask_p_only = torch.tensor([[1, 0]] * B, device=self.device).float()
            fixed_mask_v_only = torch.tensor([[0, 1]] * B, device=self.device).float()

            logits_complete, _ = self.student(
                pressure_mag, vibration_mag,
                freq_masks=None,
                mode='test',
                fixed_mask=fixed_mask_complete,
                enable_pvnfm_inference=self.use_pvnfm_inference,
                pvnfm_inference_weight=self.pvnfm_inference_weight
            )

            logits_p_only, _ = self.student(
                pressure_mag, vibration_mag,
                freq_masks=None,
                mode='test',
                fixed_mask=fixed_mask_p_only,
                enable_pvnfm_inference=self.use_pvnfm_inference,
                pvnfm_inference_weight=self.pvnfm_inference_weight
            )

            logits_v_only, _ = self.student(
                pressure_mag, vibration_mag,
                freq_masks=None,
                mode='test',
                fixed_mask=fixed_mask_v_only,
                enable_pvnfm_inference=self.use_pvnfm_inference,
                pvnfm_inference_weight=self.pvnfm_inference_weight
            )

            logits_mixed = (
                    0.34 * logits_complete +
                    0.33 * logits_p_only +
                    0.33 * logits_v_only
            )

            batch_outputs = {
                'complete': logits_complete,
                'p_only': logits_p_only,
                'v_only': logits_v_only,
                'mixed': logits_mixed,
            }

            for key, logits in batch_outputs.items():
                loss = self.ce_loss(logits, labels)
                preds = logits.argmax(dim=1)

                stats[key]['loss'] += loss.item() * labels.size(0)
                stats[key]['correct'] += preds.eq(labels).sum().item()
                stats[key]['total'] += labels.size(0)

        results = {}
        for key, val in stats.items():
            results[key] = {
                'loss': val['loss'] / max(val['total'], 1),
                'accuracy': 100.0 * val['correct'] / max(val['total'], 1)
            }

        return results