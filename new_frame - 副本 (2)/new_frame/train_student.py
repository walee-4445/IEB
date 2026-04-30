"""
IEB-SDD Student Training Script
专门为NFM-IEB-SDD框架设计的学生网络训练脚本
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
import numpy as np
import logging
#from baseline_cmpt import BaselineCMPTModel
# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_loader import get_magnitude_dataloaders
from teacher_model_ieb import IEBTeacherMagnitude
from student_model_magnitude_sdd import StudentModelMagnitudeSDD
from train_ieb_sdd import IEBSDDTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Train Student Model with IEB-SDD Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整框架训练
  python train_student.py --use_ieb --use_sdd --use_pvnfm

  # Baseline训练（无IEB+SDD）
  python train_student.py --ablation_mode baseline

  # 消融实验：无IEB
  python train_student.py --use_sdd --use_pvnfm --ablation_mode no_ieb
        """
    )

    # 数据参数
    parser.add_argument('--data_dir', type=str,
                        default='./new_frame/pressure_vibration1_S2_02',
                        help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # 模型参数
    parser.add_argument('--seq_len', type=int, default=2560,
                        help='序列长度')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='分类类别数')

    # 教师模型
    parser.add_argument('--teacher_path', type=str, default=None,
                        help='预训练教师模型路径')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')

    # IEB-SDD参数（从Config添加）
    Config.add_ieb_sdd_args(parser)

    # PV-NFM参数
    parser.add_argument('--use_pvnfm', action='store_true', default=False,
                        help='启用PV-NFM模态补全')
    parser.add_argument('--pvnfm_context_dim', type=int, default=128,
                        help='PV-NFM上下文维度')
    parser.add_argument('--pvnfm_num_bands', type=int, default=16,
                        help='PV-NFM傅里叶频带数')
    parser.add_argument('--pvnfm_hidden_inr', type=int, default=128,
                        help='PV-NFM INR隐藏层维度')
    parser.add_argument('--test_modalities', action='store_true', default=False,
                        help='自动测试时评估 Complete / P-only / V-only / Mixed')
    # 其他参数
    parser.add_argument('--save_dir', type=str,
                        default='./checkpoints/ieb_sdd',
                        help='模型保存目录')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志打印间隔')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='评估间隔（epoch）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--auto_test', action='store_true', default=False,
                        help='训练完成后自动运行测试')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='测试batch size')


    parser.add_argument('--use_pvnfm_inference', action='store_true', default=False,
                        help='验证/测试时启用PV-NFM补全')
    parser.add_argument('--pvnfm_inference_weight', type=float, default=1.0,
                        help='验证/测试时PV-NFM补全权重')

    parser.add_argument('--magnitude_decay_schedule', type=str,
                        choices=['linear', 'cosine', 'piecewise'],
                        default='piecewise',
                        help='NFM权重衰减策略')
    parser.add_argument('--magnitude_nfm_initial_weight', type=float, default=0.3,
                        help='频域学生模型的NFM初始权重')
    parser.add_argument('--magnitude_nfm_final_weight', type=float, default=0.1,
                        help='频域学生模型的NFM最终权重，建议不要设为0')
    parser.add_argument('--magnitude_nfm_warmup_ratio', type=float, default=0.6,
                        help='NFM warmup阶段比例')
    parser.add_argument('--magnitude_nfm_decay_ratio', type=float, default=0.2,
                        help='NFM decay阶段比例')


    return parser.parse_args()
def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def apply_ablation_config(args, config):
    """
    应用消融实验配置

    Args:
        args: 命令行参数
        config: Config对象
    """
    if args.ablation_mode:
        ablation_config = config.get_ablation_config(args.ablation_mode)

        # 更新args
        args.use_ieb = ablation_config['use_ieb']
        args.use_sdd = ablation_config['use_sdd']
        args.use_pvnfm = ablation_config['use_pvnfm']

        logger.info(f"=" * 80)
        logger.info(f"消融实验模式: {args.ablation_mode}")
        logger.info(f"  - IEB: {args.use_ieb}")
        logger.info(f"  - SDD: {args.use_sdd}")
        logger.info(f"  - PV-NFM: {args.use_pvnfm}")
        logger.info(f"=" * 80)
def train(args):
    """主训练函数"""
    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 应用消融实验配置
    config = Config()
    apply_ablation_config(args, config)

    # 保存配置
    config_dict = vars(args).copy()
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)

    # 创建数据加载器（幅度谱模式）
    logger.info(f"加载数据: {args.data_dir}")
    temp_config = Config()
    temp_config.data_root = args.data_dir
    temp_config.window_size = args.seq_len
    temp_config.num_workers = args.num_workers
    temp_config.device = args.device
    temp_config.seed = args.seed

    # train_loader, val_loader, test_loader, label_encoder = get_magnitude_dataloaders(
    #     config=temp_config,
    #     batch_size=args.batch_size,
    #     mode='student'
    # )
    from data_loader import get_dataloaders_presplit
    train_loader, val_loader, test_loader, label_encoder = get_dataloaders_presplit(
        config=temp_config,
        batch_size=args.batch_size,
        mode='student',
        use_fft=True,  # 启用FFT转换为幅度谱
        seed=args.seed
    )

    logger.info(f"✓ 数据加载完成")
    logger.info(f"  训练集: {len(train_loader)} batches")
    logger.info(f"  验证集: {len(val_loader)} batches")
    logger.info(f"  测试集: {len(test_loader)} batches")

    # 计算频域维度
    F_bins = args.seq_len // 2 + 1

    # ============================================================
    # 创建IEB教师网络
    # ============================================================
    logger.info("=" * 80)
    logger.info("创建IEB教师网络")
    logger.info("=" * 80)

    teacher = IEBTeacherMagnitude(
        F_bins=F_bins,
        num_classes=args.num_classes,
        d_model=config.ieb_teacher_hidden_dim,  # 使用d_model参数
        patch_len=16,  #  添加patch_len
        stride=8,  #  添加stride
        dropout=config.ieb_teacher_dropout,
        llm_model=None,  #  使用默认LLM
        freeze_llm=True  #  冻结LLM参数
    )

    # 加载预训练教师权重
    if args.teacher_path and os.path.exists(args.teacher_path):
        teacher_checkpoint = torch.load(
            args.teacher_path,
            map_location=device,
            weights_only=False  # 明确表示：我知道这个行为，我信任这个文件
        )
        teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
        logger.info(f"✓ 加载预训练教师模型: {args.teacher_path}")
    else:
        logger.warning("⚠ 未提供教师模型，将使用随机初始化的教师网络")

    teacher_params = sum(p.numel() for p in teacher.parameters())
    logger.info(f"✓ 教师参数量: {teacher_params:,}")

    # ============================================================
    # 创建SDD学生网络
    # ============================================================
    logger.info("=" * 80)
    logger.info("创建SDD学生网络")
    logger.info("=" * 80)

    student = StudentModelMagnitudeSDD(
        F_bins=F_bins,
        num_classes=args.num_classes,
        encoder_channels=config.ieb_student_encoder_channels,
        latent_dim=config.ieb_student_latent_dim,
        num_latents=config.ieb_student_num_latents,
        perceiver_depth=config.ieb_student_perceiver_depth,
        perceiver_heads=config.ieb_student_perceiver_heads,
        max_k=config.ieb_student_max_k,
        use_pvnfm=args.use_pvnfm,
        decay_schedule=args.magnitude_decay_schedule,
        nfm_initial_weight=args.magnitude_nfm_initial_weight,
        nfm_final_weight=args.magnitude_nfm_final_weight,
        nfm_warmup_ratio=args.magnitude_nfm_warmup_ratio,
        nfm_decay_ratio=args.magnitude_nfm_decay_ratio,
        pvnfm_context_dim=args.pvnfm_context_dim,
        pvnfm_num_bands=args.pvnfm_num_bands,
        pvnfm_hidden_inr=args.pvnfm_hidden_inr
    )

    student_params = sum(p.numel() for p in student.parameters())
    logger.info(f"✓ 学生参数量: {student_params:,}")
    logger.info(f"✓ 压缩比: {teacher_params / student_params:.2f}x")

    # ============================================================
    # 创建IEB-SDD训练器
    # ============================================================
    logger.info("=" * 80)
    logger.info("创建IEB-SDD训练器")
    logger.info("=" * 80)

    trainer = IEBSDDTrainer(
        teacher_model=teacher,
        student_model=student,
        device=device,
        # IEB参数
        use_ieb=args.use_ieb,
        ieb_start_epoch=args.ieb_start_epoch,
        ieb_update_interval=args.ieb_update_interval,
        min_k=args.ieb_min_k,
        max_k=args.ieb_max_k,
        gvf_threshold=args.ieb_gvf_threshold,
        # SDD参数
        use_sdd=args.use_sdd,
        sdd_alpha=args.sdd_alpha,
        sdd_temperature=args.sdd_temperature,
        local_weight_strategy=args.sdd_local_weight_strategy,
        # 损失权重
        ce_weight=args.ieb_sdd_ce_weight,
        distill_weight=args.ieb_sdd_distill_weight,
        recon_weight=args.ieb_sdd_recon_weight,
        # 梯度控制
        gradient_clip_norm=args.ieb_sdd_gradient_clip_norm,
        log_interval=args.log_interval,use_pvnfm_inference=args.use_pvnfm_inference,
pvnfm_inference_weight=args.pvnfm_inference_weight,
    )

    logger.info(f"✓ 训练器配置:")
    logger.info(f"  - IEB: {args.use_ieb}")
    logger.info(f"  - SDD: {args.use_sdd}")
    logger.info(f"  - PV-NFM: {args.use_pvnfm}")
    logger.info(f"  - 梯度裁剪: {args.ieb_sdd_gradient_clip_norm}")

    # ============================================================
    # 创建优化器和调度器
    # ============================================================
    optimizer = optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # ============================================================
    # 训练循环
    # ============================================================
    logger.info("=" * 80)
    logger.info(f"开始训练 (共{args.epochs}个epoch)")
    logger.info("=" * 80)

    best_acc = 0.0
    train_history = []
    val_history = []

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = trainer.train_epoch(
            train_loader, optimizer, epoch, args.epochs, logger
        )
        train_history.append(train_metrics)

        # 打印训练信息
        logger.info(
            f"Epoch [{epoch}/{args.epochs}] Train - "
            f"Loss: {train_metrics['loss']:.4f} "
            f"(CE: {train_metrics['ce_loss']:.4f}, "
            f"Distill: {train_metrics['distill_loss']:.4f}, "
            f"Recon: {train_metrics['recon_loss']:.4f}) "
            f"Acc: {train_metrics['accuracy']:.2f}% "
            f"k={train_metrics['k']}"
        )

        # 评估
        if epoch % args.eval_interval == 0:
            # val_metrics = trainer.evaluate(val_loader)
            # val_history.append(val_metrics)
            #
            # logger.info(
            #     f"Epoch [{epoch}/{args.epochs}] Val - "
            #     f"Loss: {val_metrics['loss']:.4f}, "
            #     f"Acc: {val_metrics['accuracy']:.2f}%"
            # )
            val_metrics = trainer.evaluate(val_loader)

            logger.info(
                f"Epoch [{epoch}/{args.epochs}] Val Acc | "
                f"Complete: {val_metrics['complete']['accuracy']:.2f}% | "
                f"P-only: {val_metrics['p_only']['accuracy']:.2f}% | "
                f"V-only: {val_metrics['v_only']['accuracy']:.2f}% | "
                f"Mixed: {val_metrics['mixed']['accuracy']:.2f}%"

            )

            logger.info(
                f"Epoch [{epoch}/{args.epochs}] Val Loss | "
                f"Complete: {val_metrics['complete']['loss']:.4f} | "
                f"P-only: {val_metrics['p_only']['loss']:.4f} | "
                f"V-only: {val_metrics['v_only']['loss']:.4f} | "
                f"Mixed: {val_metrics['mixed']['loss']:.4f}"
            )

            current_score = val_metrics['mixed']['accuracy']
            # 保存最佳模型
            # if val_metrics['accuracy'] > best_acc:
            #     best_acc = val_metrics['accuracy']
            if current_score > best_acc:
                best_acc = current_score
                # 保存 best_model.pt
                best_model_path = os.path.join(args.save_dir, 'best_model.pt')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,


                   # # 'encoder_channels': args.encoder_channels,  #  添加这个
                   #  'encoder_channels': config.ieb_student_encoder_channels,
                   #  'config': config_dict,

                    'config': {
                        'seq_len': args.seq_len,
                        'num_classes': args.num_classes,
                        'encoder_channels': config.ieb_student_encoder_channels,
                        'latent_dim': config.ieb_student_latent_dim,
                        'num_latents': config.ieb_student_num_latents,
                        'perceiver_depth': config.ieb_student_perceiver_depth,
                        'perceiver_heads': config.ieb_student_perceiver_heads,
                        'max_k': config.ieb_student_max_k,
                        'use_pvnfm': args.use_pvnfm,
                        'pvnfm_context_dim': args.pvnfm_context_dim,
                        'pvnfm_num_bands': args.pvnfm_num_bands,
                        'pvnfm_hidden_inr': args.pvnfm_hidden_inr,
                        'use_dual_perceiver': False,
                        'use_transformer_fusion': False
                    },

                    'freq_masks': trainer.current_freq_masks,
                    'k': trainer.current_k,
                    'mag_mean': train_loader.dataset.mag_mean,
                    'mag_std': train_loader.dataset.mag_std,


                    # # 🆕 添加模型架构参数
                    # 'model_config': {
                    #     'F_bins': F_bins,
                    #     'num_classes': args.num_classes,
                    #     'encoder_channels': config.ieb_student_encoder_channels,
                    #     'latent_dim': config.ieb_student_latent_dim,
                    #     'num_latents': config.ieb_student_num_latents,
                    #     'perceiver_depth': config.ieb_student_perceiver_depth,
                    #     'perceiver_heads': config.ieb_student_perceiver_heads,
                    #     'max_k': config.ieb_student_max_k,
                    #     'use_pvnfm': args.use_pvnfm,
                    #     'pvnfm_context_dim': args.pvnfm_context_dim,
                    #     'pvnfm_num_bands': args.pvnfm_num_bands,
                    #     'pvnfm_hidden_inr': args.pvnfm_hidden_inr,
                    #     'use_dual_perceiver': False,  # 如果有这个参数
                    #     'use_transformer_fusion': False  # 如果有这个参数
                    # }
                }, best_model_path)

                logger.info(f"✓ 保存最佳模型: {best_acc:.2f}%")
                logger.info("=" * 80)
        # 更新学习率
        scheduler.step()

    # ============================================================
    # 训练完成
    # ============================================================
    logger.info("=" * 80)
    logger.info(f"训练完成！最佳准确率: {best_acc:.2f}%")
    logger.info("=" * 80)

    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_history': train_history,
        'val_history': val_history,
        'best_acc': best_acc,
        'config': config_dict
    }, final_model_path)

    # 保存训练历史
    history = {
        'train': train_history,
        'val': val_history,
        'best_acc': best_acc
    }
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    logger.info(f"✓ 模型和历史已保存到: {args.save_dir}")
    # 🆕 自动测试
    if args.auto_test:
        logger.info("=" * 80)
        logger.info("开始自动测试...")
        logger.info("=" * 80)

        import subprocess

        # 构建evaluate.py命令
        eval_cmd = [
            sys.executable,  # Python解释器
            'evaluate.py',  # 评估脚本路径
            '--model_type', 'student',
            '--model_path', os.path.join(args.save_dir, 'best_model.pt'),
            '--data_dir', args.data_dir,
            '--batch_size', str(args.test_batch_size),
            '--device', args.device,
            '--seq_len', str(args.seq_len),
            '--num_classes', str(args.num_classes)
        ]
        if args.test_modalities:
            eval_cmd.append('--test_modalities')

        if args.use_pvnfm_inference:
            eval_cmd.append('--use_pvnfm_inference')

        eval_cmd.extend([
            '--pvnfm_inference_weight', str(args.pvnfm_inference_weight)
        ])
        logger.info(f"运行命令: {' '.join(eval_cmd)}")

        try:
            # 运行evaluate.py
            result = subprocess.run(
                eval_cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # 打印输出
            logger.info("测试输出:")
            logger.info(result.stdout)

            if result.stderr:
                logger.warning("测试警告/错误:")
                logger.warning(result.stderr)

            logger.info("✓ 自动测试完成")

        except subprocess.CalledProcessError as e:
            logger.error("✗ 自动测试失败")
            logger.error(f"返回码: {e.returncode}")
            logger.error(f"输出: {e.stdout}")
            logger.error(f"错误: {e.stderr}")
        except Exception as e:
            logger.error(f"✗ 运行自动测试失败: {e}")


if __name__ == '__main__':
    args = parse_args()
    train(args)

# """
# IEB Teacher Model (FD-MVLLM Architecture)
# 基于FD-MVLLM的IEB教师网络 - 使用冻结的LLM编码器
# """
#
# import torch
# import torch.nn as nn
# from torch import Tensor
# from typing import Optional, Tuple, Any
# import inspect  # 🔧 修复：用于检查函数签名
#
#
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
#
#
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
#
#
# class PatchEmbedding(nn.Module):
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
#
#
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
#
#
# def generate_jenks_freq_masks(F_bins: int, jenks_cuts: list, device='cuda') -> list:
#     """
#     将Jenks切分点转换为频域掩码
#
#     Args:
#         F_bins: 频域维度（1281）
#         jenks_cuts: Jenks切分点列表，如 [0, 300, 600, 900, 1281]
#         device: 设备
#
#     Returns:
#         freq_masks: List[Tensor(F,)] k个频域掩码
#     """
#     masks = []
#
#     for i in range(len(jenks_cuts) - 1):
#         start_freq = jenks_cuts[i]
#         end_freq = jenks_cuts[i + 1]
#
#         # 创建频域掩码
#         mask = torch.zeros(F_bins, device=device)
#         mask[start_freq:end_freq] = 1.0
#
#         masks.append(mask)
#
#     return masks
#
#
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
#
# # # """
# # # Training script for Student Model
# # # 支持知识蒸馏、模态缺失训练、NFM渐进式权重衰减、梯度平衡、自动测试等完整功能
# # # """
# # #
# # # import torch
# # # import torch.optim as optim
# # # from torch.utils.data import DataLoader
# # # import argparse
# # # import os
# # # import sys
# # # import json
# # # import numpy as np
# # # import logging
# # # import subprocess
# # #
# # # # 添加父目录到路径
# # # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # #
# # # from Student_Model.student_model import StudentModel, StudentTrainer
# # # from new_frame.student_model_magnitude import StudentModelMagnitude
# # # from Teacher_Model.teacher_model import TeacherModel
# # # from modules.gmd_optimizer import GMDOptimizer
# # #
# # # # 设置日志
# # # logging.basicConfig(
# # #     level=logging.INFO,
# # #     format='%(asctime)s - %(levelname)s - %(message)s'
# # # )
# # # logger = logging.getLogger(__name__)
# # #
# # #
# # # def parse_args():
# # #     """解析命令行参数"""
# # #     parser = argparse.ArgumentParser(description='Train Student Model with Full Features')
# # #
# # #     # 数据参数
# # #     parser.add_argument('--data_dir', type=str, default='./datapre/pressure_vibration_S2_02',
# # #                        help='数据目录（优先使用真实数据）')
# # #     parser.add_argument('--batch_size', type=int, default=32,
# # #                        help='batch size')
# # #     parser.add_argument('--num_workers', type=int, default=4,
# # #                        help='数据加载线程数')
# # #
# # #     # 模型参数
# # #     parser.add_argument('--seq_len', type=int, default=2560,
# # #                        help='序列长度')
# # #     parser.add_argument('--num_classes', type=int, default=10,
# # #                        help='分类类别数')
# # #     parser.add_argument('--latent_dim', type=int, default=128,
# # #                        help='潜在特征维度')
# # #     parser.add_argument('--num_latents', type=int, default=32,
# # #                        help='Perceiver latent数量')
# # #     parser.add_argument('--use_nfm', action='store_true', default=True,
# # #                        help='是否使用NFM生成器')
# # #     parser.add_argument('--no_nfm', action='store_true', default=False,
# # #                        help='禁用NFM生成器（优先级高于--use_nfm）')
# # #
# # #     # 🆕 PV-NFM参数
# # #     parser.add_argument('--use_pvnfm', action='store_true', default=False,
# # #                        help='启用PV-NFM模态补全（可独立使用或与--use_magnitude_only组合）')
# # #     parser.add_argument('--no_pvnfm', action='store_true', default=False,
# # #                        help='禁用PV-NFM（优先级高于--use_pvnfm）')
# # #     parser.add_argument('--pvnfm_context_dim', type=int, default=None,
# # #                        help='PV-NFM上下文维度（None=使用config.py配置）')
# # #     parser.add_argument('--pvnfm_num_bands', type=int, default=None,
# # #                        help='PV-NFM傅里叶频带数（None=使用config.py配置）')
# # #     parser.add_argument('--pvnfm_hidden_inr', type=int, default=None,
# # #                        help='PV-NFM INR隐藏层维度（None=使用config.py配置）')
# # #
# # #     # 🆕 纯频域架构参数
# # #     parser.add_argument('--use_magnitude_only', action='store_true', default=False,
# # #                        help='使用纯频域架构（仅幅度谱）')
# # #
# # #     # 🆕 Magnitude模型专用参数（用于student_model_magnitude.py）
# # #     parser.add_argument('--magnitude_F_bins', type=int, default=None,
# # #                        help='频域维度（频点数），默认: seq_len//2+1')
# # #     parser.add_argument('--magnitude_num_classes', type=int, default=None,
# # #                        help='分类类别数（None=使用--num_classes）')
# # #     parser.add_argument('--magnitude_encoder_channels', type=int, nargs='+', default=None,
# # #                        help='CNN编码器通道数列表，例如: --magnitude_encoder_channels 32 ')
# # #     parser.add_argument('--magnitude_latent_dim', type=int, default=None,
# # #                        help='潜在特征维度（None=使用--latent_dim）')
# # #     parser.add_argument('--magnitude_num_latents', type=int, default=None,
# # #                        help='Perceiver的latent数量（None=使用--num_latents）')
# # #     parser.add_argument('--magnitude_perceiver_depth', type=int, default=None,
# # #                        help='Perceiver深度')
# # #     parser.add_argument('--magnitude_perceiver_heads', type=int, default=None,
# # #                        help='Perceiver注意力头数')
# # #     parser.add_argument('--magnitude_use_pvnfm', action='store_true', default=None,
# # #                        help='是否使用PV-NFM（None=使用--use_pvnfm）')
# # #     parser.add_argument('--magnitude_use_dual_perceiver', action='store_true', default=None,
# # #                        help='是否使用双Perceiver架构')
# # #     parser.add_argument('--magnitude_use_transformer_fusion', action='store_true', default=None,
# # #                        help='是否使用Transformer融合')
# # #     parser.add_argument('--magnitude_decay_schedule', type=str, default=None,
# # #                        choices=['linear', 'cosine', 'piecewise'],
# # #                        help='NFM权重衰减策略（None=使用--nfm_decay_strategy或config.py）')
# # #     parser.add_argument('--magnitude_nfm_initial_weight', type=float, default=None,
# # #                        help='NFM初始权重（None=使用--nfm_initial_weight或config.py）')
# # #     parser.add_argument('--magnitude_nfm_final_weight', type=float, default=None,
# # #                        help='NFM最终权重（None=使用--nfm_final_weight或config.py）')
# # #     parser.add_argument('--magnitude_nfm_warmup_ratio', type=float, default=None,
# # #                        help='NFM warmup阶段比例（None=使用--nfm_warmup_ratio或config.py）')
# # #     parser.add_argument('--magnitude_nfm_decay_ratio', type=float, default=None,
# # #                        help='NFM decay阶段比例（None=使用--nfm_decay_ratio或config.py）')
# # #     parser.add_argument('--magnitude_pvnfm_context_dim', type=int, default=None,
# # #                        help='PV-NFM上下文维度（None=使用--pvnfm_context_dim或config.py）')
# # #     parser.add_argument('--magnitude_pvnfm_num_bands', type=int, default=None,
# # #                        help='PV-NFM傅里叶频带数（None=使用--pvnfm_num_bands或config.py）')
# # #     parser.add_argument('--magnitude_pvnfm_hidden_inr', type=int, default=None,
# # #                        help='PV-NFM INR隐藏层维度（None=使用--pvnfm_hidden_inr或config.py）')
# # #
# # #     # 🆕 NFM渐进式权重衰减参数（命令行参数可覆盖config.py配置）
# # #     parser.add_argument('--nfm_decay_strategy', type=str, default=None,
# # #                        choices=['linear', 'cosine', 'piecewise', None],
# # #                        help='NFM权重衰减策略（None=使用config.py配置）')
# # #     parser.add_argument('--nfm_initial_weight', type=float, default=None,
# # #                        help='NFM初始权重（None=使用config.py配置）')
# # #     parser.add_argument('--nfm_final_weight', type=float, default=None,
# # #                        help='NFM最终权重（None=使用config.py配置）')
# # #     parser.add_argument('--nfm_warmup_ratio', type=float, default=None,
# # #                        help='NFM warmup阶段比例（None=使用config.py配置）')
# # #     parser.add_argument('--nfm_decay_ratio', type=float, default=None,
# # #                        help='NFM decay阶段比例（None=使用config.py配置）')
# # #
# # #     # 训练参数
# # #     parser.add_argument('--epochs', type=int, default=100,
# # #                        help='训练轮数')
# # #     parser.add_argument('--lr', type=float, default=1e-3,
# # #                        help='学习率')
# # #     parser.add_argument('--weight_decay', type=float, default=1e-4,
# # #                        help='权重衰减')
# # #     parser.add_argument('--warmup_epochs', type=int, default=5,
# # #                        help='warmup轮数')
# # #
# # #     # 知识蒸馏参数
# # #     parser.add_argument('--use_distillation', action='store_true', default=True,
# # #                        help='是否使用知识蒸馏')
# # #     parser.add_argument('--no_hmi', action='store_true', default=False,
# # #                        help='禁用HMI蒸馏（优先级高于--use_distillation）')
# # #     parser.add_argument('--teacher_path', type=str, default=None,
# # #                        help='教师模型路径')
# # #     parser.add_argument('--hmi_weight', type=float, default=0.7,
# # #                        help='HMI损失权重')
# # #     parser.add_argument('--ce_weight', type=float, default=1.0,
# # #                        help='交叉熵损失权重')
# # #     parser.add_argument('--recon_loss_weight', type=float, default=0.1,  # 🆕 添加这一行
# # #                         help='重建损失权重（用于PV-NFM训练）')
# # #
# # #
# # #     # 🆕 HMI损失组件权重
# # #     parser.add_argument('--hmi_mine_weight', type=float, default=None,
# # #                        help='HMI-MINE损失权重（覆盖config.py）')
# # #     parser.add_argument('--hmi_hsic_weight', type=float, default=None,
# # #                        help='HMI-HSIC损失权重（覆盖config.py）')
# # #     parser.add_argument('--hmi_kl_weight', type=float, default=None,
# # #                        help='HMI-KL损失权重（覆盖config.py）')
# # #
# # #     # GMD优化器参数
# # #     parser.add_argument('--use_gmd', action='store_true', default=False,
# # #                        help='是否使用GMD优化器')
# # #     parser.add_argument('--gmd_alpha', type=float, default=0.5,
# # #                        help='GMD alpha参数')
# # #
# # #     # 🆕 预训练模型加载
# # #     parser.add_argument('--resume_from', type=str, default=None,
# # #                        help='从检查点恢复训练（完整checkpoint路径）')
# # #     parser.add_argument('--nfm_pretrained_path', type=str, default=None,
# # #                        help='预训练NFM生成器路径')
# # #
# # #     # 🆕 混合模态评估参数
# # #     parser.add_argument('--mixed_modality_eval', action='store_true', default=True,
# # #                        help='使用混合模态分布进行验证')
# # #     parser.add_argument('--mixed_p_ratio', type=float, default=0.33,
# # #                        help='混合模态中仅压力的比例')
# # #     parser.add_argument('--mixed_v_ratio', type=float, default=0.33,
# # #                        help='混合模态中仅振动的比例')
# # #     parser.add_argument('--mixed_pv_ratio', type=float, default=0.34,
# # #                        help='混合模态中双模态的比例')
# # #
# # #     # 🆕 梯度平衡器参数（命令行参数优先级高于config.py）
# # #     parser.add_argument('--use_gradient_balancing', action='store_true', default=False,
# # #                        help='启用梯度平衡（覆盖config.py配置）')
# # #     parser.add_argument('--no_gradient_balancing', action='store_true', default=False,
# # #                        help='禁用梯度平衡（覆盖config.py配置）')
# # #     parser.add_argument('--gradient_balance_alpha', type=float, default=None,
# # #                        help='梯度平衡目标权重（覆盖config.py配置，例如：0.4, 0.5, 0.6）')
# # #     parser.add_argument('--gradient_balance_momentum', type=float, default=None,
# # #                        help='梯度平衡移动平均系数（覆盖config.py配置，例如：0.9, 0.95）')
# # #     parser.add_argument('--gradient_balance_log_interval', type=int, default=None,
# # #                        help='梯度统计日志打印间隔（覆盖config.py配置，例如：10, 20）')
# # #
# # #     # 🆕 自动测试参数
# # #     parser.add_argument('--auto_test', action='store_true', default=False,
# # #                        help='训练完成后自动运行evaluate.py')
# # #     parser.add_argument('--test_data_dir', type=str, default=None,
# # #                        help='测试数据目录（默认使用训练数据目录）')
# # #     parser.add_argument('--test_batch_size', type=int, default=64,
# # #                        help='测试batch size')
# # #
# # #     # 🆕 双分支相关
# # #     parser.add_argument('--use_dual_branch_pvnfm', action='store_true', default=None,
# # #                         help='使用双分支PVNFM（None=使用config.py配置）')
# # #     parser.add_argument('--cycle_loss_weight', type=float, default=None,
# # #                         help='循环一致性损失权重（None=使用config.py配置）')
# # #
# # #
# # #     # 其他参数
# # #     parser.add_argument('--save_dir', type=str, default='./checkpoints/student',
# # #                        help='模型保存目录')
# # #     parser.add_argument('--log_interval', type=int, default=10,
# # #                        help='日志打印间隔')
# # #     parser.add_argument('--eval_interval', type=int, default=1,
# # #                        help='评估间隔（epoch）')
# # #     parser.add_argument('--save_interval', type=int, default=10,
# # #                        help='保存间隔（epoch）')
# # #     parser.add_argument('--device', type=str, default='cuda',
# # #                        help='设备')
# # #     parser.add_argument('--seed', type=int, default=42,
# # #                        help='随机种子')
# # #
# # #     return parser.parse_args()
# # #     # 在parse_args()函数末尾，return之前添加：
# # #
# # #     # 🆕 IEB-SDD参数
# # #     from new_frame.config import Config
# # #     Config.add_ieb_sdd_args(parser)
# # #
# # #     return parser.parse_args()
# # # def set_seed(seed):
# # #     """设置随机种子"""
# # #     torch.manual_seed(seed)
# # #     torch.cuda.manual_seed_all(seed)
# # #     np.random.seed(seed)
# # #     torch.backends.cudnn.deterministic = True
# # #     torch.backends.cudnn.benchmark = False
# # # def load_teacher_model(args, device):
# # #     """加载教师模型"""
# # #     if not args.use_distillation or args.teacher_path is None:
# # #         return None
# # #
# # #     logger.info(f"Loading teacher model from {args.teacher_path}")
# # #
# # #     teacher = TeacherModel(
# # #         seq_len=args.seq_len,
# # #         num_classes=args.num_classes,
# # #         hidden_dim=args.latent_dim  # TeacherModel使用hidden_dim参数
# # #     )
# # #
# # #     checkpoint = torch.load(args.teacher_path, map_location=device, weights_only=False)
# # #     teacher.load_state_dict(checkpoint['model_state_dict'])
# # #     teacher.to(device)
# # #     teacher.eval()
# # #
# # #     logger.info("Teacher model loaded successfully")
# # #     return teacher
# # # def create_dataloaders(args):
# # #     """
# # #     创建数据加载器
# # #     优先使用真实数据，失败时回退到模拟数据
# # #     支持时域和频域（幅度谱）两种模式
# # #     """
# # #     # 尝试加载真实数据
# # #     try:
# # #         # 🆕 根据use_magnitude_only选择数据加载器
# # #         if args.use_magnitude_only:
# # #             from new_frame.data_loader import get_magnitude_dataloaders as get_dataloaders
# # #             logger.info("使用纯频域数据加载器（仅幅度谱）")
# # #         else:
# # #             from new_frame.data_loader import get_dataloaders
# # #             logger.info("使用时域数据加载器")
# # #
# # #         import sys
# # #         sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # #         from new_frame.config import Config
# # #
# # #         logger.info(f"Attempting to load real data from {args.data_dir}")
# # #
# # #         # 创建临时config对象
# # #         temp_config = Config()
# # #         temp_config.data_root = args.data_dir
# # #         temp_config.window_size = args.seq_len
# # #         temp_config.num_workers = args.num_workers
# # #         temp_config.device = args.device
# # #         temp_config.seed = args.seed
# # #
# # #         # 使用get_dataloaders加载数据
# # #         train_loader, val_loader, test_loader, label_encoder = get_dataloaders(
# # #             config=temp_config,
# # #             batch_size=args.batch_size,
# # #             mode='student'
# # #         )
# # #
# # #
# # #         logger.info(f"✓ Real data loaded successfully")
# # #         logger.info(f"  Train batches: {len(train_loader)}")
# # #         logger.info(f"  Val batches: {len(val_loader)}")
# # #         logger.info(f"  Test batches: {len(test_loader)}")
# # #         logger.info(f"  Number of classes: {len(label_encoder.classes_)}")
# # #
# # #         return train_loader, val_loader, test_loader
# # #
# # #     except Exception as e:
# # #         logger.warning(f"Failed to load real data: {e}")
# # #         logger.warning(f"Error details: {type(e).__name__}: {str(e)}")
# # #         logger.warning("Falling back to dummy data for testing...")
# # #
# # #         # 创建模拟数据
# # #         from torch.utils.data import TensorDataset
# # #
# # #         # 训练集
# # #         train_pressure = torch.randn(1000, args.seq_len, 1)
# # #         train_vibration = torch.randn(1000, args.seq_len, 1)
# # #         train_labels = torch.randint(0, args.num_classes, (1000,))
# # #         train_dataset = TensorDataset(train_pressure, train_vibration, train_labels)
# # #         train_loader = DataLoader(
# # #             train_dataset,
# # #             batch_size=args.batch_size,
# # #             shuffle=True,
# # #             num_workers=0,  # 模拟数据不需要多线程
# # #             pin_memory=True
# # #         )
# # #
# # #         # 验证集
# # #         val_pressure = torch.randn(200, args.seq_len, 1)
# # #         val_vibration = torch.randn(200, args.seq_len, 1)
# # #         val_labels = torch.randint(0, args.num_classes, (200,))
# # #         val_dataset = TensorDataset(val_pressure, val_vibration, val_labels)
# # #         val_loader = DataLoader(
# # #             val_dataset,
# # #             batch_size=args.batch_size,
# # #             shuffle=False,
# # #             num_workers=0,
# # #             pin_memory=True
# # #         )
# # #
# # #         # 测试集
# # #         test_pressure = torch.randn(200, args.seq_len, 1)
# # #         test_vibration = torch.randn(200, args.seq_len, 1)
# # #         test_labels = torch.randint(0, args.num_classes, (200,))
# # #         test_dataset = TensorDataset(test_pressure, test_vibration, test_labels)
# # #         test_loader = DataLoader(
# # #             test_dataset,
# # #             batch_size=args.batch_size,
# # #             shuffle=False,
# # #             num_workers=0,
# # #             pin_memory=True
# # #         )
# # #
# # #         logger.info(f"✓ Dummy data created")
# # #         logger.info(f"  Train batches: {len(train_loader)}")
# # #         logger.info(f"  Val batches: {len(val_loader)}")
# # #         logger.info(f"  Test batches: {len(test_loader)}")
# # #
# # #         return train_loader, val_loader, test_loader
# # def run_auto_test(args, best_model_path):
# #     """
# #     🆕 训练完成后自动运行evaluate.py
# #     """
# #     logger.info("=" * 80)
# #     logger.info("Starting automatic evaluation...")
# #     logger.info("=" * 80)
# #
# #     test_data_dir = args.test_data_dir if args.test_data_dir else args.data_dir
# #
# #     # 构建evaluate.py命令
# #     eval_cmd = [
# #         sys.executable,  # Python解释器
# #         'evaluate.py',
# #         '--model_type', 'student',  # 🆕 指定模型类型
# #         '--model_path', best_model_path,
# #         '--data_dir', test_data_dir,
# #         '--batch_size', str(args.test_batch_size),
# #         '--device', args.device,
# #         '--test_modalities'  # 🆕 测试不同模态组合
# #     ]
# #
# #     logger.info(f"Running command: {' '.join(eval_cmd)}")
# #
# #     try:
# #         # 运行evaluate.py
# #         result = subprocess.run(
# #             eval_cmd,
# #             capture_output=True,
# #             text=True,
# #             check=True
# #         )
# #
# #         # 打印输出
# #         logger.info("Evaluation output:")
# #         logger.info(result.stdout)
# #
# #         if result.stderr:
# #             logger.warning("Evaluation warnings/errors:")
# #             logger.warning(result.stderr)
# #
# #         logger.info("✓ Automatic evaluation completed successfully")
# #
# #     except subprocess.CalledProcessError as e:
# #         logger.error("✗ Automatic evaluation failed")
# #         logger.error(f"Return code: {e.returncode}")
# #         logger.error(f"Output: {e.stdout}")
# #         logger.error(f"Error: {e.stderr}")
# #     except Exception as e:
# #         logger.error(f"✗ Failed to run automatic evaluation: {e}")
# # def load_pretrained_models(student, args, device):
# #     """
# #     🆕 加载预训练模型
# #     支持：
# #     1. resume_from: 完整checkpoint恢复训练
# #     2. nfm_pretrained_path: 仅加载预训练NFM生成器
# #     """
# #     start_epoch = 1
# #     best_acc = 0.0
# #     optimizer_state = None
# #     scheduler_state = None
# #
# #     # 1. 从完整checkpoint恢复
# #     if args.resume_from and os.path.exists(args.resume_from):
# #         logger.info(f"Resuming from checkpoint: {args.resume_from}")
# #         checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
# #
# #         student.load_state_dict(checkpoint['model_state_dict'])
# #         start_epoch = checkpoint.get('epoch', 0) + 1
# #         best_acc = checkpoint.get('best_acc', 0.0)
# #         optimizer_state = checkpoint.get('optimizer_state_dict', None)
# #         scheduler_state = checkpoint.get('scheduler_state_dict', None)
# #
# #         logger.info(f"✓ Resumed from epoch {start_epoch-1}, best_acc={best_acc:.2f}%")
# #
# #     # 2. 加载预训练NFM生成器
# #     elif args.nfm_pretrained_path and os.path.exists(args.nfm_pretrained_path):
# #         logger.info(f"Loading pretrained NFM generator: {args.nfm_pretrained_path}")
# #
# #         try:
# #             nfm_checkpoint = torch.load(args.nfm_pretrained_path, map_location=device, weights_only=False)
# #
# #             # 提取NFM相关参数
# #             nfm_state_dict = {}
# #             for key, value in nfm_checkpoint['model_state_dict'].items():
# #                 if 'nfm_generator' in key:
# #                     nfm_state_dict[key] = value
# #
# #             # 加载到student模型
# #             student.load_state_dict(nfm_state_dict, strict=False)
# #             logger.info(f"✓ Loaded {len(nfm_state_dict)} NFM parameters")
# #
# #         except Exception as e:
# #             logger.warning(f"Failed to load pretrained NFM: {e}")
# #
# #     return start_epoch, best_acc, optimizer_state, scheduler_state
# # def train(args):
# #     """主训练函数 - 完整功能版本"""
# #     # 🆕 处理 --no_nfm 和 --no_hmi 参数（优先级最高）
# #     if args.no_nfm:
# #         args.use_nfm = False
# #         logger.info("NFM disabled via --no_nfm flag")
# #
# #     if args.no_hmi:
# #         args.use_distillation = False
# #         logger.info("HMI distillation disabled via --no_hmi flag")
# #
# #     # 🆕 处理 --no_pvnfm 参数（优先级最高）
# #     if args.no_pvnfm:
# #         args.use_pvnfm = False
# #         logger.info("PV-NFM disabled via --no_pvnfm flag")
# #
# #     # 🆕 独立控制逻辑
# #     # --use_magnitude_only: 控制架构（时域 vs 频域）
# #     # --use_pvnfm: 控制模态补全（开启 vs 关闭）
# #     if args.use_magnitude_only:
# #         logger.info("✓ Frequency-domain architecture (magnitude-only) enabled")
# #     else:
# #         logger.info("✓ Time-domain architecture enabled")
# #
# #     if args.use_pvnfm:
# #         logger.info("✓ PV-NFM modality completion enabled")
# #     else:
# #         logger.info("✓ PV-NFM modality completion disabled")
# #
# #     # 设置随机种子
# #     set_seed(args.seed)
# #
# #     # 设置设备
# #     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
# #     logger.info(f"Using device: {device}")
# #
# #     # 创建保存目录
# #     os.makedirs(args.save_dir, exist_ok=True)
# #
# #     # 保存配置
# #     config_dict = vars(args).copy()
# #     with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
# #         json.dump(config_dict, f, indent=4)
# #
# #     # 创建数据加载器
# #     train_loader, val_loader, test_loader = create_dataloaders(args)
# #
# #     # 🆕 从config.py加载NFM和PV-NFM参数（命令行参数优先）
# #     from new_frame.config import Config
# #     nfm_decay_strategy = args.nfm_decay_strategy if args.nfm_decay_strategy is not None else Config.nfm_decay_schedule
# #     nfm_initial_weight = args.nfm_initial_weight if args.nfm_initial_weight is not None else Config.nfm_initial_weight
# #     nfm_final_weight = args.nfm_final_weight if args.nfm_final_weight is not None else Config.nfm_final_weight
# #     nfm_warmup_ratio = args.nfm_warmup_ratio if args.nfm_warmup_ratio is not None else Config.nfm_warmup_ratio
# #     nfm_decay_ratio = args.nfm_decay_ratio if args.nfm_decay_ratio is not None else Config.nfm_decay_ratio
# #
# #     # 🆕 PV-NFM参数（命令行优先，但已经在前面验证过组合逻辑）
# #     use_pvnfm = args.use_pvnfm  # 不再从Config读取，完全由命令行控制
# #     use_magnitude_only = args.use_magnitude_only  # 不再从Config读取，完全由命令行控制
# #     pvnfm_context_dim = args.pvnfm_context_dim if args.pvnfm_context_dim is not None else Config.pvnfm_context_dim
# #     pvnfm_num_bands = args.pvnfm_num_bands if args.pvnfm_num_bands is not None else Config.pvnfm_num_bands
# #     pvnfm_hidden_inr = args.pvnfm_hidden_inr if args.pvnfm_hidden_inr is not None else Config.pvnfm_hidden_inr
# #
# #     # 🆕 根据use_magnitude_only选择模型
# #     if use_magnitude_only:
# #         # 使用频域学生模型（纯幅度谱）
# #         # 🆕 使用magnitude专用参数（优先级：magnitude参数 > 通用参数 > config.py）
# #         mag_F_bins = args.magnitude_F_bins if args.magnitude_F_bins is not None else (args.seq_len // 2 + 1)
# #         mag_num_classes = args.magnitude_num_classes if args.magnitude_num_classes is not None else args.num_classes
# #         mag_encoder_channels = args.magnitude_encoder_channels if args.magnitude_encoder_channels is not None else Config.magnitude_encoder_channels
# #         mag_latent_dim = args.magnitude_latent_dim if args.magnitude_latent_dim is not None else args.latent_dim
# #         mag_num_latents = args.magnitude_num_latents if args.magnitude_num_latents is not None else args.num_latents
# #         mag_perceiver_depth = args.magnitude_perceiver_depth if args.magnitude_perceiver_depth is not None else Config.magnitude_perceiver_depth
# #         mag_perceiver_heads = args.magnitude_perceiver_heads if args.magnitude_perceiver_heads is not None else Config.magnitude_perceiver_heads
# #         mag_use_pvnfm = args.magnitude_use_pvnfm if args.magnitude_use_pvnfm is not None else use_pvnfm
# #         mag_use_dual_perceiver = args.magnitude_use_dual_perceiver if args.magnitude_use_dual_perceiver is not None else Config.magnitude_use_dual_perceiver
# #         mag_use_transformer_fusion = args.magnitude_use_transformer_fusion if args.magnitude_use_transformer_fusion is not None else Config.magnitude_use_transformer_fusion
# #         mag_decay_schedule = args.magnitude_decay_schedule if args.magnitude_decay_schedule is not None else nfm_decay_strategy
# #         mag_nfm_initial_weight = args.magnitude_nfm_initial_weight if args.magnitude_nfm_initial_weight is not None else nfm_initial_weight
# #         mag_nfm_final_weight = args.magnitude_nfm_final_weight if args.magnitude_nfm_final_weight is not None else nfm_final_weight
# #         mag_nfm_warmup_ratio = args.magnitude_nfm_warmup_ratio if args.magnitude_nfm_warmup_ratio is not None else nfm_warmup_ratio
# #         mag_nfm_decay_ratio = args.magnitude_nfm_decay_ratio if args.magnitude_nfm_decay_ratio is not None else nfm_decay_ratio
# #         mag_pvnfm_context_dim = args.magnitude_pvnfm_context_dim if args.magnitude_pvnfm_context_dim is not None else pvnfm_context_dim
# #         mag_pvnfm_num_bands = args.magnitude_pvnfm_num_bands if args.magnitude_pvnfm_num_bands is not None else pvnfm_num_bands
# #         mag_pvnfm_hidden_inr = args.magnitude_pvnfm_hidden_inr if args.magnitude_pvnfm_hidden_inr is not None else pvnfm_hidden_inr
# #
# #         logger.info(f"Creating Magnitude-Only Student Model (F_bins={mag_F_bins})")
# #         logger.info(f"  Encoder channels: {mag_encoder_channels}")
# #         logger.info(f"  Latent dim: {mag_latent_dim}")
# #         logger.info(f"  Num latents: {mag_num_latents}")
# #         logger.info(f"  Perceiver depth: {mag_perceiver_depth}")
# #         logger.info(f"  Perceiver heads: {mag_perceiver_heads}")
# #
# #         student = StudentModelMagnitude(
# #             F_bins=mag_F_bins,
# #             num_classes=mag_num_classes,
# #             encoder_channels=mag_encoder_channels,
# #             latent_dim=mag_latent_dim,
# #             num_latents=mag_num_latents,
# #             perceiver_depth=mag_perceiver_depth,
# #             perceiver_heads=mag_perceiver_heads,
# #             use_pvnfm=mag_use_pvnfm,
# #             use_dual_perceiver=mag_use_dual_perceiver,
# #             use_transformer_fusion=mag_use_transformer_fusion,
# #             decay_schedule=mag_decay_schedule,
# #             nfm_initial_weight=mag_nfm_initial_weight,
# #             nfm_final_weight=mag_nfm_final_weight,
# #             nfm_warmup_ratio=mag_nfm_warmup_ratio,
# #             nfm_decay_ratio=mag_nfm_decay_ratio,
# #             pvnfm_context_dim=mag_pvnfm_context_dim,
# #             pvnfm_num_bands=mag_pvnfm_num_bands,
# #             pvnfm_hidden_inr=mag_pvnfm_hidden_inr
# #         )
# #     else:
# #         # 使用时域学生模型
# #         logger.info(f"Creating Time-Domain Student Model (seq_len={args.seq_len})")
# #
# #         student = StudentModel(
# #             seq_len=args.seq_len,
# #             num_classes=args.num_classes,
# #             latent_dim=args.latent_dim,
# #             num_latents=args.num_latents,
# #             use_nfm=args.use_nfm,
# #             # 🆕 PV-NFM参数
# #             use_pvnfm=use_pvnfm,
# #             pvnfm_context_dim=pvnfm_context_dim,
# #             pvnfm_num_bands=pvnfm_num_bands,
# #             pvnfm_hidden_inr=pvnfm_hidden_inr,
# #             # NFM权重衰减参数（传入None则使用config.py配置）
# #             decay_schedule=nfm_decay_strategy,
# #             nfm_initial_weight=nfm_initial_weight,
# #             nfm_final_weight=nfm_final_weight,
# #             nfm_warmup_ratio=nfm_warmup_ratio,
# #             nfm_decay_ratio=nfm_decay_ratio
# #         )
# #     logger.info(f"Student model parameters: {sum(p.numel() for p in student.parameters()):,}")
# #
# #
# #
# #     # 🆕 打印架构和模态补全机制信息
# #     if use_magnitude_only:
# #         logger.info(f"Architecture: Pure Frequency Domain (Magnitude-Only)")
# #         logger.info(f"  - Input: Magnitude Spectrum (F, 1)")
# #         logger.info(f"  - Encoder: Frequency Domain CNN")
# #         logger.info(f"  - PV-NFM: Magnitude-Only Version")
# #     else:
# #         logger.info(f"Architecture: Time Domain")
# #         logger.info(f"  - Input: Time Series (L, 1)")
# #         logger.info(f"  - Encoder: Time Domain CNN")
# #
# #     if use_pvnfm:
# #         logger.info(f"Modality Completion: PV-NFM (Physical Transfer Function)")
# #         logger.info(f"  - Context Dim: {pvnfm_context_dim}")
# #         logger.info(f"  - Num Bands: {pvnfm_num_bands}")
# #         logger.info(f"  - Hidden INR: {pvnfm_hidden_inr}")
# #         if use_magnitude_only:
# #             logger.info(f"  - Mode: Magnitude-Only (No Phase)")
# #         else:
# #             logger.info(f"  - Mode: Magnitude + Phase Learning")
# #     elif args.use_nfm:
# #         logger.info(f"Modality Completion: NFM Generator (Frequency Domain Mapping)")
# #     else:
# #         logger.info(f"Modality Completion: Disabled (Zero Padding Only)")
# #
# #     logger.info(f"NFM Decay Strategy: {nfm_decay_strategy} (from {'CLI' if args.nfm_decay_strategy else 'config.py'})")
# #     logger.info(f"  - Initial Weight: {nfm_initial_weight} (from {'CLI' if args.nfm_initial_weight is not None else 'config.py'})")
# #     logger.info(f"  - Final Weight: {nfm_final_weight} (from {'CLI' if args.nfm_final_weight is not None else 'config.py'})")
# #     if nfm_decay_strategy == 'piecewise':
# #         logger.info(f"  - Warmup Ratio: {nfm_warmup_ratio} (from {'CLI' if args.nfm_warmup_ratio is not None else 'config.py'})")
# #         logger.info(f"  - Decay Ratio: {nfm_decay_ratio} (from {'CLI' if args.nfm_decay_ratio is not None else 'config.py'})")
# #
# #     # 🆕 加载预训练模型
# #     start_epoch, best_acc, optimizer_state, scheduler_state = load_pretrained_models(
# #         student, args, device
# #     )
# #
# #     # 加载教师模型
# #     teacher = load_teacher_model(args, device)
# #
# #     # 🆕 梯度平衡器参数处理（命令行参数优先级高于config.py）
# #     from new_frame.config import Config
# #     config = Config()
# #
# #     # 处理use_gradient_balancing（命令行优先）
# #     if args.no_gradient_balancing:
# #         use_gradient_balancing = False
# #         logger.info("Gradient Balancing: Disabled (via --no_gradient_balancing)")
# #     elif args.use_gradient_balancing:
# #         use_gradient_balancing = True
# #         logger.info("Gradient Balancing: Enabled (via --use_gradient_balancing)")
# #     else:
# #         use_gradient_balancing = config.use_gradient_balancing
# #         logger.info(f"Gradient Balancing: {'Enabled' if use_gradient_balancing else 'Disabled'} (from config.py)")
# #
# #     # 处理其他梯度平衡参数（命令行优先）
# #     gradient_balance_alpha = args.gradient_balance_alpha if args.gradient_balance_alpha is not None else config.gradient_balance_alpha
# #     gradient_balance_momentum = args.gradient_balance_momentum if args.gradient_balance_momentum is not None else config.gradient_balance_momentum
# #     gradient_balance_log_interval = args.gradient_balance_log_interval if args.gradient_balance_log_interval is not None else config.gradient_balance_log_interval
# #
# #     if use_gradient_balancing:
# #         logger.info(f"  - Alpha: {gradient_balance_alpha}")
# #         logger.info(f"  - Momentum: {gradient_balance_momentum}")
# #         logger.info(f"  - Log Interval: {gradient_balance_log_interval}")
# #
# #     # 🆕 HMI损失组件权重（命令行优先）
# #     hmi_mine_weight = args.hmi_mine_weight if args.hmi_mine_weight is not None else config.hmi_alpha
# #     hmi_hsic_weight = args.hmi_hsic_weight if args.hmi_hsic_weight is not None else config.hmi_beta
# #     hmi_kl_weight = args.hmi_kl_weight if args.hmi_kl_weight is not None else config.hmi_gamma
# #
# #     logger.info(f"HMI Loss Weights: MINE={hmi_mine_weight}, HSIC={hmi_hsic_weight}, KL={hmi_kl_weight}")
# #
# #     # 创建训练器
# #     trainer = StudentTrainer(
# #         student_model=student,
# #         teacher_model=teacher,
# #         use_distillation=args.use_distillation,
# #         hmi_loss_weight=args.hmi_weight,
# #         recon_loss_weight=args.recon_loss_weight,  # 🆕 添加这一行
# #         ce_loss_weight=args.ce_weight,
# #         device=device,
# #         # 🆕 梯度平衡参数
# #         use_gradient_balancing=use_gradient_balancing,
# #         gradient_balance_alpha=gradient_balance_alpha,
# #         gradient_balance_momentum=gradient_balance_momentum,
# #         gradient_balance_log_interval=gradient_balance_log_interval,
# #         # 🆕 HMI损失组件权重
# #         hmi_alpha=hmi_mine_weight,
# #         hmi_beta=hmi_hsic_weight,
# #         hmi_gamma=hmi_kl_weight
# #     )
# #
# #     # 创建优化器
# #     if args.use_gmd:
# #         optimizer = GMDOptimizer(
# #             student.parameters(),
# #             lr=args.lr,
# #             weight_decay=args.weight_decay,
# #             alpha=args.gmd_alpha
# #         )
# #         logger.info("Using GMD Optimizer")
# #     else:
# #         optimizer = optim.AdamW(
# #             student.parameters(),
# #             lr=args.lr,
# #             weight_decay=args.weight_decay
# #         )
# #         logger.info("Using AdamW Optimizer")
# #
# #     # 🆕 恢复优化器状态
# #     if optimizer_state is not None:
# #         optimizer.load_state_dict(optimizer_state)
# #         logger.info("✓ Optimizer state restored")
# #
# #     # 学习率调度器
# #     scheduler = optim.lr_scheduler.CosineAnnealingLR(
# #         optimizer,
# #         T_max=args.epochs,
# #         eta_min=args.lr * 0.01
# #     )
# #
# #     # 🆕 恢复调度器状态
# #     if scheduler_state is not None:
# #         scheduler.load_state_dict(scheduler_state)
# #         logger.info("✓ Scheduler state restored")
# #
# #     # 训练循环
# #     train_history = []
# #     val_history = []
# #
# #     logger.info("=" * 80)
# #     logger.info(f"Starting training from epoch {start_epoch}...")
# #     logger.info("=" * 80)
# #
# #     for epoch in range(start_epoch, args.epochs + 1):
# #         # 🆕 训练（传入logger以打印梯度统计和HMI损失分解）
# #         train_metrics = trainer.train_epoch(
# #             train_loader,
# #             optimizer,
# #             epoch,
# #             args.epochs,
# #             logger=logger
# #         )
# #
# #         # 记录
# #         train_history.append(train_metrics)
# #
# #         # 🆕 打印训练信息（包含HMI损失分解）
# #         log_str = f"Epoch [{epoch}/{args.epochs}] Train - "
# #         log_str += f"Loss: {train_metrics['loss']:.4f}, "
# #         log_str += f"CE Loss: {train_metrics['ce_loss']:.4f}, "
# #
# #         # 🆕 添加重建损失输出
# #         if 'recon_loss' in train_metrics and train_metrics['recon_loss'] > 0:
# #             log_str += f"Recon Loss: {train_metrics['recon_loss']:.4f}, "
# #
# #
# #         if 'hmi_loss' in train_metrics:
# #             log_str += f"HMI Loss: {train_metrics['hmi_loss']:.4f} "
# #             # HMI损失分解
# #             if 'mine_loss' in train_metrics:
# #                 log_str += f"(MINE: {train_metrics['mine_loss']:.4f}, "
# #                 log_str += f"HSIC: {train_metrics['hsic_loss']:.4f}, "
# #                 log_str += f"KL: {train_metrics['kl_loss']:.4f}), "
# #
# #         log_str += f"Acc: {train_metrics['accuracy']:.2f}%"
# #         logger.info(log_str)
# #
# #         # 评估
# #         if epoch % args.eval_interval == 0:
# #             # 🆕 混合模态评估
# #             if args.mixed_modality_eval:
# #                 val_metrics, modality_accs = trainer.evaluate(
# #                     val_loader,
# #                     use_mixed_modality=True
# #                 )
# #                 logger.info(f"Epoch [{epoch}/{args.epochs}] Val (Mixed Modality) - "
# #                            f"Loss: {val_metrics['loss']:.4f}, "
# #                            f"Acc: {val_metrics['accuracy']:.2f}%")
# #                 logger.info(f"  Modality Accs: P={modality_accs.get('pressure', 0):.2f}%, "
# #                            f"V={modality_accs.get('vibration', 0):.2f}%, "
# #                            f"P+V={modality_accs.get('full', 0):.2f}%")
# #             else:
# #                 val_metrics = trainer.evaluate(val_loader)
# #                 logger.info(f"Epoch [{epoch}/{args.epochs}] Val - "
# #                            f"Loss: {val_metrics['loss']:.4f}, "
# #                            f"Acc: {val_metrics['accuracy']:.2f}%")
# #
# #             val_history.append(val_metrics)
# #
# #             # 保存最佳模型
# #             if val_metrics['accuracy'] > best_acc:
# #                 best_acc = val_metrics['accuracy']
# #                 best_model_path = os.path.join(args.save_dir, 'best_model.pt')
# #                 torch.save({
# #                     'epoch': epoch,
# #                     'model_state_dict': student.state_dict(),
# #                     'optimizer_state_dict': optimizer.state_dict(),
# #                     'scheduler_state_dict': scheduler.state_dict(),
# #                     'best_acc': best_acc,
# #                     'train_metrics': train_metrics,
# #                     'val_metrics': val_metrics,
# #                     'config': config_dict,
# #                     # 🔧 添加 PV-NFM 配置
# #                     'pvnfm_config': {
# #                         'context_dim': mag_pvnfm_context_dim,
# #                         'num_bands': mag_pvnfm_num_bands,
# #                         'hidden_inr': mag_pvnfm_hidden_inr,
# #                     } if use_pvnfm else None,
# #
# #                     # 🆕 添加归一化统计
# #                     'mag_mean': train_loader.dataset.mag_mean if hasattr(train_loader.dataset, 'mag_mean') else None,
# #                     'mag_std': train_loader.dataset.mag_std if hasattr(train_loader.dataset, 'mag_std') else None
# #
# #                 }, best_model_path)
# #                 logger.info(f"✓ Saved best model with accuracy: {best_acc:.2f}%")
# #
# #         # 定期保存检查点
# #         if epoch % args.save_interval == 0:
# #             checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
# #             # 确保目录存在
# #             os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
# #             torch.save({
# #                 'epoch': epoch,
# #                 'model_state_dict': student.state_dict(),
# #                 'optimizer_state_dict': optimizer.state_dict(),
# #                 'scheduler_state_dict': scheduler.state_dict(),
# #                 'train_history': train_history,
# #                 'val_history': val_history,
# #                 'best_acc': best_acc,
# #                 'config': config_dict
# #             }, checkpoint_path)
# #
# #         # 更新学习率
# #         scheduler.step()
# #
# #     # 保存最终模型
# #     final_model_path = os.path.join(args.save_dir, 'final_model.pt')
# #     torch.save({
# #         'epoch': args.epochs,
# #         'model_state_dict': student.state_dict(),
# #         'optimizer_state_dict': optimizer.state_dict(),
# #         'train_history': train_history,
# #         'val_history': val_history,
# #         'best_acc': best_acc,
# #         'config': config_dict
# #     }, final_model_path)
# #
# #     logger.info("=" * 80)
# #     logger.info(f"Training completed! Best accuracy: {best_acc:.2f}%")
# #     logger.info("=" * 80)
# #
# #     # 保存训练历史
# #     history = {
# #         'train': train_history,
# #         'val': val_history,
# #         'best_acc': best_acc
# #     }
# #     with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
# #         json.dump(history, f, indent=4)
# #
# #     # 🆕 自动测试
# #     if args.auto_test:
# #         best_model_path = os.path.join(args.save_dir, 'best_model.pt')
# #         run_auto_test(args, best_model_path)
# #
# #
# # if __name__ == '__main__':
# #     args = parse_args()
# #     train(args)