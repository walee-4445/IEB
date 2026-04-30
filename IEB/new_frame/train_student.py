
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
    #  自动测试
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
