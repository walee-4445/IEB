

import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import sys
import json
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from teacher_model_ieb import IEBTeacherMagnitude
# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


#from new_frame.student_model_magnitude import StudentModelMagnitude
from student_model_magnitude_sdd import StudentModelMagnitudeSDD
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Evaluate CMAD Models',
        epilog="""

        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 演示模式
    parser.add_argument('--demo', action='store_true',
                        help='演示模式：使用随机初始化的模型进行测试（无需模型文件）')

    # 模型参数
    parser.add_argument('--model_type', type=str,
                        choices=['teacher', 'student'],
                        help='模型类型 (teacher/student)')
    parser.add_argument('--model_path', type=str,
                        help='模型检查点路径')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./pressure_vibration1_S2_02',
                        help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # 模型配置
    parser.add_argument('--seq_len', type=int, default=2560,
                        help='序列长度')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='分类类别数')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='潜在特征维度')
    parser.add_argument('--num_latents', type=int, default=32,
                        help='Perceiver latent数量（默认32，会从checkpoint自动读取）')

    # 评估参数
    parser.add_argument('--test_modalities', action='store_true',
                        help='测试不同模态组合（仅Student）')
    parser.add_argument('--use_pvnfm_inference', action='store_true',
                        help='测试缺失模态时启用PV-NFM补全')
    parser.add_argument('--pvnfm_inference_weight', type=float, default=1.0,
                        help='测试时PV-NFM补全权重')
    parser.add_argument('--save_results', action='store_true',
                        help='保存评估结果')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='结果保存目录')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')

    args = parser.parse_args()
    
    # 验证参数
    if not args.demo:
        if not args.model_type:
            parser.error("需要指定 --model_type (teacher/student)，或使用 --demo 模式")
        if not args.model_path:
            parser.error("需要指定 --model_path，或使用 --demo 模式进行测试")
    else:
        # 演示模式的默认值
        if not args.model_type:
            args.model_type = 'teacher'
            logger.info("演示模式：使用默认模型类型 'teacher'")
    
    return args


def load_model(args, device):

   # is_magnitude_model = False  # 默认为时域模型
    checkpoint = None  # 保存checkpoint以便后续使用
    
    if args.demo:
        logger.warning("=" * 60)
        logger.warning("演示模式：使用随机初始化的模型（非训练模型）")
        logger.warning("结果仅用于测试代码流程，不代表实际性能")
        logger.warning("=" * 60)
    else:
        logger.info(f"Loading {args.model_type} model from {args.model_path}")

    # 修改后：
    if args.model_type == 'teacher':
        F_bins = args.seq_len // 2 + 1
        is_magnitude_model = True  # 教师也是频域模型

        if not args.demo:
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            model = IEBTeacherMagnitude(
                F_bins=F_bins,
                num_classes=args.num_classes,
                d_model=args.latent_dim,
                dropout=0.1,
                llm_model=None,
                freeze_llm=True
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Teacher model loaded from {args.model_path}")
        else:
            model = IEBTeacherMagnitude(
                F_bins=F_bins,
                num_classes=args.num_classes,
                d_model=args.latent_dim,
                dropout=0.1,
                llm_model=None,
                freeze_llm=True
            )
            logger.info("Teacher model initialized with random weights (demo mode)")
    else:  # student
        # 先加载checkpoint以获取配置信息
        if not args.demo:
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)  # 保存checkpoint
            
            # 检测模型类型：StudentModel（时域）vs StudentModelMagnitude（频域）
            # 通过检查卷积核形状来判断（最可靠的方法）
            state_dict = checkpoint['model_state_dict']
            state_dict_keys = list(state_dict.keys())
            
            #  调试：打印前10个键来帮助诊断
            logger.info(f"First 10 keys in checkpoint: {state_dict_keys[:10]}")
            
            # 检查第一个卷积层的形状来判断模型类型
            # 频域模型：kernel_size=5 → shape [32, 1, 5]
            # 时域模型：kernel_size=7 → shape [32, 1, 7]
            conv_weight_key = 'pressure_encoder.conv_blocks.0.weight'
            
            if conv_weight_key in state_dict:
                conv_shape = state_dict[conv_weight_key].shape
                kernel_size = conv_shape[2]  # [out_ch, in_ch, kernel_size]
                
                logger.info(f"Detected conv kernel shape: {conv_shape}, kernel_size={kernel_size}")
                
                # 根据卷积核大小判断
                if kernel_size == 5:
                    is_magnitude_model = True
                    logger.info("Detected kernel_size=5 → Magnitude-Only Model (Frequency-Domain)")
                elif kernel_size == 7:
                    is_magnitude_model = False
                    logger.info("Detected kernel_size=7 → Time-Domain Model")
                else:
                    # 未知卷积核大小，尝试其他方法
                    logger.warning(f"Unknown kernel_size={kernel_size}, trying alternative detection")
                    # 检查是否有modality_generator（PV-NFM的特征）
                    has_modality_generator = any('modality_generator' in key for key in state_dict_keys)
                    if has_modality_generator:
                        is_magnitude_model = True
                        logger.info("Detected modality_generator → assuming Magnitude-Only Model")
                    else:
                        is_magnitude_model = False
                        logger.warning("Cannot determine model type clearly, assuming Time-Domain Model")
            else:
                # 找不到卷积层，使用备用方法
                logger.warning(f"Cannot find {conv_weight_key}, using alternative detection")
                has_modality_generator = any('modality_generator' in key for key in state_dict_keys)
                if has_modality_generator:
                    is_magnitude_model = True
                    logger.info("Detected modality_generator → assuming Magnitude-Only Model")
                else:
                    is_magnitude_model = False
                    logger.warning("Assuming Time-Domain Model by default")
            
            # 检查是否使用NFM/PV-NFM
            use_nfm = any('nfm_generator' in key for key in state_dict_keys)
            use_pvnfm = any('modality_generator' in key for key in state_dict_keys)

            num_latents = args.num_latents  # 默认值
            encoder_channels = [32, 64, 128]  # 默认值
            latent_dim = args.latent_dim  #  添加latent_dim读取

            # 🔧 从checkpoint的config中读取参数
            if 'config' in checkpoint:
                config = checkpoint['config']
                if 'num_latents' in config:
                    num_latents = config['num_latents']
                    logger.info(f"Using num_latents={num_latents} from checkpoint config")

                # 🔧 优先从权重推断latent_dim（最可靠！）
                fc_key = 'pressure_encoder.fc.weight'
                if fc_key in state_dict:
                    inferred_latent_dim = state_dict[fc_key].shape[0]  # [latent_dim, last_conv_out]
                    
                    # 检查config中的值是否匹配
                    config_latent_dim = None
                    if 'latent_dim' in config:
                        config_latent_dim = config['latent_dim']
                    elif 'magnitude_latent_dim' in config:
                        config_latent_dim = config['magnitude_latent_dim']
                    
                    if config_latent_dim and config_latent_dim != inferred_latent_dim:
                        logger.warning(f"Config latent_dim={config_latent_dim} doesn't match weights latent_dim={inferred_latent_dim}")
                        logger.warning(f"Using inferred value from weights: {inferred_latent_dim}")
                    
                    latent_dim = inferred_latent_dim
                    logger.info(f"✓ Using latent_dim={latent_dim} (inferred from fc layer weights)")
                else:
                    # Fallback到config
                    if 'latent_dim' in config:
                        latent_dim = config['latent_dim']
                        logger.info(f"Using latent_dim={latent_dim} from checkpoint config")
                    elif 'magnitude_latent_dim' in config:
                        latent_dim = config['magnitude_latent_dim']
                        logger.info(f"Using latent_dim={latent_dim} from checkpoint config (magnitude_latent_dim)")
                    else:
                        logger.warning(f"Cannot infer latent_dim, using default {args.latent_dim}")
                        latent_dim = args.latent_dim

                # 读取编码器配置
                if 'encoder_channels' in config:
                    encoder_channels = config['encoder_channels']
                    logger.info(f"Using encoder_channels={encoder_channels} from checkpoint config")
                elif 'student_encoder_channels' in config:
                    encoder_channels = config['student_encoder_channels']
                    logger.info(
                        f"Using encoder_channels={encoder_channels} from checkpoint config (student_encoder_channels)")
                else:
                    # 如果config中没有encoder_channels，从权重推断
                    logger.warning("encoder_channels not found in config, inferring from model weights...")
                    
                    # 方法：检查fc层的输入维度来推断最后一层卷积的输出通道数
                    fc_key = 'pressure_encoder.fc.weight'
                    if fc_key in state_dict:
                        last_conv_out = state_dict[fc_key].shape[1]  # [latent_dim, last_conv_out]
                        
                        # 检查有几层卷积（每层包含conv, bn, relu，索引间隔不固定）
                        # 统计所有conv层（权重键包含.weight且不是fc层）
                        conv_indices = []
                        for key in state_dict_keys:
                            if 'pressure_encoder.conv_blocks.' in key and '.weight' in key and 'fc' not in key:
                                try:
                                    idx = int(key.split('.')[2])
                                    if idx not in conv_indices:
                                        conv_indices.append(idx)
                                except:
                                    pass
                        
                        # 🔧 最可靠的方法：直接从fc层和第一层卷积推断
                        # fc.weight.shape = [latent_dim, last_conv_out]
                        last_conv_out = state_dict[fc_key].shape[1]
                        
                        # 读取第一层卷积的输出通道数
                        first_conv_key = 'pressure_encoder.conv_blocks.0.weight'
                        if first_conv_key in state_dict:
                            first_out = state_dict[first_conv_key].shape[0]
                            
                            # 判断是单层还是多层
                            if first_out == last_conv_out:
                                # 单层：第一层输出 = 最后一层输出
                                encoder_channels = [first_out]
                                num_layers = 1
                            else:
                                # 多层：需要推断中间层
                                # 尝试读取所有卷积层
                                encoder_channels = []
                                for key in sorted(state_dict_keys):
                                    if 'pressure_encoder.conv_blocks.' in key and '.weight' in key and 'fc' not in key and 'bn' not in key:
                                        try:
                                            idx = int(key.split('.')[2])
                                            # 卷积层的索引：0, 4, 8, ... (每层4个组件)
                                            if idx % 4 == 0:
                                                out_ch = state_dict[key].shape[0]
                                                encoder_channels.append(out_ch)
                                        except:
                                            pass
                                
                                # 如果推断失败，使用简单策略
                                if not encoder_channels or encoder_channels[-1] != last_conv_out:
                                    encoder_channels = [first_out, last_conv_out]
                                
                                num_layers = len(encoder_channels)
                        else:
                            # 找不到第一层，直接使用last_conv_out
                            encoder_channels = [last_conv_out]
                            num_layers = 1
                        
                        logger.info(f"✓ Inferred encoder_channels={encoder_channels} from weights (layers={num_layers}, last_out={last_conv_out})")
                    else:
                        logger.error(f"Cannot find {fc_key} in checkpoint, using default [32, 64]")
                        encoder_channels = [32, 64]
            else:
                logger.warning(f"Config not found in checkpoint, using defaults")

            # 根据检测结果创建对应的模型
            if is_magnitude_model:
                # 频域模型（StudentModelMagnitude）
                F_bins = args.seq_len // 2 + 1

                # 直接使用checkpoint中的encoder_channels，不做任何假设
                magnitude_encoder_channels = encoder_channels

                logger.info(f"Detected Magnitude-Only Student Model (F_bins={F_bins})")
                logger.info(
                    f"Model configuration: use_pvnfm={use_pvnfm}, num_latents={num_latents}, encoder_channels={magnitude_encoder_channels}, latent_dim={latent_dim}")


                pvnfm_config = checkpoint.get('pvnfm_config', None)
                if pvnfm_config:
                    pvnfm_context_dim = pvnfm_config['context_dim']
                    pvnfm_num_bands = pvnfm_config['num_bands']
                    pvnfm_hidden_inr = pvnfm_config['hidden_inr']
                else:
                    # 使用默认值
                    pvnfm_context_dim = 256
                    pvnfm_num_bands = 32
                    pvnfm_hidden_inr = 128
                # 方案2：优先从checkpoint的model_config读取完整配置
                if 'config' in checkpoint:
                    config = checkpoint['config']
                    logger.info("✓ Using saved config from checkpoint")

                    # 从model_config读取所有参数
                    pvnfm_context_dim = config.get('pvnfm_context_dim', 128)
                    pvnfm_num_bands = config.get('pvnfm_num_bands', 16)
                    pvnfm_hidden_inr = config.get('pvnfm_hidden_inr', 128)
                    num_latents = config.get('num_latents', num_latents)
                    latent_dim = config.get('latent_dim', latent_dim)
                    magnitude_encoder_channels = config.get('encoder_channels', magnitude_encoder_channels)
                    use_dual_perceiver = config.get('use_dual_perceiver', False)
                    use_transformer_fusion = config.get('use_transformer_fusion', False)
                    perceiver_depth = config.get('perceiver_depth', 3)
                    perceiver_heads = config.get('perceiver_heads', 8)
                    max_k = config.get('max_k', 6)

                    # 从权重推断num_latents（最可靠！）
                    latent_queries_key = 'perceiver.latent_queries'
                    if latent_queries_key in state_dict:
                        inferred_num_latents = state_dict[latent_queries_key].shape[1]
                        if inferred_num_latents != num_latents:
                            logger.warning(f"⚠ num_latents不匹配: config={num_latents}, weights={inferred_num_latents}")
                            logger.warning(f"   使用权重中的值: {inferred_num_latents}")
                            num_latents = inferred_num_latents

                    logger.info(f"  pvnfm_context_dim={pvnfm_context_dim}")
                    logger.info(f"  pvnfm_num_bands={pvnfm_num_bands}")
                    logger.info(f"  pvnfm_hidden_inr={pvnfm_hidden_inr}")
                    logger.info(f"  num_latents={num_latents}")
                    logger.info(f"  latent_dim={latent_dim}")
                    logger.info(f"  perceiver_depth={perceiver_depth}")
                    logger.info(f"  perceiver_heads={perceiver_heads}")
                    logger.info(f"  max_k={max_k}")
                else:
                    # 回退：尝试从旧的pvnfm_config读取
                    logger.warning("⚠ model_config not found in checkpoint, trying pvnfm_config...")

                    # 从权重推断num_latents
                    latent_queries_key = 'perceiver.latent_queries'
                    if latent_queries_key in state_dict:
                        num_latents = state_dict[latent_queries_key].shape[1]
                        logger.info(f"✓ 从权重推断num_latents={num_latents}")

                    pvnfm_config = checkpoint.get('pvnfm_config', None)
                    if pvnfm_config:
                        pvnfm_context_dim = pvnfm_config['context_dim']
                        pvnfm_num_bands = pvnfm_config['num_bands']
                        pvnfm_hidden_inr = pvnfm_config['hidden_inr']
                        logger.info(f"✓ Using pvnfm_config from checkpoint")
                    else:
                        # 最后的回退：使用推断的默认值
                        logger.warning("⚠ No config found, using inferred defaults")
                        pvnfm_context_dim = 128  # 修正默认值
                        pvnfm_num_bands = 16  #  修正默认值
                        pvnfm_hidden_inr = 128

                    # 其他参数使用默认值
                    use_dual_perceiver = False
                    use_transformer_fusion = False
                    perceiver_depth = 3
                    perceiver_heads = 8
                    max_k = 6

                model = StudentModelMagnitudeSDD(
                    F_bins=F_bins,
                    num_classes=args.num_classes,
                    encoder_channels=magnitude_encoder_channels,
                    latent_dim=latent_dim,
                    num_latents=num_latents,
                    perceiver_depth=perceiver_depth,
                    perceiver_heads=perceiver_heads,
                    max_k=max_k,
                    use_pvnfm=use_pvnfm,
                    use_dual_perceiver=use_dual_perceiver,
                    use_transformer_fusion=use_transformer_fusion,
                    pvnfm_context_dim=pvnfm_context_dim,
                    pvnfm_num_bands=pvnfm_num_bands,
                    pvnfm_hidden_inr=pvnfm_hidden_inr
                )

            
            # 加载权重（使用strict=False以支持动态网络）
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['model_state_dict'],
                strict=False
            )
            

            non_dynamic_unexpected = [k for k in unexpected_keys
                                     if 'nfm_generator.freq_' not in k and 'modality_generator' not in k]
            if non_dynamic_unexpected:
                logger.warning(f"Unexpected keys (non-dynamic): {non_dynamic_unexpected}")
            
            logger.info(f"Model loaded successfully from checkpoint")
            if use_nfm or use_pvnfm:
                logger.info(f"Dynamic networks will be created on first forward pass")
        else:
            # 演示模式：默认使用时域模型
            F_bins = args.seq_len // 2 + 1
            is_magnitude_model = True
            model = StudentModelMagnitudeSDD(
                F_bins=F_bins,
                num_classes=args.num_classes,
                encoder_channels=[32, 64],
                latent_dim=args.latent_dim,
                num_latents=args.num_latents,
                use_pvnfm=False
            )
            logger.info(f"Model initialized with random weights (demo mode)")
    
    model.to(device)
    model.eval()

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, is_magnitude_model, checkpoint


def create_dataloader(args, is_magnitude_model=False, checkpoint=None):
    """
    创建测试数据加载器
    """

    try:
        # 始终使用get_dataloaders_presplit（与训练一致）
        from data_loader import get_dataloaders_presplit
        from config import Config

        logger.info(f"Loading real test dataset from {args.data_dir}")
        logger.info(f"Loading real test dataset from {args.data_dir}")
        
        # 创建配置对象
        config = Config()
        # 从checkpoint读取训练时的配置
        if checkpoint and 'config' in checkpoint:
            saved_config = checkpoint['config']
            if 'seq_len' in saved_config:
                config.window_size = saved_config['seq_len']
                logger.info(f"Using seq_len={config.window_size} from checkpoint")

            train_loader, val_loader, test_loader, label_encoder = get_dataloaders_presplit(
                config=config,
                batch_size=args.batch_size,
                mode='student',
                use_fft=is_magnitude_model,  #  根据模型类型决定是否FFT
                seed=42  # 使用固定种子确保一致性
            )

            #  如果checkpoint中有保存的归一化统计，使用它
            if checkpoint and 'mag_mean' in checkpoint and checkpoint['mag_mean'] is not None:
                logger.info("Using normalization stats from training checkpoint")
                test_loader.dataset.set_mag_stats(
                    checkpoint['mag_mean'],
                    checkpoint['mag_std']
                )
                logger.info(f"  P_mean={checkpoint['mag_mean'][0]:.6f}, V_mean={checkpoint['mag_mean'][1]:.6f}")

            logger.info(f"Loaded real test dataset with {len(test_loader.dataset)} samples")
            logger.info(f"Number of classes: {len(label_encoder.classes_)}")
            logger.info(f"Class labels: {label_encoder.classes_}")
            logger.info(f"Test batches: {len(test_loader)}")

            return test_loader
    except Exception as e:
        logger.warning(f"Failed to load real data: {e}")
        logger.warning("Falling back to dummy dataloader for testing.")
        
        # 回退到模拟数据
        logger.warning(" Using dummy dataloader - results are not meaningful!")
        
        # 创建模拟测试数据
        test_pressure = torch.randn(500, args.seq_len, 1)
        test_vibration = torch.randn(500, args.seq_len, 1)
        test_labels = torch.randint(0, args.num_classes, (500,))

        test_dataset = TensorDataset(test_pressure, test_vibration, test_labels)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        return test_loader


def evaluate_teacher(model, dataloader, device):
    """评估教师模型"""
    all_preds = []
    all_labels = []
    all_probs = []
    #  添加标志用于只在第一个batch打印调试信息
    first_batch = True

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # 处理字典格式的batch（来自MultiModalTimeSeriesDataset）
            if isinstance(batch, dict):
                pressure = batch['pressure'].to(device)
                vibration = batch['vibration'].to(device)
                labels = batch['label'].to(device)
            # 处理tuple格式的batch（可能有filename）
            elif len(batch) == 4:
                pressure, vibration, labels, _ = batch
                pressure = pressure.to(device)
                vibration = vibration.to(device)
                labels = labels.to(device)
            else:
                pressure, vibration, labels = batch
                pressure = pressure.to(device)
                vibration = vibration.to(device)
                labels = labels.to(device)

             #  调试：检查输入（只在第一个batch）
            if first_batch:
                logger.info(f"🔍 Debug - Input shapes:")
                logger.info(f"  Pressure: {pressure.shape}")
                logger.info(f"  Vibration: {vibration.shape}")
                logger.info(f"  Labels: {labels.shape}")


            # 修改后：
            if isinstance(model, StudentModelMagnitudeSDD):
                global_logits, _ = model(pressure, vibration, freq_masks=None, mode='test')
                logits = global_logits
            elif isinstance(model, IEBTeacherMagnitude):
                logits, _ = model(pressure, vibration, freq_mask=None)
            else:
                logits = model(pressure, vibration)

                #  调试：检查输出（只在第一个batch）
            if first_batch:
                logger.info(f"🔍 Debug - Model outputs:")
                logger.info(f"  Logits shape: {logits.shape}")
                logger.info(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                logger.info(f"  Logits mean: {logits.mean().item():.2f}")
                logger.info(f"  Logits std: {logits.std().item():.2f}")

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            # 调试：检查预测分布（只在第一个batch）
            if first_batch:
                logger.info(f"🔍 Debug - Predictions:")
                logger.info(f"  Unique predictions: {preds.unique().tolist()}")
                pred_counts = [(p.item(), (preds == p).sum().item()) for p in preds.unique()]
                logger.info(f"  Prediction counts: {pred_counts}")
                first_batch = False  # 设置为False，后续batch不再打印

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())


    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)

    logger.info(f"📊 Complete prediction distribution:")
    logger.info(f"  Predicted classes: {unique_preds.tolist()}")
    logger.info(f"  Prediction counts: {pred_counts.tolist()}")
    logger.info(f"  True classes: {unique_labels.tolist()}")
    logger.info(f"  True counts: {label_counts.tolist()}")

    missing_classes = set(unique_labels) - set(unique_preds)
    if missing_classes:
        logger.warning(f"Classes never predicted: {missing_classes}")
        logger.warning(f"   This will cause UndefinedMetricWarning!")

    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0  # 添加zero_division参数
    )

    results = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

    return results



def evaluate_student_modalities(model, dataloader, device,
                                use_pvnfm_inference=False,
                                pvnfm_inference_weight=1.0):


    results_storage = {
        'Complete': {'preds': [], 'labels': []},
        'P_only':   {'preds': [], 'labels': []},
        'V_only':   {'preds': [], 'labels': []},
        'Mixed':    {'preds': [], 'labels': []},
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Eval Student Modalities'):
            # 处理batch
            if isinstance(batch, dict):
                pressure = batch['pressure'].to(device)
                vibration = batch['vibration'].to(device)
                labels = batch['label'].to(device)
            elif len(batch) == 4:
                pressure, vibration, labels, _ = batch
                pressure = pressure.to(device)
                vibration = vibration.to(device)
                labels = labels.to(device)
            else:
                pressure, vibration, labels = batch
                pressure = pressure.to(device)
                vibration = vibration.to(device)
                labels = labels.to(device)

            batch_size = pressure.size(0)

            fixed_mask_complete = torch.tensor([[1, 1]] * batch_size, device=device).float()
            fixed_mask_p_only   = torch.tensor([[1, 0]] * batch_size, device=device).float()
            fixed_mask_v_only   = torch.tensor([[0, 1]] * batch_size, device=device).float()

            # 1) Complete
            if isinstance(model, StudentModelMagnitudeSDD):
                logits_complete, _ = model(
                    pressure, vibration,
                    freq_masks=None,
                    mode='test',
                    fixed_mask=fixed_mask_complete,
                    enable_pvnfm_inference=use_pvnfm_inference,
                    pvnfm_inference_weight=pvnfm_inference_weight
                )
                logits_p_only, _ = model(
                    pressure, vibration,
                    freq_masks=None,
                    mode='test',
                    fixed_mask=fixed_mask_p_only,
                    enable_pvnfm_inference=use_pvnfm_inference,
                    pvnfm_inference_weight=pvnfm_inference_weight
                )
                logits_v_only, _ = model(
                    pressure, vibration,
                    freq_masks=None,
                    mode='test',
                    fixed_mask=fixed_mask_v_only,
                    enable_pvnfm_inference=use_pvnfm_inference,
                    pvnfm_inference_weight=pvnfm_inference_weight
                )
            else:
                logits_complete = model(
                    pressure, vibration,
                    mode='test',
                    fixed_mask=fixed_mask_complete,
                    enable_pvnfm_inference=use_pvnfm_inference,
                    pvnfm_inference_weight=pvnfm_inference_weight
                )
                logits_p_only = model(
                    pressure, vibration,
                    mode='test',
                    fixed_mask=fixed_mask_p_only,
                    enable_pvnfm_inference=use_pvnfm_inference,
                    pvnfm_inference_weight=pvnfm_inference_weight
                )
                logits_v_only = model(
                    pressure, vibration,
                    mode='test',
                    fixed_mask=fixed_mask_v_only,
                    enable_pvnfm_inference=use_pvnfm_inference,
                    pvnfm_inference_weight=pvnfm_inference_weight
                )

            # 2) Mixed：与验证一致的logits融合
            logits_mixed = (
                0.34 * logits_complete +
                0.33 * logits_p_only +
                0.33 * logits_v_only
            )

            preds_complete = logits_complete.argmax(dim=1)
            preds_p_only   = logits_p_only.argmax(dim=1)
            preds_v_only   = logits_v_only.argmax(dim=1)
            preds_mixed    = logits_mixed.argmax(dim=1)

            labels_np = labels.cpu().numpy()

            results_storage['Complete']['preds'].extend(preds_complete.cpu().numpy())
            results_storage['Complete']['labels'].extend(labels_np)

            results_storage['P_only']['preds'].extend(preds_p_only.cpu().numpy())
            results_storage['P_only']['labels'].extend(labels_np)

            results_storage['V_only']['preds'].extend(preds_v_only.cpu().numpy())
            results_storage['V_only']['labels'].extend(labels_np)

            results_storage['Mixed']['preds'].extend(preds_mixed.cpu().numpy())
            results_storage['Mixed']['labels'].extend(labels_np)

    # 统一计算指标
    results = {}
    for key, data in results_storage.items():
        all_preds = np.array(data['preds'])
        all_labels = np.array(data['labels'])

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        results[key] = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'predictions': all_preds,
            'labels': all_labels
        }

    return results
def plot_confusion_matrix(labels, preds, num_classes, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()

def main():
    """主函数"""
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 创建输出目录
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    model, is_magnitude_model, checkpoint = load_model(args, device)

    #
    test_loader = create_dataloader(args, is_magnitude_model=is_magnitude_model, checkpoint=checkpoint)
    logger.info(f"Test batches: {len(test_loader)}")
    

    if args.model_type == 'student' and is_magnitude_model:
        logger.info("Note: Evaluating Magnitude-Only (Frequency-Domain) Student Model")
        logger.info("Data will be automatically converted to magnitude spectrum during evaluation")

    # 评估
    if args.model_type == 'teacher':
        logger.info("Evaluating Teacher Model...")
        results = evaluate_teacher(model, test_loader, device)

        print("\n" + "=" * 60)
        print("Teacher Model Evaluation Results")
        print("=" * 60)
        print(f"Accuracy:  {results['accuracy']:.2f}%")
        print(f"Precision: {results['precision']:.2f}%")
        print(f"Recall:    {results['recall']:.2f}%")
        print(f"F1 Score:  {results['f1']:.2f}%")
        print("=" * 60)

        if args.save_results:
            # 保存结果
            save_dict = {k: v for k, v in results.items()
                         if k not in ['predictions', 'labels', 'probabilities']}
            with open(os.path.join(args.output_dir, 'teacher_results.json'), 'w') as f:
                json.dump(save_dict, f, indent=4)

            # 绘制混淆矩阵
            plot_confusion_matrix(
                results['labels'], results['predictions'], args.num_classes,
                os.path.join(args.output_dir, 'teacher_confusion_matrix.png')
            )

    else:  # student
        if args.test_modalities:
            logger.info("Evaluating Student Model with different modalities...")
            results = evaluate_student_modalities(model, test_loader, device,
    use_pvnfm_inference=args.use_pvnfm_inference,
    pvnfm_inference_weight=args.pvnfm_inference_weight)

            mixed_accuracy = results['Mixed']['accuracy']

            print("\n" + "=" * 60)
            print("Student Model Test Results (Different Modalities)")
            print("=" * 60)
            print(f"Mixed Accuracy: {results['Mixed']['accuracy']:.2f}%")
            print(f"  - Full Modality (P+V): {results['Complete']['accuracy']:.2f}%")
            print(f"  - Pressure Only (P):   {results['P_only']['accuracy']:.2f}%")
            print(f"  - Vibration Only (V):  {results['V_only']['accuracy']:.2f}%")
            print("=" * 60)

            print("\nDetailed Metrics:")
            for mask_name, metrics in results.items():
                print(f"\n{mask_name}:")
                print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
                print(f"  Precision: {metrics['precision']:.2f}%")
                print(f"  Recall:    {metrics['recall']:.2f}%")
                print(f"  F1 Score:  {metrics['f1']:.2f}%")
            print("=" * 60)
            if args.save_results:
                # 保存结果（包含混合准确率）
                save_dict = {
                    'mixed_accuracy': results['Mixed']['accuracy'],
                    'modalities': {}
                }
                for mask_name, metrics in results.items():
                    save_dict['modalities'][mask_name] = {
                        k: v for k, v in metrics.items()
                        if k not in ['predictions', 'labels']
                    }

                with open(os.path.join(args.output_dir, 'student_modality_results.json'), 'w') as f:
                    json.dump(save_dict, f, indent=4)
                
                logger.info(f"Results saved to {args.output_dir}/student_modality_results.json")
        else:
            logger.info("Evaluating Student Model (complete modality)...")
            results = evaluate_teacher(model, test_loader, device)  # 使用相同的评估函数

            print("\n" + "=" * 60)
            print("Student Model Evaluation Results")
            print("=" * 60)
            print(f"Accuracy:  {results['accuracy']:.2f}%")
            print(f"Precision: {results['precision']:.2f}%")
            print(f"Recall:    {results['recall']:.2f}%")
            print(f"F1 Score:  {results['f1']:.2f}%")
            print("=" * 60)
if __name__ == '__main__':
    main()