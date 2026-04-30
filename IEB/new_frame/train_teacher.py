
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse  # ✅ 新增
from config import Config
#from data_loader import get_magnitude_dataloaders
from teacher_model_ieb import IEBTeacherMagnitude
from data_loader import get_dataloaders_presplit  # 使用新函数
import numpy as np
import random

def set_seed(seed=42):
    """
    固定所有随机种子,确保训练结果完全可复现

    Args:
        seed: 随机种子 (默认42)
    """
    print(f"\n{'=' * 60}")
    print(f" 固定随机种子: {seed}")
    print(f"{'=' * 60}")

    # 1. Python内置随机数
    random.seed(seed)

    # 2. Numpy随机数
    np.random.seed(seed)

    # 3. PyTorch CPU随机数
    torch.manual_seed(seed)

    # 4. PyTorch GPU随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU

    # 5. CUDNN确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. Python哈希随机化
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✓ 所有随机源已固定")
    print(f"✓ 训练结果将完全可复现")
    print(f"{'=' * 60}\n")


def parse_args():
    """解析命令行参数"""



    parser = argparse.ArgumentParser(description='训练IEB教师网络')

    # 基础参数
    parser.add_argument('--data_dir', type=str, default='./pressure_vibration1_S2_02')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认42)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='../checkpoints/teacher')

    # 分层学习率参数
    parser.add_argument('--lr_cnn', type=float, default=5e-4,
                        help='CNN编码器学习率')
    parser.add_argument('--lr_projection', type=float, default=5e-4,
                        help='CNN投影层学习率')
    parser.add_argument('--lr_classifier', type=float, default=3e-4,
                        help='分类头学习率')

    #  分层权重衰减参数
    parser.add_argument('--wd_cnn', type=float, default=1e-4,
                        help='CNN编码器权重衰减')
    parser.add_argument('--wd_projection', type=float, default=1e-4,
                        help='CNN投影层权重衰减')
    parser.add_argument('--wd_classifier', type=float, default=5e-4,
                        help='分类头权重衰减')

    # 其他参数
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--d_model', type=int, default=256)

    parser.add_argument('--patience', type=int, default=15)
    return parser.parse_args()
def train_teacher():
    #  解析参数
    args = parse_args()
    #  在所有操作之前固定种子
    set_seed(args.seed)
    # 配置
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 配置
    config = Config()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 数据加载
    print("加载数据...")
    # train_loader, val_loader, test_loader, _ = get_magnitude_dataloaders(
    #     config=config,
    #     batch_size=32,
    #     mode='student'
    # )
    # 使用新的数据加载函数
    train_loader, val_loader, test_loader, _ = get_dataloaders_presplit(
        config=config,
        batch_size=args.batch_size,
        mode='student', use_fft=True,
        seed=args.seed
    )
    # 🔍 添加调试代码
    print("\n" + "=" * 60)
    print("🔍 数据维度检查:")
    first_batch = next(iter(train_loader))
    p_shape = first_batch['pressure'].shape
    v_shape = first_batch['vibration'].shape
    print(f"  Pressure shape: {p_shape}")
    print(f"  Vibration shape: {v_shape}")

    if p_shape[1] == 2560:
        print("   时域数据（未进行FFT转换）")
        print("  ️  警告：教师模型期望 (B, 1281, 1)，但数据是 (B, 2560, 1)")
        print("  ️  这会导致维度不匹配错误！")
    elif p_shape[1] == 1281:
        print("   频域数据（已进行FFT转换）")
        print("  ✓ 与教师模型匹配")
    else:
        print(f"  ❓ 未知维度: {p_shape[1]}")
    print("=" * 60 + "\n")

    print("\n重新固定随机种子 (创建模型前)...")
    set_seed(args.seed)
    # 创建教师网络
    print("创建教师网络...")
    teacher = IEBTeacherMagnitude(
        F_bins=1281,
        num_classes=9,
        d_model=args.d_model,  #  保持256 (与CNN输出匹配,避免额外投影)
        patch_len=16,  #  这个参数现在无用了,但保留兼容性
        stride=8,  #  这个参数现在无用了,但保留兼容性
        dropout=args.dropout,  #从0.4降到0.3 (CNN参数更多,需要更少dropout)
        llm_model=None,
        freeze_llm=True
    ).to(device)


    # 学习率调度器
    # ✅ 手动解冻GPT-2最后2层
    if hasattr(teacher, 'llm_model') and hasattr(teacher.llm_model, 'h'):
        print("\n解冻GPT-2最后2层...")
        # 冻结前10层
        for i in range(10):
            for param in teacher.llm_model.h[i].parameters():
                param.requires_grad = False

        # 解冻最后2层 (h.10, h.11)
        for i in range(10, 12):
            for param in teacher.llm_model.h[i].parameters():
                param.requires_grad = True

        # 统计可训练参数
        llm_trainable = sum(p.numel() for p in teacher.llm_model.parameters() if p.requires_grad)
        print(f"✓ GPT-2可训练参数: {llm_trainable:,}")

    print(f"教师参数量: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in teacher.parameters() if p.requires_grad):,}")

    # ✅ 修复: 收集参数时避免重复
    # 1. 收集CNN编码器参数
    cnn_params = list(teacher.cnn_encoder.parameters())
    cnn_param_ids = {id(p) for p in cnn_params}

    # 2. 收集CNN投影层参数
    projection_params = list(teacher.cnn_projection.parameters())
    projection_param_ids = {id(p) for p in projection_params}

    # 3. 收集分类头参数
    classifier_params = list(teacher.projection.parameters())
    classifier_param_ids = {id(p) for p in classifier_params}

    # 4. 收集GPT-2最后2层参数 (排除已收集的)
    llm_params = []
    llm_param_ids = set()
    if hasattr(teacher, 'llm_model') and hasattr(teacher.llm_model, 'h'):
        for i in range(10, 12):  # 最后2层
            for param in teacher.llm_model.h[i].parameters():
                if param.requires_grad:
                    param_id = id(param)
                    # 确保不重复
                    if param_id not in cnn_param_ids and \
                            param_id not in projection_param_ids and \
                            param_id not in classifier_param_ids and \
                            param_id not in llm_param_ids:
                        llm_params.append(param)
                        llm_param_ids.add(param_id)

    # 5. 收集其他可训练参数 (排除已收集的)
    all_collected_ids = cnn_param_ids | projection_param_ids | classifier_param_ids | llm_param_ids
    other_params = []
    for param in teacher.parameters():
        if param.requires_grad and id(param) not in all_collected_ids:
            other_params.append(param)

    # ✅ 创建优化器 (确保无重复)
    param_groups = [
        # CNN编码器
        {
            'params': cnn_params,
            'lr': args.lr_cnn,
            'weight_decay': args.wd_cnn
        },
        # CNN投影层
        {
            'params': projection_params,
            'lr': args.lr_projection,
            'weight_decay': args.wd_projection
        },
        # 分类头
        {
            'params': classifier_params,
            'lr': args.lr_classifier,
            'weight_decay': args.wd_classifier
        }
    ]

    # 如果有GPT-2参数,添加到参数组
    if llm_params:
        param_groups.append({
            'params': llm_params,
            'lr': 8e-4,  # 极小学习率
            'weight_decay': 1e-5
        })
        print(f"✓ GPT-2最后2层: {len(llm_params)}个参数, lr=xe-6")

    # 如果有其他参数,添加到参数组
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': args.lr_classifier,
            'weight_decay': args.wd_classifier
        })
        print(f"✓ 其他参数: {len(other_params)}个参数")

    optimizer = optim.AdamW(param_groups)

    print(f"\n✓ 使用分层学习率:")
    print(f"  CNN编码器: lr={args.lr_cnn}, wd={args.wd_cnn}")
    print(f"  CNN投影层: lr={args.lr_projection}, wd={args.wd_projection}")
    print(f"  分类头: lr={args.lr_classifier}, wd={args.wd_classifier}")
    if llm_params:
        print(f"  GPT-2最后2层: lr=3e-6, wd=1e-5")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )



    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 训练循环
    best_acc = 0.0
    patience_counter = 0 # ✅ 新增: 15个epoch无提升则停止

    epochs = args.epochs

    for epoch in range(1, epochs + 1):
        # 训练
        teacher.train()
        train_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}',
                    disable=False,  # 启用进度条
                    leave=True,  # 完成后保留
                   # file=sys.stdout,  # 输出到标准输出
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')  # 自定义格式，移除动态postfix

        for batch in pbar:
            pressure = batch['pressure'].to(device)
            vibration = batch['vibration'].to(device)
            labels = batch['label'].to(device)


            # 前向传播（无掩码）
            logits, _ = teacher(pressure, vibration, freq_mask=None)
            loss = criterion(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


        # 验证
        teacher.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                pressure = batch['pressure'].to(device)
                vibration = batch['vibration'].to(device)
                labels = batch['label'].to(device)

                logits, _ = teacher(pressure, vibration, freq_mask=None)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Acc: {100.0 * correct / total:.2f}%, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('../checkpoints/teacher', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, '../checkpoints/teacher/best_teacher.pt')
            print(f"✓ 保存最佳教师模型: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n早停: {args.patience}个epoch无提升")
                print(f"最佳验证准确率: {best_acc:.2f}%")
                break
        scheduler.step()

    print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    train_teacher()
