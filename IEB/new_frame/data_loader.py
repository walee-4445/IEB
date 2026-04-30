
"""
统一数据加载接口 - 整合datapre到GMD项目
将datapre的数据预处理和Dataset封装为统一接口供GMD使用
支持时域和频域（幅度谱）两种数据加载模式
"""
from torch.utils.data import DataLoader
from data_prep import prepare_dataset, check_numpy_dataset
from dataset import MultiModalTimeSeriesDataset
from magnitude_dataset import MagnitudeSpectrumDataset
import numpy as np
import random
import torch
# data_loader.py

from dataset import (
    MultiModalTimeSeriesDataset,
    load_presplit_data,  # 新增
    check_numpy_dataset
)


class SimpleLabelEncoder:
    """简单的标签编码器，用于预划分数据"""
    def __init__(self, classes):
        self.classes_ = classes

def get_dataloaders_presplit(config, batch_size=None, mode='student', use_fft=False, seed=42):  # 🆕 添加FFT开关
    """
    ✅ 使用预划分数据（无泄露风险）

    Args:
        config: 配置对象
        batch_size: 批次大小
        mode: 模式（'teacher', 'student', 'nfm'）
        use_fft: 是否执行FFT转换（True=频域输出，False=时域输出）

    适用于已经划分好train/val/test的数据集
    """
    print(f"正在从 {config.data_root} 加载预划分数据...")
    if use_fft:
        print("  模式: 频域（幅度谱）")
    else:
        print("  模式: 时域")

    # 加载数据
    data = load_presplit_data(
        file_path=config.data_root,
        step_size=config.window_size,
        train_max_rows=614400,
        val_max_rows=204800,
        test_max_rows=204800,
        seed=seed  # ✅ 传递seed
    )

    # 数据检查
    check_numpy_dataset(data)

    # 确定batch_size
    if batch_size is None:
        if mode == 'teacher':
            batch_size = config.teacher_batch_size
        elif mode == 'student':
            batch_size = config.student_batch_size
        else:
            batch_size = 32

    # 创建Dataset（启用FFT模式）
    train_dataset = MultiModalTimeSeriesDataset(
        X=data['train']['x'],
        y=data['train']['y'],
        mean=None,  # 已归一化
        std=None,
        compute_mag_stats=use_fft,
        use_fft=use_fft,
        seed=seed  #
    )

    val_dataset = MultiModalTimeSeriesDataset(
        X=data['val']['x'],
        y=data['val']['y'],
        mean=None,
        std=None,
        compute_mag_stats=False,  # 验证集不计算统计
        use_fft=use_fft
    )

    test_dataset = MultiModalTimeSeriesDataset(
        X=data['test']['x'],
        y=data['test']['y'],
        mean=None,
        std=None,
        compute_mag_stats=False,
        use_fft=use_fft
    )

    train_dataset.time_mean = data["norm"]["mean"]
    train_dataset.time_std = data["norm"]["std"]

    val_dataset.time_mean = data["norm"]["mean"]
    val_dataset.time_std = data["norm"]["std"]

    test_dataset.time_mean = data["norm"]["mean"]
    test_dataset.time_std = data["norm"]["std"]


    # 如果使用FFT，设置验证/测试集的幅度谱统计
    if use_fft:
        val_dataset.set_mag_stats(
            train_dataset.mag_mean,
            train_dataset.mag_std
        )
        test_dataset.set_mag_stats(
            train_dataset.mag_mean,
            train_dataset.mag_std
        )

    # 创建DataLoader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    # 打印数据集信息（包含频域信息）
    print("\n" + "=" * 60)
    print("数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} batches")
    print(f"  验证集: {len(val_dataset)} 样本, {len(val_loader)} batches")
    print(f"  测试集: {len(test_dataset)} 样本, {len(test_loader)} batches")
    print(f"  批次大小: {batch_size}")

    if use_fft:
        F_bins = config.window_size // 2 + 1
        print(f"  频域维度: {F_bins} (幅度谱)")
        print(f"  输出格式: (B, {F_bins}, 1)")
    else:
        print(f"  时域维度: {config.window_size}")
        print(f"  输出格式: (B, {config.window_size}, 1)")

    print("=" * 60 + "\n")
    unique_classes = np.unique(data['train']['y'])
    num_classes = len(unique_classes)

    # 创建一个简单的类别映射对象
    class SimpleLabelEncoder:
        def __init__(self, classes):
            self.classes_ = classes

    label_encoder = SimpleLabelEncoder(unique_classes)
    return train_loader, val_loader, test_loader, label_encoder  # label_encoder=None
def get_dataloaders(config, batch_size=None, mode='teacher'):

    print(f"正在从 {config.data_root} 加载数据...")
    data = prepare_dataset(
        root_dir=config.data_root,
        label_regex='',  # 使用首字母规则
        window_size=config.window_size,
        stride=config.stride,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=config.seed
    )

    # 2. 数据检查和预处理
    check_numpy_dataset(data)

    # 3. 确定batch_size
    if batch_size is None:
        if mode == 'teacher':
            batch_size = config.teacher_batch_size
        elif mode == 'student':
            batch_size = config.student_batch_size
        elif mode == 'nfm':
            batch_size = config.nfm_batch_size
        else:
            batch_size = 32  # 默认值

    train_dataset = MultiModalTimeSeriesDataset(
        X=data['train']['x'],
        y=data['train']['y'],
        mean=data['norm']['mean'],
        std=data['norm']['std'],
        compute_mag_stats = True
    )

    val_dataset = MultiModalTimeSeriesDataset(
        X=data['val']['x'],
        y=data['val']['y'],
        mean=data['norm']['mean'],
        std=data['norm']['std']
        ,compute_mag_stats=False  #
    )
    val_dataset.set_mag_stats(
        train_dataset.mag_mean,
        train_dataset.mag_std
    )  #使用训练集统计

    test_dataset = MultiModalTimeSeriesDataset(
        X=data['test']['x'],
        y=data['test']['y'],
        mean=data['norm']['mean'],
        std=data['norm']['std'],compute_mag_stats=False
    )
    test_dataset.mag_mean = train_dataset.mag_mean
    test_dataset.mag_std = train_dataset.mag_std

    def worker_init_fn(worker_id):

        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # 训练集打乱
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False,
        drop_last=True,  # 避免最后一个batch过小
        worker_init_fn=worker_init_fn,  # ✅ 添加这个
        generator=torch.Generator().manual_seed(42)  # ✅ 添加这个
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )


    # 6. 打印数据集信息
    print("\n" + "=" * 60)
    print("数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} batches")
    print(f"  验证集: {len(val_dataset)} 样本, {len(val_loader)} batches")
    print(f"  测试集: {len(test_dataset)} 样本, {len(test_loader)} batches")
    print(f"  类别数: {len(data['label_encoder'].classes_)}")
    print(f"  类别标签: {list(data['label_encoder'].classes_)}")
    print(f"  批次大小: {batch_size}")
    print(f"  归一化统计: mean={data['norm']['mean']}, std={data['norm']['std']}")
    print("=" * 60 + "\n")

    return train_loader, val_loader, test_loader, data['label_encoder']


def get_data_info(config):
    """
    获取数据集基本信息（不创建DataLoader）

    Args:
        config: Config对象

    Returns:
        dict: 包含数据集统计信息
    """
    data = prepare_dataset(
        root_dir=config.data_root,
        label_regex='',
        window_size=config.window_size,
        stride=config.stride,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=config.seed
    )

    return {
        'num_train': len(data['train']['x']),
        'num_val': len(data['val']['x']),
        'num_test': len(data['test']['x']),
        'num_classes': len(data['label_encoder'].classes_),
        'classes': list(data['label_encoder'].classes_),
        'window_size': config.window_size,
        'num_modalities': config.num_modalities,
        'norm_mean': data['norm']['mean'],
        'norm_std': data['norm']['std']
    }


def get_magnitude_dataloaders(config, batch_size=None, mode='student'):

    #准备数据集
    print(f"正在从 {config.data_root} 加载数据（幅度谱模式）...")
    data = prepare_dataset(
        root_dir=config.data_root,
        label_regex='',  # 使用首字母规则
        window_size=config.window_size,
        stride=config.stride,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=config.seed
    )
    
    # 2. 数据检查和预处理
    check_numpy_dataset(data)
    
    # 3. 确定batch_size
    if batch_size is None:
        if mode == 'teacher':
            batch_size = config.teacher_batch_size
        elif mode == 'student':
            batch_size = config.student_batch_size
        elif mode == 'nfm':
            batch_size = config.nfm_batch_size
        else:
            batch_size = 32  # 默认值
    
    # 4. 创建PyTorch Dataset
    train_dataset = MagnitudeSpectrumDataset(
        X=data['train']['x'],
        y=data['train']['y'],
        mean=data['norm']['mean'],
        std=data['norm']['std'],
        compute_mag_stats=True
    )
    
    #  验证集：使用训练集统计
    val_dataset = MagnitudeSpectrumDataset(
        X=data['val']['x'],
        y=data['val']['y'],
        mean=data['norm']['mean'],
        std=data['norm']['std'],
        compute_mag_stats=False
    )
    val_dataset.set_mag_stats(
        train_dataset.mag_mean,
        train_dataset.mag_std
    )  # 使用训练集统计
    
    # 测试集：使用训练集统计
    test_dataset = MagnitudeSpectrumDataset(
        X=data['test']['x'],
        y=data['test']['y'],
        mean=data['norm']['mean'],
        std=data['norm']['std'],
        compute_mag_stats=False
    )
    test_dataset.set_mag_stats(
        train_dataset.mag_mean,
        train_dataset.mag_std
    )
    
    # 5. 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    
    # 6. 打印数据集信息
    F_bins = config.window_size // 2 + 1
    print("\n" + "=" * 60)
    print("幅度谱数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} batches")
    print(f"  验证集: {len(val_dataset)} 样本, {len(val_loader)} batches")
    print(f"  测试集: {len(test_dataset)} 样本, {len(test_loader)} batches")
    print(f"  类别数: {len(data['label_encoder'].classes_)}")
    print(f"  类别标签: {list(data['label_encoder'].classes_)}")
    print(f"  批次大小: {batch_size}")
    print(f"  频域维度: {F_bins} (幅度谱)")
    print(f"  归一化统计: mean={data['norm']['mean']}, std={data['norm']['std']}")
    print("=" * 60 + "\n")
    
    return train_loader, val_loader, test_loader, data['label_encoder']
