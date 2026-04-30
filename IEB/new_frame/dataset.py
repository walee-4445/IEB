"""
dataset.py - 更新版
支持预划分的train/val/test数据集
避免数据泄露，保持向后兼容
"""
import torch
import numpy as np
import os
import re
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
class MultiModalTimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 mean: np.ndarray = None, std: np.ndarray = None,
                 compute_mag_stats: bool = False,
                 use_fft: bool = False,
                 seed: int = 42):  # 🆕 添加FFT开关
        """
        Args:
            X: (N, L, 2) 时域信号
            y: (N,) 标签
            mean: (2,) 归一化均值（None表示已归一化）
            std: (2,) 归一化标准差
            compute_mag_stats: 是否计算幅度谱统计
            use_fft: 是否执行FFT转换（True=频域输出，False=时域输出）
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.seed = seed  # ✅ 保存seed
        # 归一化参数
        if mean is not None:
            self.mean = mean.reshape(1, 2).astype(np.float32)
            self.std = std.reshape(1, 2).astype(np.float32)
            self.need_normalize = True
        else:
            self.mean = np.zeros((1, 2), dtype=np.float32)
            self.std = np.ones((1, 2), dtype=np.float32)
            self.need_normalize = False

        # 🆕 FFT模式
        self.use_fft = use_fft

        # 🆕 如果使用FFT，计算幅度谱统计
        if self.use_fft and compute_mag_stats:
            print("正在计算幅度谱全局统计...")
            self.mag_mean, self.mag_std = self._compute_magnitude_stats(seed=self.seed)
            print(f"  压力幅度谱: mean={self.mag_mean[0]:.6f}, std={self.mag_std[0]:.6f}")
            print(f"  振动幅度谱: mean={self.mag_mean[1]:.6f}, std={self.mag_std[1]:.6f}")
        else:
            self.mag_mean = np.zeros(2, dtype=np.float32)
            self.mag_std = np.ones(2, dtype=np.float32)

    def _compute_magnitude_stats(self, seed=42):
        """计算整个数据集的幅度谱统计（从MagnitudeSpectrumDataset复制）"""
        p_mags = []
        v_mags = []

        # 采样部分数据计算统计（避免内存溢出）
        sample_size = min(1000, len(self.X))
        rng = np.random.RandomState(42)  # 固定种子
        indices = rng.choice(len(self.X), sample_size, replace=False)

        for idx in indices:
            # 归一化（如果需要）
            if self.need_normalize:
                seg = (self.X[idx] - self.mean) / self.std
            else:
                seg = self.X[idx]

            # FFT转换
            p_fft = np.fft.rfft(seg[:, 0])
            v_fft = np.fft.rfft(seg[:, 1])

            # Log变换
            p_mag = np.log(np.abs(p_fft) + 1e-8)
            v_mag = np.log(np.abs(v_fft) + 1e-8)

            p_mags.append(p_mag)
            v_mags.append(v_mag)

        p_mags = np.array(p_mags)
        v_mags = np.array(v_mags)

        mag_mean = np.array([p_mags.mean(), v_mags.mean()], dtype=np.float32)
        mag_std = np.array([p_mags.std(), v_mags.std()], dtype=np.float32)

        return mag_mean, mag_std

    def set_mag_stats(self, mag_mean, mag_std):
        """设置幅度谱统计（用于验证/测试集）"""
        self.mag_mean = mag_mean
        self.mag_std = mag_std
        print(f"✓ 已设置幅度谱统计: P_mean={mag_mean[0]:.6f}, V_mean={mag_mean[1]:.6f}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 1. 归一化时域信号（如果需要）
        if self.need_normalize:
            seg = (self.X[idx] - self.mean) / self.std
        else:
            seg = self.X[idx]

        # 🆕 2. 根据use_fft决定输出时域还是频域
        if self.use_fft:
            # ============================================================
            # 频域模式：执行FFT转换（从MagnitudeSpectrumDataset复制）
            # ============================================================
            # 2.1 FFT转换
            p_fft = np.fft.rfft(seg[:, 0])  # (L,) → (F,) 复数
            v_fft = np.fft.rfft(seg[:, 1])

            # 2.2 提取幅度谱
            p_mag = np.abs(p_fft)  # (F,) 实数
            v_mag = np.abs(v_fft)

            # 2.3 Log变换
            p_mag = np.log(p_mag + 1e-8)
            v_mag = np.log(v_mag + 1e-8)

            # 2.4 频域归一化
            p_mag = (p_mag - self.mag_mean[0]) / (self.mag_std[0] + 1e-8)
            v_mag = (v_mag - self.mag_mean[1]) / (self.mag_std[1] + 1e-8)

            # 2.5 添加通道维度
            pressure = p_mag[:, np.newaxis]  # (F, 1)
            vibration = v_mag[:, np.newaxis]
        else:
            # ============================================================
            # 时域模式：直接输出时域信号
            # ============================================================
            pressure = seg[:, 0:1]  # (L, 1)
            vibration = seg[:, 1:2]

        # 3. 返回字典
        return {
            'pressure': torch.from_numpy(pressure.astype(np.float32)),
            'vibration': torch.from_numpy(vibration.astype(np.float32)),
            'label': torch.tensor(self.y[idx], dtype=torch.long)
        }
def load_presplit_data(file_path: str, step_size: int = 1024,train_max_rows: int = 614400,val_max_rows: int = 204800,test_max_rows: int = 204800, seed: int = 42):
    """
    ✅ 加载预划分的train/val/test数据（无泄露风险）

    数据格式要求：
    - 文件命名：电流-{故障类型}-S1-{train/val/test}.csv
    - 每个CSV：第1列=压力，第2列=振动

    Args:
        file_path: 数据目录路径
        step_size: 窗口大小
        train_max_rows: 训练集最大行数
        val_max_rows: 验证集最大行数
        test_max_rows: 测试集最大行数

    Returns:
        dict: {
            'train': {'x': (N, L, 2), 'y': (N,)},
            'val': {'x': (N, L, 2), 'y': (N,)},
            'test': {'x': (N, L, 2), 'y': (N,)},
            'norm': {'mean': (2,), 'std': (2,)},
            'label_encoder': None  # 保持兼容性
        }
    """
    print("\n" + "=" * 60)
    print("加载预划分数据集（无泄露风险）")
    print("=" * 60)

    # -------- ① 按后缀区分 train / val / test --------
    all_files = os.listdir(file_path)
    train_files = [f for f in all_files if f.endswith("-train.csv")]
    val_files = [f for f in all_files if f.endswith("-val.csv")]
    test_files = [f for f in all_files if f.endswith("-test.csv")]

    print(f"\n文件统计:")
    print(f"  训练文件: {len(train_files)} 个")
    print(f"  验证文件: {len(val_files)} 个")
    print(f"  测试文件: {len(test_files)} 个")

    # -------- ② 文件读取函数 --------
    def capture(path, filenames, max_rows=None):
        """读取CSV文件的前两列"""
        pressure_data, vibration_data = {}, {}

        for filename in tqdm(filenames, desc="读取文件"):
            file_full_path = os.path.join(path, filename)

            try:
                # 读取CSV，只取前两列
                file = np.genfromtxt(file_full_path,
                    delimiter=",",
                    dtype=float,
                    skip_header=0,
                    max_rows=max_rows
                )

                pressure = file[:, 0:1]  # 第1列：压力
                vibration = file[:, 1:2]  # 第2列：振动

                pressure_data[filename] = pressure
                vibration_data[filename] = vibration

            except Exception as e:
                print(f"⚠️  读取文件失败 {filename}: {e}")
                continue

        return pressure_data, vibration_data

    # -------- ③ 切片函数 --------
    def slice_by_file(pressure_data, vibration_data):
        """将连续信号切分为固定长度的窗口"""
        Data_P, Data_V, Labels = [], [], []

        # 故障类别映射
        key_mapping = {
            'A': 0, 'C': 1, 'G': 2, 'N': 3, 'B': 4,
            'B+G': 5, 'H': 6, 'H+N': 7, 'E': 8
        }

        for key in tqdm(pressure_data.keys(), desc="切分窗口"):
            p = pressure_data[key]
            v = vibration_data[key]

            # 提取故障类别
            match = re.search(r'电流-([A-Z+]+)-S', key)
            key_part = match.group(1) if match else 'UNKNOWN'
            class_num = key_mapping.get(key_part, -1)

            if class_num == -1:
                print(f"⚠️  未知故障类型: {key_part} in {key}")
                continue

            length = p.shape[0]
            num_segments = length // step_size
            start = 0

            for _ in range(num_segments):
                p_seg = p[start:start + step_size]
                v_seg = v[start:start + step_size]

                if p_seg.shape[0] == step_size:
                    Data_P.append(p_seg)
                    Data_V.append(v_seg)
                    Labels.append(class_num)

                start += step_size

        return np.array(Data_P), np.array(Data_V), np.array(Labels)

    # -------- ④ 归一化函数（仅用训练集统计） --------
    # def standardize_with_train(Train, Val, Test):
    #     """使用训练集统计量归一化所有数据集"""
    #     N, L, C = Train.shape
    #
    #     # 展平为2D
    #     Train_2d = Train.reshape(-1, C)
    #     Val_2d = Val.reshape(-1, C)
    #     Test_2d = Test.reshape(-1, C)
    #
    #     # ✅ 仅用训练集拟合Scaler
    #     scaler = StandardScaler().fit(Train_2d)
    #
    #     # 应用到所有集合
    #     Train_2d = scaler.transform(Train_2d)
    #     Val_2d = scaler.transform(Val_2d)
    #     Test_2d = scaler.transform(Test_2d)
    #
    #     # 恢复形状
    #     Train_norm = Train_2d.reshape(N, L, C)
    #     Val_norm = Val_2d.reshape(Val.shape[0], L, C)
    #     Test_norm = Test_2d.reshape(Test.shape[0], L, C)
    #
    #     # 提取均值和标准差
    #     mean = scaler.mean_.astype(np.float32)
    #     std = scaler.scale_.astype(np.float32)
    #
    #     return Train_norm, Val_norm, Test_norm, mean, std

    # -------- ⑤ 分别处理 train / val / test --------
    def standardize_with_train(Train, Val, Test):
        """
        ✅ Per-Modality归一化 (科学方案)

        原理:
        - 每个传感器在自己的物理尺度上归一化
        - 归一化后所有模态都变为均值0、标准差1的标准正态分布
        - 消除物理单位差异,保留相对变化信息

        Args:
            Train: (N, L, 2) 训练集 [压力(A), 振动(V)]
            Val: (N, L, 2) 验证集
            Test: (N, L, 2) 测试集

        Returns:
            归一化后的数据 + 统计参数
        """
        N, L, C = Train.shape

        # 初始化
        Train_norm = np.zeros_like(Train, dtype=np.float32)
        Val_norm = np.zeros_like(Val, dtype=np.float32)
        Test_norm = np.zeros_like(Test, dtype=np.float32)

        mean = np.zeros(C, dtype=np.float32)
        std = np.zeros(C, dtype=np.float32)

        print("\n" + "=" * 70)
        print("Per-Modality归一化 (科学方案)")
        print("=" * 70)

        modality_names = ["压力传感器(电流A)", "振动传感器(电压V)"]

        for c in range(C):
            print(f"\n【{modality_names[c]}】")

            # 提取单模态数据
            train_c = Train[:, :, c].flatten()  # 展平为1D
            val_c = Val[:, :, c].flatten()
            test_c = Test[:, :, c].flatten()

            # 计算训练集统计量
            mean[c] = train_c.mean()
            std[c] = train_c.std()

            # 打印原始统计
            print(f"  原始数据统计:")
            print(f"    均值: {mean[c]:.6f}")
            print(f"    标准差: {std[c]:.6f}")
            print(f"    范围: [{train_c.min():.6f}, {train_c.max():.6f}]")
            print(f"    动态范围: {train_c.max() - train_c.min():.6f}")

            # ✅ 关键: 检测并处理异常std
            if std[c] < 1e-6:
                print(f"  ⚠️  警告: std极小 ({std[c]:.2e}), 该传感器可能故障!")
                print(f"  ⚠️  将使用动态范围的1/6作为std")
                data_range = train_c.max() - train_c.min()
                std[c] = max(data_range / 6.0, 1e-6)  # 6-sigma原则

            # 归一化 (Z-score标准化)
            Train_norm[:, :, c] = ((Train[:, :, c] - mean[c]) / std[c]).astype(np.float32)
            Val_norm[:, :, c] = ((Val[:, :, c] - mean[c]) / std[c]).astype(np.float32)
            Test_norm[:, :, c] = ((Test[:, :, c] - mean[c]) / std[c]).astype(np.float32)

            # 验证归一化效果
            train_norm_c = Train_norm[:, :, c].flatten()
            print(f"  归一化后统计:")
            print(f"    均值: {train_norm_c.mean():.6f} (期望≈0)")
            print(f"    标准差: {train_norm_c.std():.6f} (期望≈1)")
            print(f"    范围: [{train_norm_c.min():.2f}, {train_norm_c.max():.2f}]")

            # 健康检查
            if abs(train_norm_c.mean()) > 0.01:
                print(f"  ⚠️  归一化后均值偏离0: {train_norm_c.mean():.6f}")
            if abs(train_norm_c.std() - 1.0) > 0.1:
                print(f"  ⚠️  归一化后std偏离1: {train_norm_c.std():.6f}")

        print("\n" + "=" * 70)
        print("归一化完成:")
        print(f"  压力: mean={mean[0]:.6f}, std={std[0]:.6f}")
        print(f"  振动: mean={mean[1]:.6f}, std={std[1]:.6f}")
        print(f"  尺度比: {std[0] / std[1]:.6f} (原始物理尺度)")
        print("=" * 70 + "\n")

        return Train_norm, Val_norm, Test_norm, mean, std

    print("\n正在读取训练集...")
    train_P_raw, train_V_raw = capture(file_path, train_files, max_rows=train_max_rows)

    print("正在读取验证集...")
    val_P_raw, val_V_raw = capture(file_path, val_files, max_rows=val_max_rows)

    print("正在读取测试集...")
    test_P_raw, test_V_raw = capture(file_path, test_files, max_rows=test_max_rows)

    # 切分窗口
    print("\n正在切分窗口...")
    Train_P, Train_V, Train_Y = slice_by_file(train_P_raw, train_V_raw)
    Val_P, Val_V, Val_Y = slice_by_file(val_P_raw, val_V_raw)
    Test_P, Test_V, Test_Y = slice_by_file(test_P_raw, test_V_raw)

    # 拼接压力和振动
    Train_X = np.concatenate((Train_P, Train_V), axis=2)  # (N, L, 2)
    Val_X = np.concatenate((Val_P, Val_V), axis=2)
    Test_X = np.concatenate((Test_P, Test_V), axis=2)

    print(f"\n切分后数据形状:")
    print(f"  训练集: {Train_X.shape}")
    print(f"  验证集: {Val_X.shape}")
    print(f"  测试集: {Test_X.shape}")

    # ✅ 归一化（仅用训练集统计）
    print("\n正在归一化（仅用训练集统计）...")
    Train_X_norm, Val_X_norm, Test_X_norm, mean, std = standardize_with_train(
        Train_X, Val_X, Test_X
    )

    print(f"\n归一化统计:")
    print(f"  压力: mean={mean[0]:.6f}, std={std[0]:.6f}")
    print(f"  振动: mean={mean[1]:.6f}, std={std[1]:.6f}")

    # ✅ 随机打乱（仅打乱各自集合内部）
    print("\n正在打乱数据...")
    rng = np.random.RandomState(seed)  # 创建独立的随机数生成器
    idx_train =  rng.permutation(Train_X_norm.shape[0])
    Train_X_norm = Train_X_norm[idx_train]
    Train_Y = Train_Y[idx_train]

    idx_val = rng.permutation(Val_X_norm.shape[0])
    Val_X_norm = Val_X_norm[idx_val]
    Val_Y = Val_Y[idx_val]

    idx_test = rng.permutation(Test_X_norm.shape[0])
    Test_X_norm = Test_X_norm[idx_test]
    Test_Y = Test_Y[idx_test]

    print("\n✓ 数据加载完成（无泄露风险）")
    print("=" * 60 + "\n")

    # 返回与data_prep.py兼容的格式
    return {
        'train': {'x': Train_X_norm, 'y': Train_Y},
        'val': {'x': Val_X_norm, 'y': Val_Y},
        'test': {'x': Test_X_norm, 'y': Test_Y},
        'norm': {
            'mean': mean,
            'std': std
        },
        'label_encoder': None  # 保持兼容性，实际不需要
    }

# ============================================================
# 向后兼容：保留原有接口
# ============================================================
def check_numpy_dataset(data):
    """检查数据集（保持向后兼容）"""

    def stats_split(name, X, y):
        print(f'[{name}] X shape={X.shape}, y shape={y.shape}')
        if X.size > 0:
            has_nan = np.isnan(X).any()
            finite = np.isfinite(X).all()

            if has_nan:
                modal_means = np.nanmean(X, axis=(0, 1))
                X[np.isnan(X)] = np.take(modal_means, np.where(np.isnan(X))[2])
                finite = np.isfinite(X).all()

            x_min = np.nanmin(X)
            x_max = np.nanmax(X)

        if y.size > 0:
            uniq, cnt = np.unique(y, return_counts=True)

    stats_split('train', data['train']['x'], data['train']['y'])
    stats_split('val', data['val']['x'], data['val']['y'])
    stats_split('test', data['test']['x'], data['test']['y'])

    mean, std = data['norm']['mean'], data['norm']['std']

    eps_floor = 1e-3
    if np.any(std < eps_floor):
        data['norm']['std'] = np.maximum(std, eps_floor).astype(np.float32)
if __name__ == '__main__':
    # 测试新的数据加载
    print("测试预划分数据加载...")

    data_path = "./pressure_vibration1_S2_02"  #

    data = load_presplit_data(
        file_path=data_path,
        step_size=1024,
        train_max_rows=614400,
        val_max_rows=204800,
        test_max_rows=204800
    )

    print("\n数据集统计:")
    print(f"  训练集: {data['train']['x'].shape}")
    print(f"  验证集: {data['val']['x'].shape}")
    print(f"  测试集: {data['test']['x'].shape}")
    print(f"  类别数: {len(np.unique(data['train']['y']))}")

    # 创建Dataset
    train_dataset = MultiModalTimeSeriesDataset(
        X=data['train']['x'],
        y=data['train']['y'],
        mean=None,  # 数据已归一化
        std=None
    )

    print(f"\n✓ Dataset创建成功，样本数: {len(train_dataset)}")

    # 测试__getitem__
    sample = train_dataset[0]
    print(f"  Pressure shape: {sample['pressure'].shape}")
    print(f"  Vibration shape: {sample['vibration'].shape}")
    print(f"  Label: {sample['label']}")

# # dataset.py
# import torch
# import numpy as np
# from torch.utils.data import Dataset
#
# class MultiModalTimeSeriesDataset(Dataset):
#     def __init__(self, X: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray,
#              compute_mag_stats: bool = False):
#         """
#         X: (N, 5120, 2)  两列：pressure, vibration
#         y: (N,)
#         y: (N,)
#         mean/std: (2,)   模态级统计
#         """
#         self.mag_std = None
#         self.mag_mean = None
#         assert isinstance(X, np.ndarray) and X.ndim == 3 and X.shape[2] == 2, f"X shape expected (N,W,2), got {X.shape}"
#         assert isinstance(y, np.ndarray) and y.ndim == 1 and len(y) == len(X), "y must be (N,) and match X"
#         assert isinstance(mean, np.ndarray) and mean.shape == (2,), "mean must be shape (2,)"
#         assert isinstance(std, np.ndarray) and std.shape == (2,), "std must be shape (2,)"
#
#         self.X = X
#         self.y = y
#         self.mean = mean
#         self.std = std
#         self.X = X.astype(np.float32, copy=False)
#         self.y = y.astype(np.int64, copy=False)
#         self.mean = mean.astype(np.float32, copy=False)
#         self.std = std.astype(np.float32, copy=False)
#
#         # 预先 reshape，便于广播 (W,2) <- (1,2)
#         self._mean_row = self.mean.reshape(1, 2)
#         self._std_row = self.std.reshape(1, 2)
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         seg = self.X[idx]  # (W, 2)
#         seg = (seg - self._mean_row) / self._std_row
#         #print(f"归归一化后数据范围：{seg.min().item()} ~ {seg.max().item()}")  # 正常应在 [-5, 5] 内
#         pressure = seg[:, 0:1]  # (W,1)
#         vibration = seg[:, 1:2] # (W,1)
#         label = int(self.y[idx])             # 标量整数       # 转为 Conv1d 期望的 (B,C,L) 形状在模型中做，这里保留 (W,1)
#         sample = {
#             'pressure': torch.from_numpy(pressure.astype(np.float32)),  # (W,1)
#             'vibration': torch.from_numpy(vibration.astype(np.float32)),# (W,1)
#             'label': torch.tensor(label, dtype=torch.long)
#         }
#         return sample
#
#     def set_mag_stats(self, mag_mean, mag_std):
#         """
#         设置幅度谱统计（用于验证/测试集）
#
#         Args:
#             mag_mean: (2,) 训练集的幅度谱均值
#             mag_std: (2,) 训练集的幅度谱标准差
#         """
#         self.mag_mean = mag_mean
#         self.mag_std = mag_std
#         print(f"✓ 已设置幅度谱统计: P_mean={mag_mean[0]:.6f}, V_mean={mag_mean[1]:.6f}")


# class MultiModalTimeSeriesDataset(Dataset):
#     """
#     多模态时序数据集（保持向后兼容）
#
#     支持两种数据格式：
#     1. 旧格式：传入X, y, mean, std（用于data_prep.py）
#     2. 新格式：传入预处理好的数据（用于新的数据加载流程）
#     """
#
#     def __init__(self, X: np.ndarray, y: np.ndarray,
#                  mean: np.ndarray = None, std: np.ndarray = None,
#                  compute_mag_stats: bool = False):
#         """
#         Args:
#             X: (N, L, 2) 时域信号 [pressure, vibration]
#             y: (N,) 标签
#             mean: (2,) 归一化均值（可选）
#             std: (2,) 归一化标准差（可选）
#             compute_mag_stats: 是否计算幅度谱统计
#
#         """
#         self.mag_std = None
#         self.mag_mean = None
#
#         assert isinstance(X, np.ndarray) and X.ndim == 3 and X.shape[2] == 2, \
#             f"X shape expected (N,L,2), got {X.shape}"
#         assert isinstance(y, np.ndarray) and y.ndim == 1 and len(y) == len(X), \
#             "y must be (N,) and match X"
#
#         self.X = X.astype(np.float32, copy=False)
#         self.y = y.astype(np.int64, copy=False)
#
#         # 归一化参数（如果提供）
#         if mean is not None and std is not None:
#             assert isinstance(mean, np.ndarray) and mean.shape == (2,), "mean must be shape (2,)"
#             assert isinstance(std, np.ndarray) and std.shape == (2,), "std must be shape (2,)"
#             self.mean = mean.astype(np.float32, copy=False)
#             self.std = std.astype(np.float32, copy=False)
#             self._mean_row = self.mean.reshape(1, 2)
#             self._std_row = self.std.reshape(1, 2)
#         else:
#             # 如果没有提供，假设数据已归一化
#             self.mean = None
#             self.std = None
#             self._mean_row = None
#             self._std_row = None
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         seg = self.X[idx]  # (L, 2)
#
#         # 如果提供了归一化参数，则应用归一化
#         if self._mean_row is not None and self._std_row is not None:
#             seg = (seg - self._mean_row) / self._std_row
#
#         pressure = seg[:, 0:1]  # (L, 1)
#         vibration = seg[:, 1:2]  # (L, 1)
#         label = int(self.y[idx])
#
#         sample = {
#             'pressure': torch.from_numpy(pressure.astype(np.float32)),
#             'vibration': torch.from_numpy(vibration.astype(np.float32)),
#             'label': torch.tensor(label, dtype=torch.long)
#         }
#         return sample
#
#     def set_mag_stats(self, mag_mean, mag_std):
#         """设置幅度谱统计（用于验证/测试集）"""
#         self.mag_mean = mag_mean
#         self.mag_std = mag_std
#         print(f"✓ 已设置幅度谱统计: P_mean={mag_mean[0]:.6f}, V_mean={mag_mean[1]:.6f}")
# dataset.py
