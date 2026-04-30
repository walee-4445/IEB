
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
                 seed: int = 42):  #

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.seed = seed
        # 归一化参数
        if mean is not None:
            self.mean = mean.reshape(1, 2).astype(np.float32)
            self.std = std.reshape(1, 2).astype(np.float32)
            self.need_normalize = True
        else:
            self.mean = np.zeros((1, 2), dtype=np.float32)
            self.std = np.ones((1, 2), dtype=np.float32)
            self.need_normalize = False


        self.use_fft = use_fft

        if self.use_fft and compute_mag_stats:
            print("正在计算幅度谱全局统计...")
            self.mag_mean, self.mag_std = self._compute_magnitude_stats(seed=self.seed)
            print(f"  压力幅度谱: mean={self.mag_mean[0]:.6f}, std={self.mag_std[0]:.6f}")
            print(f"  振动幅度谱: mean={self.mag_mean[1]:.6f}, std={self.mag_std[1]:.6f}")
        else:
            self.mag_mean = np.zeros(2, dtype=np.float32)
            self.mag_std = np.ones(2, dtype=np.float32)

    def _compute_magnitude_stats(self, seed=42):

        p_mags = []
        v_mags = []


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

        #  2. 根据use_fft决定输出时域还是频域
        if self.use_fft:

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

            pressure = seg[:, 0:1]  # (L, 1)
            vibration = seg[:, 1:2]

        # 3. 返回字典
        return {
            'pressure': torch.from_numpy(pressure.astype(np.float32)),
            'vibration': torch.from_numpy(vibration.astype(np.float32)),
            'label': torch.tensor(self.y[idx], dtype=torch.long)
        }
def load_presplit_data(file_path: str, step_size: int = 1024,train_max_rows: int = 614400,val_max_rows: int = 204800,test_max_rows: int = 204800, seed: int = 42):

    print("\n" + "=" * 60)
    print("加载预划分数据集（无泄露风险）")
    print("=" * 60)

    #按后缀区分 train / val / tes
    all_files = os.listdir(file_path)
    train_files = [f for f in all_files if f.endswith("-train.csv")]
    val_files = [f for f in all_files if f.endswith("-val.csv")]
    test_files = [f for f in all_files if f.endswith("-test.csv")]

    print(f"\n文件统计:")
    print(f"  训练文件: {len(train_files)} 个")
    print(f"  验证文件: {len(val_files)} 个")
    print(f"  测试文件: {len(test_files)} 个")


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
                print(f"  读取文件失败 {filename}: {e}")
                continue

        return pressure_data, vibration_data


    def slice_by_file(pressure_data, vibration_data):

        Data_P, Data_V, Labels = [], [], []

        # 故障类别映射
        key_mapping = {
            'A': 0, 'C': 1, 'G': 2, 'N': 3, 'B': 4,
            'B+G': 5, 'H': 6, 'H+N': 7, 'E': 8
        }

        for key in tqdm(pressure_data.keys(), desc="切分窗口"):
            p = pressure_data[key]
            v = vibration_data[key]


            match = re.search(r'电流-([A-Z+]+)-S', key)
            key_part = match.group(1) if match else 'UNKNOWN'
            class_num = key_mapping.get(key_part, -1)

            if class_num == -1:
                print(f"  未知故障类型: {key_part} in {key}")
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


    def standardize_with_train(Train, Val, Test):

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

            #
            if std[c] < 1e-6:
                print(f"    警告: std极小 ({std[c]:.2e}), 该传感器可能故障!")
                print(f"    将使用动态范围的1/6作为std")
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
                print(f"    归一化后均值偏离0: {train_norm_c.mean():.6f}")
            if abs(train_norm_c.std() - 1.0) > 0.1:
                print(f"    归一化后std偏离1: {train_norm_c.std():.6f}")

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

    # 归一化（仅用训练集统计）
    print("\n正在归一化（仅用训练集统计）...")
    Train_X_norm, Val_X_norm, Test_X_norm, mean, std = standardize_with_train(
        Train_X, Val_X, Test_X
    )

    print(f"\n归一化统计:")
    print(f"  压力: mean={mean[0]:.6f}, std={std[0]:.6f}")
    print(f"  振动: mean={mean[1]:.6f}, std={std[1]:.6f}")

    #
    print("\n正在打乱数据...")
    rng = np.random.RandomState(seed)
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
        'label_encoder': None  # 保持兼容性
    }

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


    train_dataset = MultiModalTimeSeriesDataset(
        X=data['train']['x'],
        y=data['train']['y'],
        mean=None,  # 数据已归一化
        std=None
    )

    print(f"\n✓ Dataset创建成功，样本数: {len(train_dataset)}")

    sample = train_dataset[0]
    print(f"  Pressure shape: {sample['pressure'].shape}")
    print(f"  Vibration shape: {sample['vibration'].shape}")
    print(f"  Label: {sample['label']}")


