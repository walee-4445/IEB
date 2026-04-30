import os
#import re
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from typing import List, Dict, Tuple
from collections import Counter

def load_csv_series(root_dir: str, pattern: str = "*.csv") -> List[Tuple[str, np.ndarray]]:
    files = sorted(glob.glob(os.path.join(root_dir, pattern)))
    series = []
    for fp in files:
        df = pd.read_csv(fp)

        x = df.values.astype(np.float32)
        assert x.ndim == 2 and x.shape[1] >= 2, f"Bad shape in {fp}: {x.shape}"
        x = x[:, :2]  # 只取前两列
        series.append((fp, x))
    return series

def parse_label_from_name(filepath: str, regex: str) -> str:
    name = os.path.splitext(os.path.basename(filepath))[0]
    label_char = None
    if len(name) > 0 and name[0].isalpha():
        label_char = name[0]
    else:
        for ch in name:
            if ch.isalpha():
                label_char = ch
                break
    if label_char is None:
        raise ValueError(f"Cannot parse alpha label from filename: {filepath}")
    return label_char.upper()

def windowing(x: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    T = x.shape[0]
    if T < window_size:
        return np.empty((0, window_size, x.shape[1]), dtype=x.dtype)
    starts = np.arange(0, T - window_size + 1, stride)
    windows = np.stack([x[s:s+window_size] for s in starts], axis=0)
    return windows  # (Nw, window_size, 2) 窗口数量 Nw, 窗口大小

def prepare_dataset(
    root_dir: str,
    label_regex: str,
    window_size: int = 5120,
    stride: int = 2560,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, Dict[str, np.ndarray]]: # 用于代码的运行逻辑没有影响，但能提高代码的可读性和可维护性 类型提示语法

    # 1) 读取所有文件
    series = load_csv_series(root_dir)

    # 2) 对每个文件切窗并汇总为“窗口级样本池”
    X_list, y_list = [], []
    for fp, x in series:
        label = parse_label_from_name(fp, label_regex)  # 新规则：首字母标签
        Xw = windowing(x, window_size, stride)          # (Nw, W, 2)
        if len(Xw) > 0:
            X_list.append(Xw)
            y_list.append(np.array([label] * len(Xw), dtype=object))
    if len(X_list) == 0:
        # 无有效窗口
        empty_x = np.empty((0, window_size, 2), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)

        return {
            'train': {'x': empty_x, 'y': empty_y},
            'val':   {'x': empty_x, 'y': empty_y},
            'test':  {'x': empty_x, 'y': empty_y},
            'norm':  {'mean': np.zeros(2, dtype=np.float32), 'std': np.ones(2, dtype=np.float32)},
            'label_encoder': LabelEncoder()
        }

    X_all = np.concatenate(X_list, axis=0).astype(np.float32)  # (N_total, W, 2)
    y_raw_all = np.concatenate(y_list, axis=0)                 # (N_total,), str/object

    N_total = len(X_all)
    assert 0 < test_ratio < 1 and 0 < val_ratio < 1 and (val_ratio + test_ratio) < 1.0, \
        "val_ratio 和 test_ratio 必须在 (0,1) 且和小于 1。"

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)

    idx_trainval, idx_test = next(sss1.split(X_all, y_raw_all))

    X_trainval, y_trainval = X_all[idx_trainval], y_raw_all[idx_trainval]
    X_test,      y_test_raw = X_all[idx_test],      y_raw_all[idx_test]

    val_size_adj = val_ratio / (1.0 - test_ratio)  # 在 trainval 子集中所占比例
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adj, random_state=seed)
    idx_train, idx_val = next(sss2.split(X_trainval, y_trainval))

    X_train, y_train_raw = X_trainval[idx_train], y_trainval[idx_train] #
    X_val,   y_val_raw   = X_trainval[idx_val],   y_trainval[idx_val]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val   = le.transform(y_val_raw)   if len(y_val_raw)   > 0 else np.empty((0,), dtype=np.int64)
    y_test  = le.transform(y_test_raw)  if len(y_test_raw)  > 0 else np.empty((0,), dtype=np.int64)

    eps = 1e-6
    min_std_threshold = 0.01
    
    print(f"第一次：X_train 样本数: {len(X_train)}, 是否为空: {len(X_train) == 0}")
    

    mean = np.nanmean(X_train.reshape(-1, 2), axis=0)
    std = np.nanstd(X_train.reshape(-1, 2), axis=0)
    

    norm_method = np.array(['zscore', 'zscore'])  # 默认使用Z-score
    
    for i in range(2):
        channel_name = 'Pressure' if i == 0 else 'Vibration'
        
        if std[i] < min_std_threshold:
            print(f"  {channel_name} channel: std={std[i]:.6f} too small, using Min-Max normalization")
            

            channel_data = X_train[:, :, i]
            data_min = np.nanmin(channel_data)
            data_max = np.nanmax(channel_data)
            data_range = data_max - data_min
            
            if data_range > eps:

                mean[i] = data_min + data_range / 2.0  # 中点
                std[i] = data_range / 20.0  # 半范围
                norm_method[i] = 'minmax'
                print(f"   → {channel_name}: range=[{data_min:.6f}, {data_max:.6f}], "
                      f"normalized to [-1, 1]")
            else:

                print(f"     {channel_name}: data is constant, cannot normalize")
                mean[i] = 0.0
                std[i] = 1.0
                norm_method[i] = 'constant'
        else:
            print(f"✓ {channel_name} channel: std={std[i]:.6f} normal, using Z-score normalization")
    
    # 添加eps避免除零
    std = std + eps
    
    data = {
        'train': {'x': X_train, 'y': y_train},
        'val':   {'x': X_val,   'y': y_val},
        'test':  {'x': X_test,  'y': y_test},
        'norm':  {
            'mean': mean.astype(np.float32),
            'std': std.astype(np.float32),
            'method': norm_method  # 记录每个通道的归一化方法
        },
        'label_encoder': le
    }

    return data


def check_numpy_dataset(data):
    def stats_split(name, X, y):
        print(f'[{name}] X shape={X.shape}, y shape={y.shape}')
        if X.size > 0:
            # 检查是否存在NaN（单独判断NaN，避免inf干扰）
            has_nan = np.isnan(X).any()
            # 检查是否所有元素都是有限值（无NaN和inf）
            finite = np.isfinite(X).all()

            # 如果存在NaN，按模态（最后一维）计算均值并填补
            if has_nan:

                modal_means = np.nanmean(X, axis=(0, 1))  # 对样本数和时间步求平均

                X[np.isnan(X)] = np.take(modal_means, np.where(np.isnan(X))[2])

                after_nan = np.isnan(X).any()


                finite = np.isfinite(X).all()

            #print(f'  finite(all)? {finite}')  # 打印最终有限值状态
            x_min = np.nanmin(X)
            x_max = np.nanmax(X)
           # print(f'  X[min,max]=({x_min:.6g},{x_max:.6g})')

        if y.size > 0:
            uniq, cnt = np.unique(y, return_counts=True)
           # print(f'  labels: {dict(zip(uniq.tolist(), cnt.tolist()))}')

   # print('==== Dataset NP-level Check ====')
    stats_split('train', data['train']['x'], data['train']['y'])
    stats_split('val', data['val']['x'], data['val']['y'])
    stats_split('test', data['test']['x'], data['test']['y'])
    mean, std = data['norm']['mean'], data['norm']['std']
    eps_floor = 1e-3
    if np.any(std < eps_floor):
       # print(f'WARN: std too small, applying floor {eps_floor}')
        data['norm']['std'] = np.maximum(std, eps_floor).astype(np.float32)

