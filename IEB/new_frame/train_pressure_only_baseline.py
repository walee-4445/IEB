"""
Pressure-only baseline training script
目标：
1. 上游数据处理尽量与原项目保持一致
2. 仅使用 pressure 幅值谱做分类
3. 用于验证 pressure 模态本身的表征能力

复用原项目：
- Config
- get_dataloaders_presplit(..., use_fft=True)
- MagnitudeEncoder
"""

import os
import json
import argparse
import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from config import Config
from data_loader import get_dataloaders_presplit
from magnitude_encoder import MagnitudeEncoder


# ============================================================
# logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# utils
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# model
# ============================================================
class PressureOnlyMagnitudeBaseline(nn.Module):
    """
    纯压力单模态基线
    输入: pressure magnitude spectrum, shape = (B, F, 1)
    输出: logits, shape = (B, num_classes)

    说明：
    - 上游频域输入格式与原 student/teacher 一致
    - 编码器直接复用原项目的 MagnitudeEncoder
    - 仅保留 pressure 路径，不使用 vibration / PV-NFM / fusion
    """

    def __init__(
        self,
        F_bins: int,
        num_classes: int,
        feature_dim: int = 256,
        encoder_channels=None,
        dropout: float = 0.2
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [32, 64, 128]

        self.pressure_encoder = MagnitudeEncoder(
            F_bins=F_bins,
            channels=encoder_channels,
            feature_dim=feature_dim
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, pressure: torch.Tensor):
        feat = self.pressure_encoder(pressure)     # (B, feature_dim)
        logits = self.classifier(feat)             # (B, num_classes)
        return logits, feat


# ============================================================
# eval
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in loader:
        pressure = batch["vibration"].to(device)   # only pressure
        labels = batch["label"].to(device)

        logits, _ = model(pressure)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": acc * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# train one epoch
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in loader:
        pressure = batch["vibration"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, _ = model(pressure)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": acc * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100
    }


# ============================================================
# args
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train pressure-only baseline with the same upstream pipeline"
    )

    # data
    parser.add_argument("--data_dir", type=str,
                        default="./new_frame/pressure_vibration1_S2_02",
                        help="数据目录")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    # model
    parser.add_argument("--seq_len", type=int, default=2560)
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)

    # train
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str,
                        default="./checkpoints/pressure_only_baseline")

    return parser.parse_args()


# ============================================================
# main train
# ============================================================
def train(args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    # --------------------------------------------------------
    # 1) data loader
    # 与原项目保持一致：同样的预划分、同样的FFT/log/归一化逻辑
    # --------------------------------------------------------
    temp_config = Config()
    temp_config.data_root = args.data_dir
    temp_config.window_size = args.seq_len
    temp_config.num_workers = args.num_workers
    temp_config.device = args.device
    temp_config.seed = args.seed

    train_loader, val_loader, test_loader, label_encoder = get_dataloaders_presplit(
        config=temp_config,
        batch_size=args.batch_size,
        mode="student",
        use_fft=True,     # 与原频域 student 保持一致
        seed=args.seed
    )

    logger.info("✓ 数据加载完成")
    logger.info(f"  训练集: {len(train_loader)} batches")
    logger.info(f"  验证集: {len(val_loader)} batches")
    logger.info(f"  测试集: {len(test_loader)} batches")

    F_bins = args.seq_len // 2 + 1

    # --------------------------------------------------------
    # 2) model
    # 只保留 pressure 分支，编码器仍复用原项目 MagnitudeEncoder
    # --------------------------------------------------------
    model = PressureOnlyMagnitudeBaseline(
        F_bins=F_bins,
        num_classes=args.num_classes,
        feature_dim=args.latent_dim,
        encoder_channels=[32,64,128],   # 与原 MagnitudeEncoder 默认一致
        dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ 模型参数量: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(args.save_dir, "pressure_only_best.pt")

    # --------------------------------------------------------
    # 3) train loop
    # --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, criterion)

        logger.info(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Train Acc={train_metrics['accuracy']:.2f}% | "
            f"Val Loss={val_metrics['loss']:.4f}, "
            f"Val Acc={val_metrics['accuracy']:.2f}%"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "config": vars(args)
            }, best_ckpt_path)

            logger.info(f"✓ 保存最佳模型到: {best_ckpt_path}")

    # --------------------------------------------------------
    # 4) test with best model
    # --------------------------------------------------------
    logger.info("=" * 80)
    logger.info("加载最佳模型并在测试集上评估")
    logger.info("=" * 80)

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # test loader batch size可单独调大
    temp_config_test = Config()
    temp_config_test.data_root = args.data_dir
    temp_config_test.window_size = args.seq_len
    temp_config_test.num_workers = args.num_workers
    temp_config_test.device = args.device
    temp_config_test.seed = args.seed

    _, _, test_loader_eval, _ = get_dataloaders_presplit(
        config=temp_config_test,
        batch_size=args.test_batch_size,
        mode="student",
        use_fft=True,
        seed=args.seed
    )

    test_metrics = evaluate(model, test_loader_eval, device, criterion)

    logger.info("\n" + "=" * 80)
    logger.info("Pressure-only Baseline Test Results")
    logger.info("=" * 80)
    logger.info(f"Accuracy : {test_metrics['accuracy']:.2f}%")
    logger.info(f"Precision: {test_metrics['precision']:.2f}%")
    logger.info(f"Recall   : {test_metrics['recall']:.2f}%")
    logger.info(f"F1 Score : {test_metrics['f1']:.2f}%")
    logger.info("=" * 80)

    with open(os.path.join(args.save_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)

    logger.info(f"✓ 测试结果已保存到: {os.path.join(args.save_dir, 'test_results.json')}")


if __name__ == "__main__":
    args = parse_args()
    train(args)

import subprocess

