# -*- coding: utf-8 -*-
"""
根据 results/metrics.csv 重新绘制训练曲线，不需要重新训练模型。
用法：
    python plot_curves.py
    python plot_curves.py --csv results/metrics.csv --out results/curves.png
"""
import argparse
import csv
from pathlib import Path

from mnist_mlp import plot_curves


def load_history(csv_path):
    """从 metrics.csv 读回每一轮的训练/验证指标。"""
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "epoch_time": []}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            history["epoch"].append(int(row["epoch"]))
            history["train_loss"].append(float(row["train_loss"]))
            history["train_acc"].append(float(row["train_acc"]))
            history["val_loss"].append(float(row["val_loss"]))
            history["val_acc"].append(float(row["val_acc"]))
            history["epoch_time"].append(float(row["epoch_time"]))
    return history


def main():
    parser = argparse.ArgumentParser(description="重新绘制 MNIST 训练曲线")
    parser.add_argument("--csv", type=str, default="results/metrics.csv")
    parser.add_argument("--out", type=str, default="results/curves.png")
    args = parser.parse_args()

    history = load_history(args.csv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_curves(history, args.out)
    print(f"已根据 {args.csv} 重新绘制曲线：{args.out}")


if __name__ == "__main__":
    main()
