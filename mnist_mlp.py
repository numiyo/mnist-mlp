# -*- coding: utf-8 -*-
"""
第七章实验：基于多层感知机（MLP）的 MNIST 手写数字识别
=====================================================================
实验内容：
  1. 使用 torchvision 的 transforms 完成张量转换（ToTensor）与标准化（Normalize）；
  2. 使用 datasets 加载 MNIST，用 DataLoader 实现批量、随机打乱的数据读取；
  3. 继承 nn.Module 自定义三层全连接 MLP（784 -> 256 -> 128 -> 10），ReLU 激活；
  4. 完成 20 轮训练，逐轮记录训练集/验证集的 loss 与准确率；
  5. 保存验证集上最优的模型，在测试集上评估，并绘制 loss/准确率曲线、
     混淆矩阵与预测样例图。

运行方式：
  python mnist_mlp.py                       # 默认训练 20 轮
  python mnist_mlp.py --epochs 20 --lr 1e-3 # 自定义超参数

输出文件（默认 results/ 目录）：
  metrics.csv            每轮训练集/验证集的 loss 与准确率
  history.json           同上，JSON 格式
  train_log.txt          完整训练日志
  curves.png             loss 与准确率迭代曲线
  samples.png            MNIST 数据样例
  predictions.png        测试集预测样例（绿色为预测正确，红色为错误）
  confusion_matrix.png   测试集混淆矩阵
  mlp_mnist.pth          验证集上最优的模型参数
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 使用无界面的 Agg 后端，只把图片保存到文件，不弹窗
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# MNIST 训练集整体的像素均值与标准差（按单通道统计），用于 Normalize
MEAN = (0.1307,)
STD = (0.3081,)

# 让 matplotlib 正常显示中文与负号，避免图中出现方块
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# 一、通用工具
# ----------------------------------------------------------------------
class Tee:
    """把 print 的内容同时输出到控制台和日志文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def set_seed(seed):
    """固定随机种子，保证每次运行结果可复现。"""
    random.seed(seed)                 # Python 内置随机数
    np.random.seed(seed)              # NumPy 随机数
    torch.manual_seed(seed)           # PyTorch CPU 随机数
    torch.cuda.manual_seed_all(seed)  # PyTorch 所有 GPU 的随机数


def get_device(name="auto"):
    """选择计算设备：有 GPU 用 GPU，否则用 CPU。"""
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# 二、数据准备
# ----------------------------------------------------------------------
def build_dataloaders(data_dir, batch_size, seed, num_workers=0, val_ratio=1 / 6):
    """构建训练集、验证集、测试集三个 DataLoader。

    MNIST 官方训练集有 60000 张图片，这里按 5:1 划分为
    训练集 50000 张 + 验证集 10000 张，官方测试集 10000 张只用于最终评估。
    """
    # Compose 把多个预处理操作串联起来，按顺序执行
    transform = transforms.Compose([
        # ToTensor：PIL 图片 -> (1, 28, 28) 的 float32 张量，像素值从 [0,255] 缩放到 [0,1]
        transforms.ToTensor(),
        # Normalize：对每个通道做 (x - mean) / std，使输入近似零均值、单位方差，利于网络收敛
        transforms.Normalize(MEAN, STD),
    ])

    # download=True：本地没有数据时自动下载；已有则直接读取
    train_full = torchvision.datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=True)

    # 按比例划分训练集与验证集；固定 generator 保证每次划分结果一致
    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed))

    # 训练集需要 shuffle=True 随机打乱，验证/测试集不需要打乱
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------
# 三、模型定义
# ----------------------------------------------------------------------
class MLP(nn.Module):
    """三层全连接多层感知机：784 -> 256 -> 128 -> 10。

    输入是 28x28 的灰度图，展平后为 784 维向量；
    经过两个隐藏层（每层后接 ReLU）提取特征，最后输出 10 个类别的得分（logits）。
    """

    def __init__(self, in_dim=28 * 28, hidden1=256, hidden2=128, num_classes=10):
        super().__init__()  # 初始化父类 nn.Module，必须调用
        # Flatten：把 (N, 1, 28, 28) 的张量展平成 (N, 784)
        self.flatten = nn.Flatten()
        # 第一个全连接层：784 维输入 -> 256 维隐藏特征
        self.fc1 = nn.Linear(in_dim, hidden1)
        # 第二个全连接层：256 维 -> 128 维
        self.fc2 = nn.Linear(hidden1, hidden2)
        # 输出层：128 维 -> 10 个类别（0~9 的得分）
        self.fc3 = nn.Linear(hidden2, num_classes)
        # ReLU 激活函数：负值置 0，正值不变，为网络引入非线性
        self.relu = nn.ReLU()

    def forward(self, x):
        """前向传播：定义数据从输入到输出的计算流程。"""
        x = self.flatten(x)        # (N, 1, 28, 28) -> (N, 784)
        x = self.relu(self.fc1(x))  # 全连接 + 激活 -> (N, 256)
        x = self.relu(self.fc2(x))  # 全连接 + 激活 -> (N, 128)
        x = self.fc3(x)             # 输出层不加激活，直接得到 10 个 logits
        return x


# ----------------------------------------------------------------------
# 四、训练与评估
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回该轮的平均 loss 与准确率（%）。"""
    model.train()  # 切换到训练模式（启用 Dropout、BatchNorm 等训练期行为）
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        # 把数据搬到 GPU（或 CPU），与模型所在设备保持一致
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)              # 前向传播，得到 (N, 10) 的预测得分
        loss = criterion(outputs, labels)    # 计算交叉熵损失

        optimizer.zero_grad()                # 清空上一轮累积的梯度
        loss.backward()                      # 反向传播，计算每个参数的梯度
        optimizer.step()                     # 用优化器更新参数

        # loss.item() 是当前 batch 的平均损失，乘以样本数还原成总损失
        total_loss += loss.item() * labels.size(0)
        # argmax(dim=1) 取 10 个得分中最大的下标作为预测类别
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()  # 评估阶段不需要计算梯度，可节省显存并加速
def evaluate(model, loader, criterion, device, collect_preds=False):
    """在验证集/测试集上评估，返回平均 loss 与准确率（%）。"""
    model.eval()  # 切换到评估模式（关闭 Dropout 等训练期行为）
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if collect_preds:  # 测试集需要预测结果来画混淆矩阵
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    if collect_preds:
        return (total_loss / total, 100.0 * correct / total,
                torch.cat(all_preds), torch.cat(all_labels))
    return total_loss / total, 100.0 * correct / total


# ----------------------------------------------------------------------
# 五、可视化
# ----------------------------------------------------------------------
def plot_curves(history, save_path):
    """绘制训练集/验证集的 loss 与准确率随 epoch 变化的曲线。"""
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # x 轴只显示整数轮次
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 左图：loss 曲线
    axes[0].plot(epochs, history["train_loss"], "o-", color="#1f77b4",
                 linewidth=2, markersize=4, label="训练集 Loss")
    axes[0].plot(epochs, history["val_loss"], "s-", color="#d62728",
                 linewidth=2, markersize=4, label="验证集 Loss")
    axes[0].set_title("损失函数（Loss）随迭代轮数变化")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3, linestyle="--")
    axes[0].legend()

    # 右图：准确率曲线
    axes[1].plot(epochs, history["train_acc"], "o-", color="#1f77b4",
                 linewidth=2, markersize=4, label="训练集准确率")
    axes[1].plot(epochs, history["val_acc"], "s-", color="#d62728",
                 linewidth=2, markersize=4, label="验证集准确率")
    axes[1].set_title("准确率（Accuracy）随迭代轮数变化")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3, linestyle="--")
    axes[1].legend()

    # 在右图标出验证集准确率最高的那一轮
    best_idx = int(np.argmax(history["val_acc"]))
    best_epoch = epochs[best_idx]
    best_acc = history["val_acc"][best_idx]
    axes[1].scatter([best_epoch], [best_acc], marker="*", s=180,
                    color="#ff7f0e", zorder=5,
                    label="最优验证准确率")
    axes[1].annotate(f"best {best_acc:.2f}% @ epoch {best_epoch}",
                     xy=(best_epoch, best_acc),
                     xytext=(0, -30), textcoords="offset points",
                     ha="center", fontsize=9, color="#ff7f0e",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               ec="#ff7f0e", alpha=0.9))
    axes[1].legend(loc="lower right")

    fig.suptitle(f"MLP 在 MNIST 上的训练过程（{len(epochs)} Epochs）", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_samples(dataset, save_path, n=10):
    """随机抽取若干张 MNIST 图片，展示图像与真实标签。"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.6))
    idxs = np.random.choice(len(dataset), size=n, replace=False)

    for ax, idx in zip(axes.ravel(), idxs):
        image, label = dataset[idx]
        # 反标准化：x * std + mean，把像素还原到 [0,1] 才能正常显示
        image = image * STD[0] + MEAN[0]
        ax.imshow(image.squeeze(0).numpy(), cmap="gray")
        ax.set_title(f"标签: {label}", fontsize=10)
        ax.axis("off")

    fig.suptitle("MNIST 手写数字样例", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(model, loader, device, save_path, n=12):
    """在测试集上展示 n 张图片的预测结果：绿色正确，红色错误。"""
    model.eval()
    images, labels = next(iter(loader))  # 取一个 batch 即可
    images, labels = images[:n].to(device), labels[:n]

    with torch.no_grad():
        preds = model(images).argmax(dim=1).cpu()

    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.6))
    for ax, image, label, pred in zip(axes.ravel(), images.cpu(), labels, preds):
        image = image * STD[0] + MEAN[0]
        ok = int(pred) == int(label)
        ax.imshow(image.squeeze(0).numpy(), cmap="gray")
        ax.set_title(f"预测:{int(pred)} / 真实:{int(label)}",
                     fontsize=10, color="green" if ok else "red")
        ax.axis("off")

    fig.suptitle("测试集预测样例（绿色：预测正确；红色：预测错误）", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, save_path):
    """绘制 10x10 混淆矩阵，行是真实类别，列是预测类别。"""
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("测试集混淆矩阵")

    # 在每个格子里写上数量；深色格子用白字，浅色格子用黑字
    threshold = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > threshold else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 六、主流程
# ----------------------------------------------------------------------
def parse_args():
    """解析命令行参数，方便修改轮数、学习率等超参数。"""
    parser = argparse.ArgumentParser(description="MLP 手写数字识别（MNIST）")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 正则系数")
    parser.add_argument("--data-dir", type=str, default="data", help="数据集目录")
    parser.add_argument("--out-dir", type=str, default="results", help="结果输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 工作进程数")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto / cpu / cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 同时把输出写到控制台与日志文件（UTF-8 编码，避免中文乱码）
    log_file = open(out_dir / "train_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)

    set_seed(args.seed)
    device = get_device(args.device)

    print("=" * 70)
    print("实验：基于 MLP 的 MNIST 手写数字识别")
    print(f"PyTorch 版本：{torch.__version__}      torchvision 版本：{torchvision.__version__}")
    print(f"计算设备：{device}" +
          (f"（{torch.cuda.get_device_name(0)}）" if device.type == "cuda" else ""))
    print(f"超参数：epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, weight_decay={args.weight_decay}, seed={args.seed}")
    print("=" * 70)

    # ---------- 1. 数据 ----------
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_dir, args.batch_size, args.seed, args.num_workers)
    print(f"训练集：{len(train_loader.dataset)} 张，"
          f"{len(train_loader)} 个 batch")
    print(f"验证集：{len(val_loader.dataset)} 张，"
          f"{len(val_loader)} 个 batch")
    print(f"测试集：{len(test_loader.dataset)} 张，"
          f"{len(test_loader)} 个 batch")

    # ---------- 2. 模型 / 损失函数 / 优化器 ----------
    model = MLP().to(device)
    print("\n模型结构：")
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"可学习参数总量：{num_params:,}")

    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵，内部自带 softmax
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # ---------- 3. 训练 ----------
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "epoch_time": []}
    best_val_acc = 0.0
    best_epoch = 0
    model_path = out_dir / "mlp_mnist.pth"

    print("\n开始训练：")
    print("-" * 78)
    print(f"{'Epoch':>5} | {'train_loss':>10} {'train_acc':>9} | "
          f"{'val_loss':>9} {'val_acc':>8} | {'time(s)':>8}")
    print("-" * 78)
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        # 记录本轮指标，供后面画图与写表
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        # 保存验证集准确率最高的模型参数（防止后续轮次过拟合）
        flag = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), model_path)
            flag = "  <- best"

        print(f"{epoch:5d} | {train_loss:10.4f} {train_acc:8.2f}% | "
              f"{val_loss:9.4f} {val_acc:7.2f}% | {epoch_time:8.2f}{flag}")

    total_time = time.time() - train_start
    print("-" * 78)
    print(f"训练完成，共耗时 {total_time:.1f} 秒；"
          f"最优验证准确率 {best_val_acc:.2f}%（第 {best_epoch} 轮）")

    # ---------- 4. 用最优模型在测试集上评估 ----------
    print("\n加载验证集上最优的模型，在测试集上评估：")
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device, collect_preds=True)
    print(f"测试集 Loss = {test_loss:.4f}，准确率 = {test_acc:.2f}%")

    # 统计每个数字的召回率，找出最容易混淆的类别
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels.numpy(), preds.numpy()):
        cm[t, p] += 1
    print("\n各类别准确率（召回率）：")
    for digit in range(10):
        print(f"  数字 {digit}：{100.0 * cm[digit, digit] / cm[digit].sum():5.2f}%  "
              f"（{cm[digit, digit]}/{cm[digit].sum()}）")

    # ---------- 5. 保存结果 ----------
    # 每轮指标写成 CSV，方便用 Excel 查看或后续重绘曲线
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "epoch_time"])
        for i in range(len(history["epoch"])):
            writer.writerow([history["epoch"][i],
                             f"{history['train_loss'][i]:.6f}",
                             f"{history['train_acc'][i]:.4f}",
                             f"{history['val_loss'][i]:.6f}",
                             f"{history['val_acc'][i]:.4f}",
                             f"{history['epoch_time'][i]:.4f}"])

    # 历史指标与本次实验的配置、测试结果一起写入 JSON
    summary = {
        "args": vars(args),
        "num_params": num_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "per_class_acc": [100.0 * cm[d, d] / cm[d].sum() for d in range(10)],
        "history": history,
    }
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 画图
    plot_curves(history, out_dir / "curves.png")
    plot_samples(test_loader.dataset, out_dir / "samples.png")
    plot_predictions(model, test_loader, device, out_dir / "predictions.png")
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    print(f"\n结果已保存到 {out_dir.resolve()}")
    print("  metrics.csv / history.json / train_log.txt")
    print("  curves.png / samples.png / predictions.png / confusion_matrix.png")
    print("  mlp_mnist.pth")
    print("=" * 70)

    sys.stdout = sys.__stdout__  # 恢复标准输出
    log_file.close()


if __name__ == "__main__":
    main()
