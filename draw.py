import os
import re
import math
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime


def parse_log(log_path):
    """
    解析日志，返回：
    - curves: {model: {"epochs": [...], "train": [...], "val": [...]}}
    - tests:  {model: {"test_loss": x, "test_ppl": y}}
    - times:  {model: minutes_float}
    """
    assert os.path.exists(log_path), f"未找到日志文件: {log_path}"
    pattern_epoch = re.compile(
        r"^(RNN|LSTM|Transformer)\s*\|\s*Epoch\s*(\d+)\s*\|\s*TrainLoss:\s*([0-9.]+)\s*\|\s*ValLoss:\s*([0-9.]+)",
        re.IGNORECASE,
    )
    pattern_test = re.compile(
        r"^(RNN|LSTM|Transformer)\s*\|\s*TestLoss:\s*([0-9.]+)\s*\|\s*TestPPL:\s*([0-9.]+)",
        re.IGNORECASE,
    )
    pattern_run_start = re.compile(r"^\[Run Start\]\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")
    pattern_finish = re.compile(
        r"^(RNN|LSTM|Transformer)\s*\|\s*Training finished in\s*([0-9.]+)\s*min\.", re.IGNORECASE
    )

    curves = defaultdict(lambda: {"epochs": [], "train": [], "val": []})
    tests = {}
    times = {}

    # 仅解析最后一次运行
    last_run_offset = 0
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.readlines()
    for i, line in enumerate(content):
        if pattern_run_start.search(line):
            last_run_offset = i

    lines = content[last_run_offset:] if last_run_offset > 0 else content

    for line in lines:
        line = line.strip()
        m_ep = pattern_epoch.search(line)
        if m_ep:
            model = m_ep.group(1)
            epoch = int(m_ep.group(2))
            tr = float(m_ep.group(3))
            va = float(m_ep.group(4))
            curves[model]["epochs"].append(epoch)
            curves[model]["train"].append(tr)
            curves[model]["val"].append(va)
            continue

        m_te = pattern_test.search(line)
        if m_te:
            model = m_te.group(1)
            tests[model] = {
                "test_loss": float(m_te.group(2)),
                "test_ppl": float(m_te.group(3)),
            }
            continue

        m_fn = pattern_finish.search(line)
        if m_fn:
            model = m_fn.group(1)
            minutes = float(m_fn.group(2))
            times[model] = minutes
            continue

    # 按 epoch 排序
    for m, d in curves.items():
        if d["epochs"]:
            zipped = sorted(zip(d["epochs"], d["train"], d["val"]), key=lambda x: x[0])
            d["epochs"] = [z[0] for z in zipped]
            d["train"] = [z[1] for z in zipped]
            d["val"] = [z[2] for z in zipped]

    return curves, tests, times


def plot_single_model(name, epochs, train_loss, val_loss, out_dir, suffix):
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Val Loss", marker="s")
    plt.title(f"{name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{name}_loss_curve_{suffix}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_compare_val(curves, out_dir, suffix, ylabel="Val Loss", transform=None, filename="val_loss_comparison"):
    plt.figure(figsize=(7, 5))
    for name, d in curves.items():
        if not d["epochs"]:
            continue
        y = d["val"]
        if transform is not None:
            y = [transform(v) for v in y]
        plt.plot(d["epochs"], y, label=f"{name}", marker="o")
    plt.title(f"Validation {'PPL' if transform else 'Loss'} Comparison")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{filename}_{suffix}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path

# 新增：训练时间对比柱状图
def plot_time_bar(times, out_dir, suffix, filename="train_time_comparison"):
    # times: {model: minutes}
    labels = []
    values = []
    for m in ["RNN", "LSTM", "Transformer"]:
        if m in times:
            labels.append(m)
            values.append(times[m])
    if not labels:
        return None
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518"])
    plt.ylabel("Training Time (min)")
    plt.title("Training Time Comparison")
    plt.grid(axis="y", alpha=0.3)
    # 显示数值标签
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h, f"{h:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{filename}_{suffix}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Parse training log and draw curves.")
    parser.add_argument("--log", type=str, default="./loglog/train_log_9_4_128.txt", help="日志文件路径")
    parser.add_argument("--out", type=str, default="runs", help="输出图片目录")
    # 去掉 --ppl 开关，直接始终输出 PPL 对比
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    curves, tests, times = parse_log(args.log)
    if not curves:
        print("未在日志中解析到任何曲线数据，请检查日志格式或路径。")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[Info] 从 {args.log} 解析完成。将输出到目录: {args.out}，文件后缀: {timestamp}")

    # 单模型曲线
    saved = []
    for name, d in curves.items():
        if not d["epochs"]:
            continue
        p = plot_single_model(name, d["epochs"], d["train"], d["val"], args.out, suffix=timestamp)
        saved.append(p)

    # 验证集 Loss 对比
    p_val = plot_compare_val(curves, args.out, suffix=timestamp, ylabel="Val Loss", transform=None,
                             filename="val_loss_comparison")
    saved.append(p_val)

    # 验证集 PPL 对比（exp(ValLoss)）
    p_ppl = plot_compare_val(curves, args.out, suffix=timestamp, ylabel="Perplexity",
                             transform=lambda v: math.exp(v), filename="Perplexity_comparison")
    saved.append(p_ppl)

    # 训练时间柱状图
    p_time = plot_time_bar(times, args.out, suffix=timestamp, filename="train_time_comparison")
    if p_time:
        saved.append(p_time)
    else:
        print("[Warn] 日志中未解析到训练时间行，无法绘制训练时间对比图。")

    # 输出测试集汇总（若日志包含）
    if tests:
        print("\n[Test Summary]")
        for m in ["RNN", "LSTM", "Transformer"]:
            if m in tests:
                print(f"- {m}: TestLoss={tests[m]['test_loss']:.4f} | TestPPL={tests[m]['test_ppl']:.2f}")
    else:
        print("\n[Warn] 日志中未解析到测试集指标行（TestLoss/TestPPL）。")

    print("\n[Saved Figures]")
    for p in saved:
        print(f"- {p}")


if __name__ == "__main__":
    main()