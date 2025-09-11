import os
import torch
from torch.utils.data import DataLoader
from utils import TextDataset  # 你自己的 Dataset 类

# ===== 1. 数据准备 =====
data_path = "data/news_corpus.txt"  # 改成你的文件路径
if not os.path.exists(data_path):
    raise FileNotFoundError(f"找不到文件: {data_path}")

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"数据总长度: {len(text):,} 个字符")

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print(f"词表大小: {vocab_size}")
print("示例 stoi:", list(stoi.items())[:10])

n = len(text)
train_text = text[:int(0.8*n)]
val_text   = text[int(0.8*n):int(0.9*n)]
test_text  = text[int(0.9*n):]

print(f"训练集: {len(train_text):,} 字符")
print(f"验证集: {len(val_text):,} 字符")
print(f"测试集: {len(test_text):,} 字符")

# ===== 2. DataLoader 构建 =====
seq_len = 128
batch_size = 64

train_dataset = TextDataset(train_text, seq_len=seq_len, stoi=stoi, itos=itos)
val_dataset   = TextDataset(val_text,   seq_len=seq_len, stoi=stoi, itos=itos)
test_dataset  = TextDataset(test_text,  seq_len=seq_len, stoi=stoi, itos=itos)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

print("训练集样本数:", len(train_dataset))
print("验证集样本数:", len(val_dataset))
