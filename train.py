import torch
import torch.nn as nn
import math

from models import SmallTransformer


def train_model(model, dataloader, optimizer, device, log_every=100):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    steps_per_epoch = len(dataloader)
    # 训练开始时给出步数，帮助预估时间
    if steps_per_epoch > 0:
        print(f"[Train] steps_per_epoch={steps_per_epoch}", flush=True)

    for step, (x, y) in enumerate(dataloader, start=1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output, _ = model(x) if not isinstance(model, SmallTransformer) else (model(x), None)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / steps_per_epoch


def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output, _ = model(x) if not isinstance(model, SmallTransformer) else (model(x), None)
            # 统一处理输出，避免依赖具体模型类型
            if isinstance(output, (tuple, list)):
                output = output[0]
            loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl