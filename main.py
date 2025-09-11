import torch
from torch.utils.data import DataLoader
from models import RNNModel, LSTMModel, SmallTransformer
from train import train_model, evaluate_model
from utils import TextDataset, generate_text
from dataset_prepare import vocab_size, stoi, itos, train_text, val_text, test_text
import time
import math
import os, sys, io, atexit

# 同时屏幕+文件保存（保存到一个 txt）
os.makedirs("runs", exist_ok=True)
log_path = "/root/ASSI_2/train_log.txt"  # 如需每次单独文件，可改为 f"runs/train_{time.strftime('%Y%m%d_%H%M%S')}.txt"

class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)
    def flush(self):
        for st in self.streams:
            st.flush()

# 以追加模式写入到同一个 txt
_log_fh = open(log_path, "a", encoding="utf-8")
# 可选：写入一次性分隔头，便于区分多次运行
_log_fh.write("\n" + "="*80 + f"\n[Run Start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*80 + "\n")
_log_fh.flush()

# 将 stdout/stderr 重定向到 Tee（屏幕 + 文件）
sys.stdout = Tee(sys.stdout, _log_fh)
sys.stderr = Tee(sys.stderr, _log_fh)

@atexit.register
def _close_log_fh():
    try:
        _log_fh.flush()
        _log_fh.close()
    except Exception:
        pass

print(f"[Logger] 所有输出将同时写入: {log_path}", flush=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 可复现性设置
torch.manual_seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

# 构建数据集（训练/验证/测试）
train_dataset = TextDataset(train_text, seq_len=128, stoi=stoi, itos=itos)
val_dataset = TextDataset(val_text, seq_len=128, stoi=stoi, itos=itos)
test_dataset = TextDataset(test_text, seq_len=128, stoi=stoi, itos=itos)

# DataLoader 公共参数（batch_size 将在每个模型处单独指定）
loader_kwargs = {"batch_size": 64, "shuffle": True}
if device.type == "cuda":
    loader_kwargs.update({"num_workers": 4, "pin_memory": True, "persistent_workers": True})

models = {
    "RNN": RNNModel(vocab_size, 256, 256, 2, 0.2),
    "LSTM": LSTMModel(vocab_size, 256, 256, 2, 0.2),
    "Transformer": SmallTransformer(vocab_size, 256, 4, 2, 0.2)
}

num_epochs = 15
temperatures = [0.7, 1.0, 1.3]

# 为不同模型设置不同的 batch_size（可按显存情况调整）
per_model_batch = {
    "RNN": 96,         # RNN 通常可用更大的 batch
    "LSTM": 96,        # LSTM 也较省显存
    "Transformer": 32  # Transformer 更吃显存，保守一些
}

for name, model in models.items():
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 基于当前模型设置重建 DataLoader（仅 batch_size 不同，其他参数沿用）
    bs = per_model_batch.get(name, loader_kwargs["batch_size"])
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=loader_kwargs.get("num_workers", 0),
        pin_memory=loader_kwargs.get("pin_memory", False),
        persistent_workers=loader_kwargs.get("persistent_workers", False)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=loader_kwargs.get("num_workers", 0),
        pin_memory=loader_kwargs.get("pin_memory", False),
        persistent_workers=loader_kwargs.get("persistent_workers", False)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False,
        num_workers=loader_kwargs.get("num_workers", 0),
        pin_memory=loader_kwargs.get("pin_memory", False),
        persistent_workers=loader_kwargs.get("persistent_workers", False)
    )

    best_val_loss = float("inf")
    best_state = None
    train_curve = []
    val_curve = []

    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, device)
        val_loss, val_ppl = evaluate_model(model, val_loader, device)
        train_curve.append(train_loss)
        val_curve.append(val_loss)
        print(f"{name} | Epoch {epoch+1} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | PPL: {val_ppl:.2f}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    elapsed = time.time() - start_time
    print(f"{name} | Training finished in {elapsed/60:.2f} min. Best ValLoss: {best_val_loss:.4f} (PPL={math.exp(best_val_loss):.2f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_ppl = evaluate_model(model, test_loader, device)
    print(f"{name} | TestLoss: {test_loss:.4f} | TestPPL: {test_ppl:.2f}")

    start_prompt = "The "
    for T in temperatures:
        sample = generate_text(model, stoi=stoi, itos=itos,
                               start_text=start_prompt, max_new_tokens=200,
                               temperature=T, device=device, top_k=50)
        print(f"\n[{name}] Sample (T={T}):\n{sample}\n")
