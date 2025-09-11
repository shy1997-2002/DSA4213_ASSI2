import os
import io
import sys
import csv
import math
import time
import atexit
import torch
from torch.utils.data import DataLoader

from models import SmallTransformer
from train import train_model, evaluate_model
from utils import TextDataset, generate_text
from dataset_prepare import vocab_size, stoi, itos, train_text, val_text, test_text

# ---- Tee：屏幕 + 文件 同步输出 ----
os.makedirs("runs", exist_ok=True)
log_path = "runs/dropout_ablation_log.txt"  # 如需每次独立文件，改为带时间戳命名

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

_log_fh = open(log_path, "a", encoding="utf-8")
_log_fh.write("\n" + "="*80 + f"\n[Run Start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*80 + "\n")
_log_fh.flush()
sys.stdout = Tee(sys.stdout, _log_fh)
sys.stderr = Tee(sys.stderr, _log_fh)

@atexit.register
def _close_log_fh():
    try:
        _log_fh.flush()
        _log_fh.close()
    except Exception:
        pass

print(f"[Logger] 所有输出将同时写入: {os.path.abspath(log_path)}", flush=True)

# ---- 设备 & 随机性 ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

# ---- 数据集与 DataLoader 构建 ----
def build_loaders(seq_len, batch_size):
    tr_ds = TextDataset(train_text, seq_len=seq_len, stoi=stoi, itos=itos)
    va_ds = TextDataset(val_text,   seq_len=seq_len, stoi=stoi, itos=itos)
    te_ds = TextDataset(test_text,  seq_len=seq_len, stoi=stoi, itos=itos)

    loader_kwargs = {"batch_size": batch_size, "shuffle": True}
    if device.type == "cuda":
        loader_kwargs.update({"num_workers": 4, "pin_memory": True, "persistent_workers": True})

    tr = DataLoader(tr_ds, **loader_kwargs)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                    num_workers=loader_kwargs.get("num_workers", 0),
                    pin_memory=loader_kwargs.get("pin_memory", False),
                    persistent_workers=loader_kwargs.get("persistent_workers", False))
    te = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                    num_workers=loader_kwargs.get("num_workers", 0),
                    pin_memory=loader_kwargs.get("pin_memory", False),
                    persistent_workers=loader_kwargs.get("persistent_workers", False))
    return tr, va, te

# ---- 单次实验运行（仅 Transformer） ----
def run_transformer(dropout, num_epochs, seq_len, batch_size, lr=1e-3, gen=False):
    print(f"\n=== Transformer | Dropout={dropout} | seq_len={seq_len} | batch_size={batch_size} ===", flush=True)
    train_loader, val_loader, test_loader = build_loaders(seq_len, batch_size)

    model = SmallTransformer(vocab_size, embed_size=256, num_heads=4, num_layers=2, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    train_curve, val_curve = [], []

    start = time.time()
    for ep in range(1, num_epochs + 1):
        tr = train_model(model, train_loader, optimizer, device)
        vl, vp = evaluate_model(model, val_loader, device)
        train_curve.append(tr)
        val_curve.append(vl)
        print(f"Transformer[dp={dropout}] | Epoch {ep} | TrainLoss: {tr:.4f} | ValLoss: {vl:.4f} | PPL: {vp:.2f}", flush=True)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    elapsed = time.time() - start
    print(f"Transformer[dp={dropout}] | Finished in {elapsed/60:.2f} min. Best ValLoss: {best_val:.4f} (PPL={math.exp(best_val):.2f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_ppl = evaluate_model(model, test_loader, device)
    print(f"Transformer[dp={dropout}] | TestLoss: {test_loss:.4f} | TestPPL: {test_ppl:.2f}")

    # 可选：生成样例，辅助定性分析
    if gen:
        temperatures = [0.7, 1.0, 1.3]
        start_prompt = "The "
        for T in temperatures:
            sample = generate_text(model, stoi=stoi, itos=itos, start_text=start_prompt,
                                   max_new_tokens=200, temperature=T, device=device, top_k=50)
            print(f"\n[Transformer dp={dropout}] Sample (T={T}):\n{sample}\n")

    # 保存曲线（CSV）
    try:
        os.makedirs("runs", exist_ok=True)
        csv_path = f"runs/Transformer_dropout_{str(dropout).replace('.', '_')}_loss_curve.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            for ep, tr, vl in zip(range(1, num_epochs+1), train_curve, val_curve):
                writer.writerow([ep, tr, vl])
        print(f"[Info] Saved loss curves to {csv_path}")
    except Exception as e:
        print(f"Warning: failed to save curves csv due to: {e}")

    return {
        "dropout": dropout,
        "best_val_loss": best_val,
        "best_val_ppl": math.exp(best_val),
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "time_min": elapsed / 60.0,
    }

def main():
    # 实验设置（仅 Transformer）
    num_epochs = 10       # 或 5 提速
    seq_len = 128         # 与主实验一致；可保持不变
    batch_size = 32       # 视显存调节；若 OOM 降到 16
    lr = 1e-3
    dropout_list = [0.0, 0.2]  # 需要更多对比可加 0.1/0.3

    results = []
    for dp in dropout_list:
        res = run_transformer(dropout=dp, num_epochs=num_epochs,
                              seq_len=seq_len, batch_size=batch_size, lr=lr, gen=False)
        results.append(res)

    # 汇总打印
    print("\n=== Summary: Transformer Dropout Ablation ===")
    for r in results:
        print(f"- dp={r['dropout']}: "
              f"BestValLoss={r['best_val_loss']:.4f} (PPL={r['best_val_ppl']:.2f}) | "
              f"TestLoss={r['test_loss']:.4f} (PPL={r['test_ppl']:.2f}) | "
              f"Time={r['time_min']:.2f} min")

    # 保存汇总 CSV
    try:
        csv_sum = "runs/transformer_dropout_ablation_summary.csv"
        with open(csv_sum, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dropout", "best_val_loss", "best_val_ppl", "test_loss", "test_ppl", "time_min"])
            for r in results:
                writer.writerow([r["dropout"], r["best_val_loss"], r["best_val_ppl"], r["test_loss"], r["test_ppl"], r["time_min"]])
        print(f"[Info] Saved summary to {csv_sum}")
    except Exception as e:
        print(f"Warning: failed to save summary due to: {e}")

if __name__ == "__main__":
    main()