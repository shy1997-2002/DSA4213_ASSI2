import time
import math
import torch
import os, sys, io, atexit

from torch.utils.data import DataLoader
from models import SmallTransformer
from train import train_model, evaluate_model
from utils import TextDataset, generate_text
from dataset_prepare import vocab_size, stoi, itos, train_text, val_text, test_text


os.makedirs("runs", exist_ok=True)
log_path = "runs/transformer_ctx_ablation_log.txt"  # 固定一个文件；如需每次独立文件，可改为带时间戳

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

def build_loaders(seq_len, batch_size, device):
    train_dataset = TextDataset(train_text, seq_len=seq_len, stoi=stoi, itos=itos)
    val_dataset   = TextDataset(val_text,   seq_len=seq_len, stoi=stoi, itos=itos)
    test_dataset  = TextDataset(test_text,  seq_len=seq_len, stoi=stoi, itos=itos)

    loader_kwargs = {"batch_size": batch_size, "shuffle": True}
    if device.type == "cuda":
        loader_kwargs.update({"num_workers": 4, "pin_memory": True, "persistent_workers": True})

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=loader_kwargs.get("num_workers", 0),
        pin_memory=loader_kwargs.get("pin_memory", False),
        persistent_workers=loader_kwargs.get("persistent_workers", False),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=loader_kwargs.get("num_workers", 0),
        pin_memory=loader_kwargs.get("pin_memory", False),
        persistent_workers=loader_kwargs.get("persistent_workers", False),
    )
    return train_loader, val_loader, test_loader

def run_transformer_for_seq_len(seq_len, batch_size, num_epochs, device, lr=1e-3):
    print(f"\n=== Transformer | seq_len={seq_len} | batch_size={batch_size} ===", flush=True)

    train_loader, val_loader, test_loader = build_loaders(seq_len, batch_size, device)

    model = SmallTransformer(vocab_size=vocab_size, embed_size=256, num_heads=4, num_layers=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    train_curve, val_curve = [], []

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss = train_model(model, train_loader, optimizer, device)
        val_loss, val_ppl = evaluate_model(model, val_loader, device)
        train_curve.append(train_loss)
        val_curve.append(val_loss)
        print(f"Transformer[ctx={seq_len}] | Epoch {epoch} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | PPL: {val_ppl:.2f}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    elapsed = time.time() - start_time
    print(f"Transformer[ctx={seq_len}] | Training finished in {elapsed/60:.2f} min. Best ValLoss: {best_val_loss:.4f} (PPL={math.exp(best_val_loss):.2f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_ppl = evaluate_model(model, test_loader, device)
    print(f"Transformer[ctx={seq_len}] | TestLoss: {test_loss:.4f} | TestPPL: {test_ppl:.2f}")

    # 样例生成（同 main.py）
    temperatures = [0.7, 1.0, 1.3]
    start_prompt = "The "
    for T in temperatures:
        sample = generate_text(model, stoi=stoi, itos=itos,
                               start_text=start_prompt, max_new_tokens=200,
                               temperature=T, device=device, top_k=50)
        print(f"\n[Transformer ctx={seq_len}] Sample (T={T}):\n{sample}\n")

    return {
        "seq_len": seq_len,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(best_val_loss),
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "time_min": elapsed / 60.0,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    # 配置：上下文长度对比
    settings = [
        {"seq_len": 128, "batch_size": 32},  # 安全起点
        {"seq_len": 256, "batch_size": 32},  # 序列加倍 -> batch 适当减半
    ]
    num_epochs = 15
    lr = 1e-3

    results = []
    for cfg in settings:
        res = run_transformer_for_seq_len(
            seq_len=cfg["seq_len"],
            batch_size=cfg["batch_size"],
            num_epochs=num_epochs,
            device=device,
            lr=lr,
        )
        results.append(res)

    # 汇总打印
    print("\n=== Summary: Transformer Context Length Ablation ===")
    for r in results:
        print(
            f"- ctx={r['seq_len']}: "
            f"BestValLoss={r['best_val_loss']:.4f} (PPL={r['best_val_ppl']:.2f}) | "
            f"TestLoss={r['test_loss']:.4f} (PPL={r['test_ppl']:.2f}) | "
            f"Time={r['time_min']:.2f} min"
        )

if __name__ == "__main__":
    main()