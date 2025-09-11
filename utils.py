import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, seq_len, stoi, itos):
        self.data = [stoi[ch] for ch in text]
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len])
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1])
        return x, y

def generate_text(model, stoi, itos, start_text, max_new_tokens=200, temperature=1.0, device="cpu", top_k=None):
    """
    从起始文本开始自回归生成字符。支持 RNN/LSTM（返回 (logits, hidden)）与 Transformer（返回 logits）。
    temperature: 温度 > 0；top_k: 可选，仅采样 top-k。
    """
    model.eval()
    # 将起始字符串映射为 token 序列
    tokens = [stoi.get(ch, stoi.get("<unk>", 0)) for ch in start_text]
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    hidden = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # RNN/LSTM 一次喂最后一个 token；Transformer 喂全序列（有因果掩码保证自回归）
            if hasattr(model, "lstm") or hasattr(model, "rnn"):
                x = input_ids[:, -1:]  # (1, 1)
                logits, hidden = model(x, hidden)
            else:
                logits = model(input_ids)
            logits = logits[:, -1, :]  # 取最后一步
            if temperature != 1.0:
                logits = logits / max(1e-8, temperature)
            probs = torch.softmax(logits, dim=-1)

            if top_k is not None and top_k > 0:
                k = min(top_k, probs.size(-1))
                topk_probs, topk_idx = torch.topk(probs, k=k, dim=-1)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                next_id = topk_idx.gather(-1, torch.multinomial(topk_probs, num_samples=1))
            else:
                next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

            input_ids = torch.cat([input_ids, next_id], dim=1)

    # 反向映射为字符
    def id_to_token(i):
        # itos 可能是 list 或 dict
        if isinstance(itos, dict):
            return itos.get(i, "")
        elif isinstance(itos, list):
            return itos[i] if 0 <= i < len(itos) else ""
        else:
            return ""
    out = "".join(id_to_token(int(i)) for i in input_ids[0].tolist())
    return out