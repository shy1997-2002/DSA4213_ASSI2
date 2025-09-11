import os
import io
import sys
import math
import time
import atexit
import torch
from torch.utils.data import DataLoader, Dataset

from models import SmallTransformer
from train import train_model, evaluate_model
from dataset_prepare import vocab_size as _char_vocab_size, stoi as _char_stoi, itos as _char_itos
from dataset_prepare import train_text, val_text, test_text

# ---- Tee：屏幕 + 文件 同步输出 ----
os.makedirs("runs", exist_ok=True)
log_path = "runs/word_subword_ablation_log.txt"

class Tee(io.TextIOBase):
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
        return len(s)
    def flush(self):
        for st in self.streams: st.flush()

_log_fh = open(log_path, "a", encoding="utf-8")
_log_fh.write("\n" + "="*80 + f"\n[Run Start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*80 + "\n")
_log_fh.flush()
sys.stdout = Tee(sys.stdout, _log_fh)
sys.stderr = Tee(sys.stderr, _log_fh)

print(f"[Logger] 输出同时写入: {os.path.abspath(log_path)}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

# ---- 通用 Dataset（基于已编码的 token id 列表）----
class GenericTokenDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.data = token_ids
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# ---- 词级分词器（空格切分）----
class WordTokenizer:
    def __init__(self, vocab=None, unk_token="<unk>"):
        self.unk = unk_token
        self.vocab = vocab or {}
        self.inv = None

    def build_from_text(self, text):
        words = text.split()
        vocab = {self.unk: 0}
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
        self.vocab = vocab
        self.inv = {i: w for w, i in vocab.items()}
        return self

    def encode(self, text):
        return [self.vocab.get(w, self.vocab[self.unk]) for w in text.split()]

    def decode(self, ids):
        if self.inv is None:
            self.inv = {i: w for w, i in self.vocab.items()}
        return " ".join(self.inv.get(i, self.unk) for i in ids)

    @property
    def size(self):
        return len(self.vocab)

# ---- 简易 BPE 分词器（纯 Python，适用于英文/空格分词后的词表）----
class SimpleBPE:
    def __init__(self, vocab_size=2000, end_token="</w>", unk_token="<unk>"):
        self.vocab_size = vocab_size
        self.end = end_token
        self.unk = unk_token
        self.merges = []
        self.token2id = None
        self.id2token = None

    def _word_to_symbols(self, word):
        return list(word) + [self.end]

    def _get_stats(self, corpus_syms):
        from collections import Counter
        pairs = Counter()
        for syms in corpus_syms:
            for i in range(len(syms)-1):
                pairs[(syms[i], syms[i+1])] += 1
        return pairs

    def _merge_corpus(self, corpus_syms, pair):
        a, b = pair
        merged = []
        for syms in corpus_syms:
            i = 0
            new_syms = []
            while i < len(syms):
                if i < len(syms)-1 and syms[i] == a and syms[i+1] == b:
                    new_syms.append(a + b)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            merged.append(new_syms)
        return merged

    def train(self, text, max_merges=None):
        words = text.split()
        corpus_syms = [self._word_to_symbols(w) for w in words if w]

        charset = set(ch for w in corpus_syms for ch in w)
        if max_merges is None:
            max_merges = max(0, self.vocab_size - len(charset) - 2)
        self.merges = []

        for _ in range(max_merges):
            stats = self._get_stats(corpus_syms)
            if not stats:
                break
            (a, b), cnt = max(stats.items(), key=lambda x: x[1])
            if cnt < 2:
                break
            self.merges.append((a, b))
            corpus_syms = self._merge_corpus(corpus_syms, (a, b))

        vocab_tokens = set()
        for syms in corpus_syms:
            vocab_tokens.update(syms)
        for a, b in self.merges:
            vocab_tokens.add(a + b)
        for w in words:
            for ch in self._word_to_symbols(w):
                vocab_tokens.add(ch)

        vocab_tokens.add(self.unk)
        tokens = sorted(vocab_tokens)
        if len(tokens) > self.vocab_size:
            tokens = tokens[:self.vocab_size-1] + [self.unk]
        self.token2id = {t: i for i, t in enumerate(tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        return self

    def _apply_merges_to_word(self, word):
        syms = self._word_to_symbols(word)
        for a, b in self.merges:
            i = 0
            new_syms = []
            while i < len(syms):
                if i < len(syms)-1 and syms[i] == a and syms[i+1] == b:
                    new_syms.append(a + b)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms
        return syms

    def encode(self, text):
        ids = []
        for w in text.split():
            syms = self._apply_merges_to_word(w)
            for s in syms:
                ids.append(self.token2id.get(s, self.token2id[self.unk]))
        return ids

    def decode(self, ids):
        toks = [self.id2token.get(i, self.unk) for i in ids]
        words, buf = [], []
        for t in toks:
            if t == self.end:
                words.append("".join(buf))
                buf = []
            else:
                buf.append(t)
        if buf:
            words.append("".join(buf))
        return " ".join(words)

    @property
    def size(self):
        return len(self.token2id) if self.token2id else 0

# ---- DataLoader 构建 ----
def build_loaders_from_ids(train_ids, val_ids, test_ids, seq_len, batch_size):
    tr = GenericTokenDataset(train_ids, seq_len)
    va = GenericTokenDataset(val_ids,  seq_len)
    te = GenericTokenDataset(test_ids, seq_len)

    loader_kwargs = {"batch_size": batch_size, "shuffle": True}
    if device.type == "cuda":
        loader_kwargs.update({"num_workers": 4, "pin_memory": True, "persistent_workers": True})

    train_loader = DataLoader(tr, **loader_kwargs)
    val_loader = DataLoader(va, batch_size=batch_size, shuffle=False,
                            num_workers=loader_kwargs.get("num_workers", 0),
                            pin_memory=loader_kwargs.get("pin_memory", False),
                            persistent_workers=loader_kwargs.get("persistent_workers", False))
    test_loader = DataLoader(te, batch_size=batch_size, shuffle=False,
                             num_workers=loader_kwargs.get("num_workers", 0),
                             pin_memory=loader_kwargs.get("pin_memory", False),
                             persistent_workers=loader_kwargs.get("persistent_workers", False))
    return train_loader, val_loader, test_loader

# ---- 采样生成文本函数 ----
@torch.no_grad()
def sample_generate(model, encode_fn, decode_fn, start_text, max_new_tokens=200, temperature=1.0, device=None, top_k=50):
    model.eval()
    device = device or next(model.parameters()).device
    start_ids = encode_fn(start_text)
    input_ids = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        if generated.size(1) > 512:
            generated = generated[:, -512:]  # 避免超长导致显存爆炸
        logits = model(generated)
        logits = logits[0, -1, :] / temperature
        if top_k is not None and top_k > 0:
            topk = torch.topk(logits, k=min(top_k, logits.size(-1)))
            indices = topk.indices
            probs = torch.softmax(topk.values, dim=0)
            next_id = indices[torch.multinomial(probs, 1)].item()
        else:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
        next_token = torch.tensor([[next_id]], device=device)
        generated = torch.cat([generated, next_token], dim=1)
    out_ids = generated[0].tolist()
    return decode_fn(out_ids)

# ---- 单次实验（Transformer）----
def run_transformer(tag, vocab_size, encode_fn, decode_fn, seq_len, batch_size, num_epochs=10, lr=1e-3):
    print(f"\n=== Transformer | {tag} | vocab_size={vocab_size} | seq_len={seq_len} | batch_size={batch_size} ===", flush=True)

    train_ids = encode_fn(train_text)
    val_ids   = encode_fn(val_text)
    test_ids  = encode_fn(test_text)

    train_loader, val_loader, test_loader = build_loaders_from_ids(train_ids, val_ids, test_ids, seq_len, batch_size)

    model = SmallTransformer(vocab_size=vocab_size, embed_size=256, num_heads=4, num_layers=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None

    start = time.time()
    for ep in range(1, num_epochs+1):
        tr = train_model(model, train_loader, optimizer, device)
        vl, vp = evaluate_model(model, val_loader, device)
        print(f"Transformer[{tag}] | Epoch {ep} | TrainLoss: {tr:.4f} | ValLoss: {vl:.4f} | PPL: {vp:.2f}", flush=True)
        if vl < best_val:
            best_val, best_state = vl, {k: v.detach().cpu() for k, v in model.state_dict().items()}
    elapsed = time.time() - start
    print(f"Transformer[{tag}] | Finished in {elapsed/60:.2f} min. Best ValLoss: {best_val:.4f} (PPL={math.exp(best_val):.2f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_ppl = evaluate_model(model, test_loader, device)
    print(f"Transformer[{tag}] | TestLoss: {test_loss:.4f} | TestPPL: {test_ppl:.2f}")

    # ----- 文本样例生成 -----
    temperatures = [0.7, 1.0, 1.3]
    start_prompt = "The "
    for T in temperatures:
        sample = sample_generate(
            model, encode_fn, decode_fn,
            start_text=start_prompt, max_new_tokens=200,
            temperature=T, device=device, top_k=50
        )
        print(f"\n[Transformer-{tag}] Sample (T={T}):\n{sample}\n", flush=True)

    return {"tag": tag, "best_val": best_val, "test_ppl": test_ppl, "time_min": elapsed/60.0}

def main():
    results = []

    # 1) 词级（word-level）
    wt = WordTokenizer().build_from_text(train_text)
    word_vocab_size = wt.size
    word_seq_len = 128
    word_bs = 32
    res_word = run_transformer(tag=f"word | V={word_vocab_size}",
                               vocab_size=word_vocab_size,
                               encode_fn=wt.encode,
                               decode_fn=wt.decode,
                               seq_len=word_seq_len,
                               batch_size=word_bs,
                               num_epochs=10, lr=1e-3)
    results.append(res_word)

    # 2) 子词级（subword-level, BPE）
    target_vocab = 4000
    bpe = SimpleBPE(vocab_size=target_vocab).train(train_text)
    sub_vocab_size = bpe.size
    sub_seq_len = 256
    sub_bs = 32
    res_sub = run_transformer(tag=f"subword(BPE) | V={sub_vocab_size}",
                              vocab_size=sub_vocab_size,
                              encode_fn=bpe.encode,
                              decode_fn=bpe.decode,
                              seq_len=sub_seq_len,
                              batch_size=sub_bs,
                              num_epochs=10, lr=1e-3)
    results.append(res_sub)

    print("\n=== Summary: Word vs Subword (Transformer) ===")
    for r in results:
        print(f"- {r['tag']}: BestValLoss={r['best_val']:.4f} (PPL={math.exp(r['best_val']):.2f}) | TestPPL={r['test_ppl']:.2f} | Time={r['time_min']:.2f} min")

if __name__ == "__main__":
    main()