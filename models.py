import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        if num_layers == 1:
            dropout = 0.0  # 避免 warning
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers,
                          dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        if num_layers == 1:
            dropout = 0.0  # 避免 warning
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(4096, embed_size)  # 简单的可学习位置编码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dropout=dropout
            # 注意：老版本不支持 batch_first 参数，这里移除
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        bsz, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.embed(x) * (self.embed.embedding_dim ** 0.5) + self.pos_embed(positions)  # (batch, seq_len, embed)

        # Transformer (老版本) 期望 src 形状为 (seq_len, batch, embed)
        x = x.transpose(0, 1)  # (seq_len, batch, embed)

        # 因果掩码，形状 (seq_len, seq_len)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        x = self.transformer(x, mask=causal_mask)  # 输出形状仍为 (seq_len, batch, embed)

        x = x.transpose(0, 1)  # (batch, seq_len, embed)
        return self.fc(x)

