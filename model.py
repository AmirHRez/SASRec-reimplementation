import numpy as np
import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        # 1x1 Convolution
        self.conv1 = nn.Conv1d(
            in_channels=hidden_units, out_channels=hidden_units, kernel_size=1
        )
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=hidden_units, out_channels=hidden_units, kernel_size=1
        )
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # inputs shape: (batch size, length, channels)
        # conv1d expected shape: (batch size, channels, length)
        x = inputs.transpose(-1, -2)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = x.transpose(-1, -2)
        outputs = x + inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.norm_first = args.norm_first

        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, args.hidden_units, padding_idx=0
        )
        self.pos_emb = torch.nn.Embedding(
            args.maxlen + 1, args.hidden_units, padding_idx=0
        )
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(
                torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            )
            self.attention_layers.append(
                torch.nn.MultiheadAttention(
                    args.hidden_units, args.num_heads, dropout=args.dropout_rate
                )
            )
            self.forward_layernorms.append(
                torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            )

        # self.pos_sigmoid = torch.nn.Sigmoid()
        # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):

        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim**0.5

        positions = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        positions *= log_seqs != 0

        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        time_len = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((time_len, time_len), dtype=torch.bool, device=self.dev)
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](
                    x, x, x, attn_mask=attention_mask
                )
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](
                    seqs, seqs, seqs, attn_mask=attention_mask
                )
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(
            torch.LongTensor(item_indices).to(self.dev)
        )  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # (U, I)
