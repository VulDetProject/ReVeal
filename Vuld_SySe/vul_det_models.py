import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import checkpoint


class BiGRUModel(nn.Module):
    def __init__(self, emb_dim, hidden_size, num_layer):
        super(BiGRUModel, self).__init__()
        self.num_layers = num_layer
        self.num_directions = 2
        self.hid_dim = hidden_size
        self.features = nn.Sequential(
            nn.GRU(hidden_size=hidden_size, input_size=emb_dim,
                   bidirectional=True, batch_first=True, num_layers=self.num_layers),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size * self.num_layers * self.num_directions, out_features=2),
            # nn.BatchNorm1d(num_features=64),
            # nn.ReLU(True),
            # nn.Linear(in_features=64, out_features=16),
            # nn.ReLU(True),
            # nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.2)

    def forward(self, sequences, masks=None):
        embs = sequences
        output, h_n = self.features(embs)
        h_n = self.drop(torch.cat([h_n[ix, :, :] for ix in range(h_n.shape[0])], 1))
        out = self.classifier(h_n)
        return out, h_n


class BiRNNModel(nn.Module):
    def __init__(self, emb_dim, hidden_size, num_layer):
        super(BiRNNModel, self).__init__()
        self.num_layers = num_layer
        self.num_directions = 2
        self.hid_dim = hidden_size
        self.features = nn.Sequential(
            nn.RNN(hidden_size=hidden_size, input_size=emb_dim,
                    bidirectional=True, batch_first=True, num_layers=self.num_layers),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size * self.num_directions * self.num_layers, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.5)

    def forward(self, sequences, masks=None):
        embs = sequences
        output, (h_n) = self.features(embs)
        h_n = self.drop(torch.cat([h_n[ix, :, :] for ix in range(h_n.shape[0])], 1))
        out = self.classifier(h_n)
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, emb_dim, hidden_size, num_layer):
        super(BiLSTMModel, self).__init__()
        self.num_layers = num_layer
        self.num_directions = 2
        self.hid_dim = hidden_size
        self.features = nn.Sequential(
            nn.LSTM(hidden_size=hidden_size, input_size=emb_dim,
                    bidirectional=True, batch_first=True, num_layers=self.num_layers),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size * self.num_directions * self.num_layers, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.5)

    def forward(self, sequences, masks=None):
        embs = sequences
        output, (h_n, c_n) = self.features(embs)
        h_n = self.drop(torch.cat([h_n[ix, :, :] for ix in range(h_n.shape[0])], 1))
        out = self.classifier(h_n)
        return out


class ConvModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, pad_idx):
        super(ConvModel, self).__init__()
        self.num_out_kernel = 512
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=pad_idx)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.num_out_kernel, kernel_size=(9, emb_dim)),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.num_out_kernel, out_features=64),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=16),
            nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.5)

    def forward(self, sentences, masks=None):
        emb = self.embedding(sentences)
        emb = emb.unsqueeze(dim=1)
        cs = self.features(emb)
        cs = cs.view(sentences.shape[0], self.num_out_kernel, -1)
        rs = self.drop(nn.functional.max_pool1d(cs, kernel_size=cs.shape[-1]))
        rs = rs.view(sentences.shape[0], self.num_out_kernel)
        soft = self.classifier(rs)
        return soft, rs


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.ff = nn.Linear(in_features=hidden_dim, out_features=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, contexts, context_masks=None):
        """
        :param contexts: (batch_size, seq_len, n_hid)
        :param context_masks: (batch_size, seq_len)
        :return: (batch_size, n_hid), (batch_size, seq_len)
        """
        out = self.ff(contexts)
        out = out.view(contexts.size(0), contexts.size(1))
        if context_masks is not None:
            masked_out = out.masked_fill(context_masks, float('-inf'))
        else:
            masked_out = out
        attn_weights = self.softmax(masked_out)
        out = attn_weights.unsqueeze(1).bmm(contexts)
        out = out.squeeze(1)
        return out, attn_weights


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, pad_idx):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=pad_idx)
        self.emb_transform = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=128),
            nn.ReLU(True)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4,
            dim_feedforward=2 * emb_dim,
            dropout=0.2, activation='relu')
        encoder_norm = nn.LayerNorm(128)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=4, norm=encoder_norm)
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=self.num_out_kernel, kernel_size=(9, emb_dim)),
        #     nn.ReLU(True)
        # )
        self.combiner = Attention(128)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=16),
            nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.3)

    def forward(self, sentences, masks=None):
        emb = self.embedding(sentences)
        emb = self.emb_transform(emb)
        encoded = self.encoder(emb.transpose(*(0, 1))).transpose(*(0, 1))
        combined, _ = self.combiner(encoded)
        combined = self.drop(combined)
        output = self.classifier(combined)
        return output, combined


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBiGRUModel(nn.Module):
    def __init__(self, emb_dim, hidden_size, num_layer):
        super(TransformerBiGRUModel, self).__init__()
        self.num_layers = num_layer
        self.num_directions = 2
        self.hid_dim = hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=4,
            dim_feedforward=2 * emb_dim,
            dropout=0.2, activation='relu')
        encoder_norm = nn.LayerNorm(emb_dim)
        self.encoder = nn.Sequential(
            PositionalEncoding(emb_dim),
            nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=4, norm=encoder_norm)
        )
        self.features = nn.Sequential(
            nn.GRU(hidden_size=hidden_size, input_size=emb_dim,
                   bidirectional=True, batch_first=True, num_layers=self.num_layers),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size * self.num_layers * self.num_directions, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.5)

    def forward(self, sequences, masks=None):
        embs = sequences
        embs = self.encoder(embs.transpose(*(0,1))).transpose(*(0, 1))
        output, h_n = self.features(embs)
        h_n = self.drop(torch.cat([h_n[ix, :, :] for ix in range(h_n.shape[0])], 1))
        out = self.classifier(h_n)
        return out


class TransformerAttentionModel(nn.Module):
    def __init__(self, emb_dim, hidden_size, num_layer):
        super(TransformerBiGRUModel, self).__init__()
        self.num_layers = num_layer
        self.num_directions = 2
        self.hid_dim = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=hidden_size),
            nn.ReLU(),
            PositionalEncoding(emb_dim),
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                            d_model=hidden_size, nhead=4,
                            dim_feedforward=2 * hidden_size,
                            dropout=0.2, activation='relu'
                ),
                num_layers=4,
                norm=nn.LayerNorm(hidden_size)
            )
        )
        self.features = nn.Sequential(
            Attention(emb_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.5)

    def forward(self, sequences, masks=None):
        embs = sequences
        embs = self.encoder(embs.transpose(*(0,1))).transpose(*(0, 1))
        output, h_n = self.features(embs)
        h_n = self.drop(output)
        out = self.classifier(h_n)
        return out


class TransformerPoolModel(nn.Module):
    def __init__(self, emb_dim, hidden_size, num_layer):
        super(TransformerPoolModel, self).__init__()
        self.num_layers = num_layer
        self.num_directions = 2
        self.hid_dim = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=hidden_size),
            nn.ReLU(),
            PositionalEncoding(hidden_size),
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                            d_model=hidden_size, nhead=4,
                            dim_feedforward=2 * hidden_size,
                            dropout=0.2, activation='relu'
                ),
                num_layers=4,
                norm=nn.LayerNorm(hidden_size)
            )
        )
        # self.features = nn.Sequential(
        #     Attention(emb_dim),
        # )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.5)

    def forward(self, sequences, masks=None):
        embs = sequences
        embs = self.encoder(embs.transpose(*(0,1))).transpose(*(0, 1))
        output = embs[:, 0, :]
        h_n = self.drop(output)
        out = self.classifier(h_n)
        return out