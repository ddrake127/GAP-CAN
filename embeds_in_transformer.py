import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import sys
sys.path.append('../')
from Transformer_torch_model import *

class Transformer_From_Embedding(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device):
        super(Transformer_From_Embedding, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, is_causal=True):
        tgt_embedded = self.positional_encoding(x)
        tgt_embedded = self.dropout(tgt_embedded)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, is_causal)

        output = self.fc(dec_output)
        return output