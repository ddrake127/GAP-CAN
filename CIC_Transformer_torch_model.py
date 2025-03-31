import torch
import torch.nn as nn
import math, os
from torch.utils.data import Dataset, DataLoader
import pandas as pd

'''
    Create a dataset from a pandas dataframe (since we are going to need to be able to modify and splice these
    things together multiple times per experiment)
'''
class CANDatasetModified(Dataset):
    def __init__(self, df, window_size, num_features=9):
        self.df = df
        self.df = self.df.map(lambda x: x + 1) # so we can ignore 0
        self.df = self.df.to_numpy().flatten()

        # save off relevant dataset information
        self.msg_window_size = window_size
        self.num_features = num_features

        self.window_size = self.msg_window_size * self.num_features

        # calculate the total number of items we can get from this dataset
        self.num_items = len(self.df) - self.window_size - 1

    def __len__(self):
        return self.num_items
    
    def __getitem__(self, idx):
        # return tokens n : n + window_size, n+1 : n+1 + window_size
        _in = self.df[idx : idx + self.window_size]
        _out = self.df[idx + 1 : idx + 1 + self.window_size]
        return _in, _out

class CANDatasetCSV(Dataset):
    def __init__(self, file, window_size, num_features, ids, msg_lower_bound, msg_upper_bound):
        self.df = pd.read_csv(file)
        # transform the data so we can use it
        self.df['ID'] = self.df['ID'].map(lambda x: ids.index(x) + 256)
        self.df = self.df.drop(columns=["label", "category", "specific_class"])
        self.df = self.df[msg_lower_bound : msg_upper_bound]
        self.df = self.df.map(lambda x: x + 1) # so we can ignore 0
        self.df = self.df.to_numpy().flatten()

        # save off relevant dataset information
        self.msg_window_size = window_size
        self.num_features = num_features
        self.ids = ids

        self.window_size = self.msg_window_size * self.num_features

        # calculate the total number of items we can get from this dataset
        self.num_items = len(self.df) - self.window_size - 1

    def __len__(self):
        # return self.num_items - 1
        return self.num_items
    
    def __getitem__(self, idx):
        # return tokens n : n + window_size, n+1 : n+1 + window_size


        # _in = self.df[idx + 1 : idx + 1 + self.window_size]
        # _out = self.df[idx + 1 + 1 : idx + 1 + 1 + self.window_size]
        _in = self.df[idx : idx + self.window_size]
        _out = self.df[idx + 1 : idx + 1 + self.window_size]
        return _in, _out
    
class CANDatasetTxt(Dataset):
    def __init__(self, file, window_size, num_features, ids_set, msg_lower_bound, msg_upper_bound):
        self.df = pd.read_csv(file, delim_whitespace=True, header=None, names=
                ["_ts", "ts", "_id", "id", "a", "_dlc", "dlc", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"])
        self.df = self.df.drop(["_ts", "ts", "_id", "a", "_dlc", "dlc"], axis=1)
        self.df = self.df.fillna(-1)
        for c in ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]:
            self.df[c] = self.df[c].apply(lambda x: int("0x" + x, 0) if x != -1 else 256)
        self.df['id'] = self.df['id'].map(lambda x: ids_set.index(x) + 257)
        self.df = self.df[msg_lower_bound : msg_upper_bound]
        self.df = self.df.map(lambda x: x + 1) # so we can ignore 0
        self.df = self.df.to_numpy().flatten()

        # save off relevant dataset information
        self.msg_window_size = window_size
        self.num_features = num_features
        self.ids = ids_set

        self.window_size = self.msg_window_size * self.num_features

        # calculate the total number of items we can get from this dataset
        self.num_items = len(self.df) - self.window_size - 1

    def __len__(self):
        return self.num_items
    
    def __getitem__(self, idx):
        # return tokens n : n + window_size, n+1 : n+1 + window_size
        _in = self.df[idx : idx + self.window_size]
        _out = self.df[idx + 1 : idx + 1 + self.window_size]
        return _in, _out
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)
    def __len__(self):
            """Number of batches"""
            return len(self.dl)
    


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, is_casual=True):
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=(0.2 if self.training else 0.0), is_causal=is_casual)

        

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, is_casual):
        attn_output = self.self_attn(x, x, x, is_casual)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device):
        super(Transformer, self).__init__()
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x):
        embedded = self.decoder_embedding(x)
        tgt_embedded = self.positional_encoding(embedded)
        tgt_embedded = self.dropout(tgt_embedded)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, is_casual=True)

        output = self.fc(dec_output)
        return output