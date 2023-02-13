from asyncio.log import logger
import math
import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder,TransformerEncoderLayer
from pretrain.TransformerEncoderFinalLayer import TransformerEncoderFinalLayer
class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=512):
        """
        initialization of required variables and functions
        :param dropout: dropout probability
        :param dim: hidden size
        :param max_len: maximum length
        """
        super(PositionalEncoding, self).__init__()
        # positional encoding initialization
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        # term to divide
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        # sinusoidal positional encoding
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        """
        create positional encoding
        :param emb: word embedding
        :param step: step for decoding in inference
        :return: positional encoding representation
        """
        emb *= math.sqrt(self.dim)
        emb = emb + self.pe[: emb.size(0)]  # [len, batch, size]
        emb = self.dropout(emb)
        return emb

class GraphEncoder(torch.nn.Module):
    def __init__(self, args):
        super(GraphEncoder,self).__init__()
        encoder_layer = TransformerEncoderLayer(
                    args.d_model, args.nhead, args.d_model*4,dropout=0.1,activation='gelu',
                    norm_first=True)
        self.final_layer = TransformerEncoderFinalLayer(
                    args.d_model, args.nhead, args.d_model*4,dropout=0.1,activation='gelu',
                    norm_first=True)
        self.encoder = TransformerEncoder(encoder_layer,args.graph_layer-1)
    def forward(self,x, need_weight = False):
        x = self.encoder(x)
        output,weight = self.final_layer(x)
        if need_weight:
            return output,weight
        else:
            return output

class SeqEncoder(torch.nn.Module):
    def __init__(self, args):
        super(SeqEncoder,self).__init__()
        encoder_layer = TransformerEncoderLayer(
                    args.d_model, args.nhead, args.d_model*4,dropout=0.1,activation='gelu',
                    norm_first=True)
        self.encoder = TransformerEncoder(encoder_layer,args.seq_layer)
        self.pos_encoder = PositionalEncoding(0.1, args.d_model)
    def forward(self,src,src_key_padding_mask):
        src = self.pos_encoder(src)
        x = self.encoder(src=src,src_key_padding_mask=src_key_padding_mask)
        return x
