import torch.nn as nn
from pretrain.encoder import GraphEncoder,SeqEncoder
import torch
import torch.nn.functional as F

class MaskedBrainNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 创建encoder
        self.graph_encoder = GraphEncoder(args)

    def forward(self, x):
        x = self.graph_encoder(x,False)
        return x

class ClassifiedBrainNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 创建encoder
        self.graph_encoder = GraphEncoder(args)
        self.norm = nn.LayerNorm(args.d_model)
        self.lin = nn.Linear(args.d_model,args.d_model)
    def forward(self, x):
        x = self.graph_encoder(x,False)
        x = F.dropout(self.args.dropout_prob)
        x = self.norm(x)
        x = self.lin(x)
        return F.softmax(x)