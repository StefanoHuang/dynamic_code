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