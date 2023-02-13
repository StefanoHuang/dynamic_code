import torch.nn as nn
from pretrain.encoder import GraphEncoder, PositionalEncoding,SeqEncoder
import torch
import torch.nn.functional as F

def generate_square_subsequent_mask(sz: int, device='cpu') -> torch.Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
class tokenModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.norm = nn.LayerNorm(args.d_model)
        self.lin = nn.Linear(args.d_model,args.d_model)
        self.dropout = nn.Dropout(self.args.dropout_prob)
    def forward(self, x):
        x = self.dropout(x)
        x = self.norm(x)
        x = self.lin(x)
        return x    

class classifiedModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lin = nn.Linear(args.d_model,self.args.num_classes)
        self.dropout = nn.Dropout(self.args.dropout_prob)
        self.norm = nn.BatchNorm1d(args.d_model)
    def forward(self, x):
        x = self.dropout(x)
        x = self.norm(x)
        x = self.lin(x)
        return F.softmax(x)    

class MaskedBrainNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 创建encoder
        self.graph_encoder = GraphEncoder(args)
        self.tokenmodel = tokenModel(args)
    def forward(self, x):
        x = self.graph_encoder(x,False)
        x = self.tokenmodel(x)
        return x

class PerlBrainNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 创建encoder
        self.graph_encoder = GraphEncoder(args)
        self.tokenmodel = tokenModel(args)
    def forward(self, x):
        x = self.graph_encoder(x,False)
        x = self.tokenmodel(x)
        return F.softmax(x)   

class NGPBrainNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 创建encoder
        self.graph_encoder = GraphEncoder(args)
        self.classifymodel = classifiedModel(args)
        self.graph_cls = nn.Parameter(torch.zeros(1, 1, args.d_model))
    def forward(self, x):
        batchsize = x.size(1)  
        graph_cls = self.graph_cls.expand(1,batchsize,-1).to(x.device)
        x = torch.cat([graph_cls,x], dim=0)
        x = self.graph_encoder(x,False)
        x = x[0]
        x = self.classifymodel(x)
        return F.softmax(x)   

class ClassifiedSeqBrainNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 创建encoder
        self.seq_encoder = SeqEncoder(args)
        self.classifymodel = classifiedModel(args)
        self.seq_cls = nn.Parameter(torch.zeros(1, 1, args.d_model))
        self.cls_mask = torch.ones(1, 1)
        self.pos_encoder = PositionalEncoding(0.1, args.d_model)
    def forward(self, x):
        seq_mask = (x == -1e9)[:,:,0].T
        batchsize = x.size(1) 
        graph_cls_mask = (self.cls_mask == 0).expand(batchsize,-1).to(x.device)
        #square_mask = generate_square_subsequent_mask(x.size(0)+1,x.device)
        #square_mask[0,:] = 0.0
        all_mask = torch.cat([graph_cls_mask,seq_mask],dim=1)
        seq_cls = self.seq_cls.expand(1,batchsize,-1)
        x = torch.cat([seq_cls,x], dim=0)
        x = self.seq_encoder(src = x, src_key_padding_mask=all_mask)
        x = x[0]
        x = self.classifymodel(x)
        return x       