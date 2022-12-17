import torch
torch.cuda.current_device()
import torch.nn as nn
from classify.encoder import GraphEncoder,SeqEncoder,generate_square_subsequent_mask

import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.normalization  import LayerNorm
class NestedTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 创建encoder
        self.graph_encoder = GraphEncoder(args)
        self.seq_encoder = SeqEncoder(args)
        # 创建decoder
        # 定义decoder tokenizer
        self.lin1 = nn.Linear(200,64)
        self.lin2 = nn.Linear(64,16)
        self.predict = nn.Linear(16,args.num_classes)
        self.b1 = nn.BatchNorm1d(num_features=64)
        self.b2 = nn.BatchNorm1d(num_features=16)
        self.graph_cls = nn.Parameter(torch.zeros(1, 1, args.d_model))
        self.seq_cls = nn.Parameter(torch.zeros(1, 1, args.d_model))
        self.cls_mask = torch.ones(1, 1)
        self.dropout_prob = args.dropout_prob
        self.cls_attn = MultiheadAttention(200,4,dropout=args.dropout_prob)
        self.cls_norm = LayerNorm(200, eps=1e-5)
    def forward(self, x):
        batchsize = x.size(1)        
        all_mask = (x == -1e9)[:,:,0,0].T
        graph_cls = self.graph_cls.expand(1,batchsize,-1).to(x.device)
        graph_cls_mask = (self.cls_mask == 0).expand(batchsize,-1).to(x.device)
        all_mask = torch.cat([graph_cls_mask,all_mask],dim=1)
        N = x.size(0)
        sequence = []
        for i in range(N):
            single_graph = x[i,:,:,:]
            single_graph = single_graph.transpose(0,1)
            single_graph = torch.cat([graph_cls,single_graph], dim=0)
            single_out = self.graph_encoder(single_graph,False)
            sequence.append(single_out)
            graph_cls = single_out.unsqueeze(0) 
            if i == 0:
                cls_emb = graph_cls
            else:
                cls_emb = torch.cat([cls_emb,graph_cls],dim=0)
                cls_srcmask = generate_square_subsequent_mask(cls_emb.shape[0],cls_emb.device)
                cls_emb = cls_emb + self.cls_attn(self.cls_norm(cls_emb),self.cls_norm(cls_emb),self.cls_norm(cls_emb),
                attn_mask=cls_srcmask, need_weights=False)[0]
        sequence = torch.stack(sequence,dim=0)
        seq_mask = generate_square_subsequent_mask(sequence.size(0)+1,sequence.device)
        seq_cls = self.seq_cls.expand(1,batchsize,-1)
        sequence = torch.cat([seq_cls,sequence], dim=0)
        seq_embs = self.seq_encoder(src=sequence,mask=seq_mask,src_key_padding_mask=all_mask)
        #以下为分类代码
        seq_embs = seq_embs[0]
        x = self.lin1(seq_embs)
        x = self.b1(x)        
        x = F.leaky_relu(x)
        x = F.dropout(x,self.dropout_prob, training=self.training)
        x = self.lin2(x)
        x = self.b2(x)        
        x = F.leaky_relu(x)
        x = F.dropout(x,self.dropout_prob, training=self.training)
        return self.predict(x)


