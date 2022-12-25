import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from collections import defaultdict

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Cross_attention(nn.Module):
    # 获得交叉attention的权重
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()
        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size
        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        assert query_inputs.dim() in [2, 3]
        assert key_inputs.dim() == 3
        assert attention_mask.dim() == 2
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores

class DecodingBCEWithMaskLoss(nn.Module):
    def __init__(self, fixed_size):
        super().__init__()
        self.one = torch.Tensor([1.])
        self.fixed_size = fixed_size

    def forward(self, pred, target, copy_target, loss_mask, unk_idx=100):
        device = pred.device
        fixed_vacab = F.one_hot(target, num_classes=self.fixed_size).float().to(device)
        target_scores = torch.cat([fixed_vacab, copy_target], dim=-1)
        target_scores[:, :, unk_idx] = 1e-12
        losses = F.binary_cross_entropy_with_logits(pred, target_scores, reduction="none")
        losses *= loss_mask.unsqueeze(-1)
        count = torch.max(torch.sum(loss_mask), self.one.to(device))
        loss = torch.sum(losses) / count
        return loss

def sequence_accuracy(logits, targets):
    assert logits.dim() == 2
    assert targets.dim() == 1
    pred = logits.argmax(dim=1)
    return (pred == targets).sum().item() / targets.shape[0]

class GenerateOOV:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.fixed_vocab = len(tokenizer)
    
    def generate(self, pred_ids, oov_vocab):
        final_token = []    # 对原始预测结果进行去重
        pred_token = []
        for ids in pred_ids:
            if ids == self.tokenizer.eos_token_id: break
            elif ids >= self.fixed_vocab:
                ids_oov = ids - self.fixed_vocab
                try:
                    token = oov_vocab[ids_oov]
                except:
                    token = "Failed"
            else:
                token = self.tokenizer.decode([ids], skip_special_tokens=True)
            pred_token.append(token)
            if len(final_token) > 0 and final_token[-1] == token:
                continue
            final_token.append(token)
        pred_token = ''.join(pred_token).replace('#', '')
        final_token = ''.join(final_token).replace('#', '')
        return final_token, pred_token

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def data_list_to_excel(data_list, output_path):
    assert isinstance(data_list, list)
    data_dict = defaultdict(list)
    for data in data_list:
        for key, value in data.items():
            data_dict[key].apepnd(value)
    dataFrame_rslt = pd.DataFrame(data_dict)
    dataFrame_rslt.to_excel(output_path, index=False)

