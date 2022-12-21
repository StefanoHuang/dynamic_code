from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
import torch

from pretrain.pretrain_dataset import BuildDataloader, GraphDataset
from utils.utils import DecodingBCEWithMaskLoss, GenerateOOV, AverageMeter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def mask_tokens(inputs,args):
    labels = inputs.clone()
    masked_indices = torch.bernoulli(torch.full(labels.shape[:2], args.mlm_probability)).bool()
    labels[~masked_indices] = 0
    labels[masked_indices] = 1
    indices_replaced = torch.bernoulli(torch.full(labels.shape[:2], 0.8)).bool() & masked_indices
    inputs[indices_replaced] = 0
    indices_random = torch.bernoulli(torch.full(labels.shape[:2], 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randn(labels.shape, dtype=torch.float32)
    random_words = random_words.to(inputs.device)
    inputs[indices_random] = random_words[indices_random]
    return inputs, masked_indices

def eval_model(model, args, dataloader, loss_fn):
    with torch.no_grad():
        eval_losses = AverageMeter()
        for step, filelist in enumerate(dataloader):
            graphtemp = GraphDataset(args=args, filepath=filelist)
            graphloader = BuildDataloader(dataset=graphtemp,batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
            for substep, graph in enumerate(graphloader):
                graph = graph.to(torch.float32)
                graph = graph.to(device)
                origin_graph = graph.clone()
                graph,masked_indices = mask_tokens(graph,args)
                graph_pred = model(graph.transpose(0,1))
                graph_pred = graph_pred.transpose(0,1)   
                graph_pred[~masked_indices] = origin_graph[~masked_indices] 
                generate_loss = loss_fn(graph_pred, origin_graph)
                batch = graph_pred.size(0)
                total_loss = generate_loss  
                eval_losses.update(total_loss.item(),batch)    
    return eval_losses.avg
