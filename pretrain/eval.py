from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
import torch

from pretrain.pretrain_dataset import BuildDataloader, GraphDataset, SequenceDataset
from utils.utils import DecodingBCEWithMaskLoss, GenerateOOV, AverageMeter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def shuffle_single(single, prob):
    single = single.unsqueeze(0)
    shuffle_indices = torch.bernoulli(torch.full((1,single.shape[1]), prob)).bool()
    origin_index = torch.tensor([i for i in range(shuffle_indices.size(1)) if shuffle_indices[0,i]])
    shuffle_index=torch.randperm(origin_index.size(0))
    single[shuffle_indices] = single[shuffle_indices][shuffle_index]
    order = torch.arange(0, single.size(1)).unsqueeze(0)
    order[shuffle_indices] = order[shuffle_indices][shuffle_index]
    return single.squeeze(0), order
def shuffle_tokens(inputs,args):
    feat_list = []
    order_list = []
    for i in range(inputs.size(0)):
        shuffled, label = shuffle_single(inputs[i],args.mlm_probability)
        feat_list.append(shuffled)
        order_list.append(label)
    inputs = torch.stack(feat_list)
    labels = torch.stack(order_list).squeeze(1)
    return inputs, labels
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
def eval_model(model, args, dataloader):
    pred_list = []
    y_list = []
    with torch.no_grad():
        for step, filelist in enumerate(dataloader):
            graphtemp = GraphDataset(args=args, filepath=filelist)
            graphloader = BuildDataloader(dataset=graphtemp,batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
            for substep, graph in enumerate(graphloader):
                graph = graph.to(torch.float32)
                graph = graph.to(device)
                graph,labels = shuffle_tokens(graph,args)
                graph_pred = model(graph.transpose(0,1))
                graph_pred = graph_pred.transpose(0,1)   
                y_pred = graph_pred.argmax(dim=-1)
                pred_list.append(y_pred)
                y_list.append(labels)
    y_list = torch.cat(y_list).squeeze(-1).cpu().numpy()
    pred_list = torch.cat(pred_list).squeeze(-1).cpu().numpy()
    return (y_list == pred_list).sum().item() / (y_list.shape[0]*y_list.shape[1])

def eval_model_loss(model, args, dataloader, loss_fn):
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

def eval_model_acc(model, args, dataloader):
    pred_list = []
    y_list = []
    with torch.no_grad():
        for step, filelist in enumerate(dataloader):
            seqtemp = SequenceDataset(args=args, filepath=filelist)
            seqloader = BuildDataloader(dataset=seqtemp,batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
            for substep, (seq,labels) in enumerate(seqloader):
                seq = seq.to(torch.float32)
                seq = seq.to(device)
                labels = labels.to(device)
                scores_pred = model(seq.transpose(0,1))
                scores_pred = torch.nn.Softmax(dim=1)(scores_pred)
                y_pred = scores_pred.argmax(dim=-1)
                #count += 1
                #accurate += torch.eq(y_pred, label.squeeze(dim=-1)).float()
                pred_list.append(y_pred)
                y_list.append(labels)
    y_list = torch.cat(y_list).squeeze(-1).cpu().numpy()
    pred_list = torch.cat(pred_list).squeeze(-1).cpu().numpy()
    acc = accuracy_score(y_list, pred_list)
    return acc