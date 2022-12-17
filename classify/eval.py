from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
import torch

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.cuda.current_device()

def eval_model(model, dataloader,  break_num=None):
    score_list = []
    pred_list = []
    y_list = []
    with torch.no_grad():
        for step, (dynamic_feature,label) in enumerate(dataloader):
            if break_num and step > break_num:
                break
            dynamic_feature = dynamic_feature.to(device)
            label = label.to(device)
            scores_pred = model(dynamic_feature)
            scores_pred = torch.nn.Softmax(dim=1)(scores_pred)
            y_pred = scores_pred.argmax(dim=-1)
            #count += 1
            #accurate += torch.eq(y_pred, label.squeeze(dim=-1)).float()
            score_list.append(scores_pred[:,1])
            pred_list.append(y_pred)
            y_list.append(label)
    y_list = torch.cat(y_list).squeeze(-1).cpu().numpy()
    score_list = torch.cat(score_list).squeeze(-1).cpu().numpy()
    pred_list = torch.cat(pred_list).squeeze(-1).cpu().numpy()
    balanced = balanced_accuracy_score(y_list, pred_list)
    acc = accuracy_score(y_list, pred_list)
    tn, fp, fn, tp = confusion_matrix(y_list, pred_list).ravel()
    sen = tp / (tp + fn)
    spec = tn / (fp + tn)
    auc = roc_auc_score(y_list, score_list)
    f1 = f1_score(y_list, pred_list, zero_division=1)
    return acc,balanced,sen,spec,auc,f1
