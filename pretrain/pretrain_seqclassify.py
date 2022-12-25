from sklearn.model_selection import StratifiedKFold
import torch
from pretrain.pretrain_model import *
from utils.logger import Log
from torch.cuda.amp import autocast, GradScaler
from pretrain.optimizer import Optimizer
from pretrain.pretrain_dataset import PretrainDataset,GraphDataset,SequenceDataset,BuildDataloader
#from classify.dataset_path import dataset_path
from utils.utils import sequence_accuracy, GenerateOOV, AverageMeter
import os
import time
import torch.nn as nn
from pretrain.eval import eval_model, eval_model_acc
import numpy as np
from sklearn.model_selection import train_test_split

logger = Log(__name__).getlog()

class Pretrain():
    def __init__(self):
        logger.info(f"hello {__name__}")
        logger.info(f"Pretrain is processing")
        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def run(self, args):
        scaler = GradScaler()
        # 数据加载
        dataset = PretrainDataset(args)
        train_set, dev_set = train_test_split(dataset,test_size=0.005)
        #train_set = dataset[train_index]
        #dev_set = dataset[test_index]
        #dec_lossfn = nn.MSELoss()
        #dec_lossfn = nn.BCELoss(reduction='mean')
        dec_lossfn = nn.CrossEntropyLoss(reduction='mean')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ClassifiedSeqBrainNetwork(args)
        # 加载模型
        if args.resume_file:
            ori_model_state_dict = torch.load(args.resume_file)
            model.load_state_dict(ori_model_state_dict, strict=True)
            logger.info(f"successfully load the previous checkpoint from {args.resume_file}")
        model = model.to(device)  # model中的tensor不会转到devcie，只有变量才会转到devcie
        train_loader = BuildDataloader(dataset=train_set, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
        dev_loader = BuildDataloader(dataset=dev_set, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
        # 优化器加载
        steps_per_epoch = len(train_loader)
        optimizer_class = Optimizer(args, [model], all_model=model,steps_per_epoch=steps_per_epoch)
        optimizer, scheduler = optimizer_class.get_optimizer()
        min_loss = 1e6
        max_dev_acc = 0
        best_dev_epoch = 0
        # 创建输出目录
        output_path = f"{args.output}/{args.exp_name}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if args.exp_name == "debug":
            save_prefix = "debug-"
            if args.resume_file:
                save_prefix += args.resume_file.split("/")[-1]
        else:
            save_prefix = time.strftime("%Y%m%d%H%M%S", time.localtime())
        train_model_output_path = f"{output_path}/{save_prefix}-min_loss.pth"
        dev_model_output_path = f"{output_path}/{save_prefix}-dev.pth"

        for epoch in range(args.epochs):
            if "train" in args.running_type:
                epoch_losses = AverageMeter()
                model.train()
                optimizer.zero_grad()
                for step, filelist in enumerate(train_loader):
                    seqtemp = SequenceDataset(args=args, filepath=filelist)
                    seqloader = BuildDataloader(dataset=seqtemp,batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
                    for substep, (seq,labels) in enumerate(seqloader):
                        seq = seq.to(torch.float32)
                        seq = seq.to(device)
                        labels = labels.to(device)
                        #origin_graph = graph.clone()
                        with autocast():
                            score_pred = model(seq.transpose(0,1))
                            generate_loss = dec_lossfn(score_pred, labels)
                            total_loss = generate_loss
                        assert torch.isnan(total_loss).sum() == 0
                        # 正常梯度回传
                        # optimizer.zero_grad()
                        # total_loss.backward()
                        # optimizer.step()
                        # scheduler.step()
                        scaler.scale(total_loss).backward()
                        if ((substep + 1) % args.accum_iter == 0) or ((substep + 1) == len(seqloader)):
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            optimizer.zero_grad()

                        batch = score_pred.size(0)
                        epoch_losses.update(total_loss.item(), batch)
                        if substep % 100 == 0:
                            logger.info(f"{epoch}-{step}--{substep} | avg_loss: {epoch_losses.avg}, cur_loss: {total_loss.item()}")

                    if epoch_losses.avg < min_loss and args.exp_name != "debug":
                        min_loss = epoch_losses.avg
                        torch.save(model.state_dict(), train_model_output_path)

            if "dev" in args.running_type:  # TO LOOK
                model.eval()
                dev_acc = eval_model_acc(model, args, dev_loader)
                logger.info(f"epoch: {epoch}, dev_acc: {dev_acc}")
                if max_dev_acc < dev_acc:
                    max_dev_acc = dev_acc
                    best_dev_epoch = epoch
                    if args.exp_name != "debug":
                        torch.save(model.state_dict(), dev_model_output_path)
        logger.info(f"best dev acc epoch: {best_dev_epoch}, dev_acc: {max_dev_acc}")

        