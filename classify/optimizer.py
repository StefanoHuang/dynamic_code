from torch.optim import AdamW
from classify.scheduler import WarmupDecayLR


class Optimizer():
    def __init__(self, args, finetune_model=[], all_model=None, steps_per_epoch=0):
        self.args = args
        self.finetune_model = finetune_model
        self.all_model = all_model
        self.steps_per_epoch = steps_per_epoch
    
    def get_optimizer_parameters(self):
        optimizer_params_groups = []
        finetune_params_set = set()
        if self.args.finetune:
            for m in self.finetune_model:
                optimizer_params_groups.append({"params": list(m.parameters()), "lr": self.args.lr * self.args.lr_scale})
                finetune_params_set.update(list(m.parameters()))
        remaining_params = [p for p in self.all_model.parameters() if p not in finetune_params_set]
        optimizer_params_groups.insert(0, {"params": remaining_params})
        return optimizer_params_groups

    def get_optimizer(self):
        optimizer = AdamW(self.get_optimizer_parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        warm_up_step = max(self.steps_per_epoch // self.args.accum_iter,1)
        #all_step = self.args.epochs * warm_up_step
        scheduler = WarmupDecayLR(optimizer, warm_up_step,self.args.d_model)
        return optimizer, scheduler