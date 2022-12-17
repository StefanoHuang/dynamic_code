import torch
torch.cuda.current_device()
from utils.utils import seed_everything
from classify.config import args_cls
from utils.logger import Log

logger = Log(__name__).getlog()

# def select_args():
#     def _select_args(args):   
#         cfg = {}
#         for arg in vars(args):
#             cfg[arg] = getattr(args, arg)
#         return cfg
#     if args_gen.task == "gen":
#         cfg = _select_args(args_gen)
#     elif args_gen.task == "cls":
#         cfg = _select_args(args_cls)
#     return cfg


def main():
    args = args_cls
    logger.info(f"args {args}")
    seed_everything(args.seed)
    from classify.classify import Classification
    task = Classification()
    task.run(args)

if __name__ == "__main__":
    main()
    