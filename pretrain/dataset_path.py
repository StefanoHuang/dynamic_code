from utils.logger import Log

logger = Log(__name__).getlog()

def dataset_path(dataset_name):
    if dataset_name == "abide":
        d_path = "/data/abide"
    elif dataset_name == "hcp":
        d_path = "/data/hcp"
    elif dataset_name == "mdd":
        d_path = "/data/mdd"
    elif dataset_name == "other":
        pass
    else:
        logger.info(f"dataset name is wrong")
        assert False

    return d_path