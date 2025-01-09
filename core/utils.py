import os
from datetime import datetime

import numpy as np
import torch

def set_random_seed(seed):
    if seed == -1:
        seed = np.random.randint(1e9)
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device(args=None):
    if args is not None and args.no_gpu:
        device = "cpu"
    elif torch.cuda.is_available():
        if args is not None and 'cuda' in args and args.cuda > 0:
            device = f"cuda:{args.cuda}"
        else:
            device = "cuda"
    else:
        device = "cpu"
    return device

def get_num_parameters(model, trainable=False):
    if trainable:
        return sum(curr_param.numel() for curr_param in model.parameters() if curr_param.requires_grad)
    else:
        return sum(curr_param.numel() for curr_param in model.parameters())

def create_paths(args):
    if args.save_path[-1] == "/":
        args.save_path = args.save_path[:-1]
    args.save_path = args.save_path + "_" + str(args.input_length) + "samp_lr" + str(args.lr) + "_seed" + str(args.seed) 
    if os.path.exists(args.save_path):
        ## modify pathname with date (DD-MM-YY) and time (H-M-S)
        t = datetime.today()
        args.save_path = args.save_path + "_" + t.strftime("%d-%m-%y-%H-%M-%S")
        os.makedirs(args.save_path)
    else:
        os.makedirs(args.save_path)
    return args.save_path