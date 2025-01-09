import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from .base_cnn import BASE_CNN
from .fully_conv import FULLY_CONV
from .lstm import LSTM
from .cldnn import CLDNN

def save(model, meta, save_path, save_name):
    model_loc = os.path.join(save_path, save_name + ".pt")
    torch.save(model.state_dict(), model_loc)

    meta_loc = os.path.join(save_path, save_name + ".json")
    with open(meta_loc, "w") as f:
        json.dump(meta, f, indent=4)

def load_pretrained(path, f_name=None, gpu=True):
    if f_name is None:
        ## derive meta path and model path
        files = os.listdir(path)
        f_name = next(f for f in files if f.endswith('.log'))[:-4]
        model_path = os.path.join(path, f_name + ".pt")
        meta_path = os.path.join(path, f_name + ".json")
    else:
        model_path = os.path.join(path, f_name + ".pt")
        meta_path = os.path.join(path, f_name + ".json")

    ## load metadata
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f) 
        state = torch.load(model_path, weights_only=True)

        if "model_type" not in meta.keys():
            meta["model_type"] = "base_cnn"

        if "input_length" not in meta.keys():
            meta["input_length"] = meta["input_shape"][-1]
    except:
        print("Job did not complete")
        return None, None

    ## create model
    try:
        if meta["model_type"] == "base_cnn":
            model = BASE_CNN(meta["input_shape"], meta["num_classes"])
        elif meta["model_type"] == "cldnn":
            try:
                model = CLDNN(meta["input_shape"], meta["num_classes"], snr_min=meta["min_snr"], snr_max=meta["max_snr"])
            except:
                model = CLDNN(meta["input_shape"], meta["num_classes"], snr_min=None, snr_max=None)
        elif meta["model_type"] == "fullconv":
            model = FULLY_CONV(meta["input_shape"], meta["num_classes"], norm="do")
        elif meta["model_type"] == "lstm":
            model = LSTM(meta["input_shape"], meta["num_classes"])
        else:
            print("Unknown model type.")
            raise
    except:
        print("Unable to create model.")

    ## load model 
    model.load_state_dict(state)

    return model, meta

def initialize_model(args, dataset):
    if args.model == "base":
        model = BASE_CNN(dataset.shape(), dataset.num_classes, n_kernels=args.n_kernels, k_size=args.k_size,
                fc_size=args.fc_size, norm="do")
    elif args.model == "fullconv":
        model = FULLY_CONV(dataset.shape(), dataset.num_classes, n_kernels=args.n_kernels, k_size=args.k_size,
                dilation=args.dilation, fc_size=args.fc_size, norm="do")
    elif args.model == "lstm":
        model = LSTM(dataset.shape(), dataset.num_classes)
    elif args.model == "cldnn":
        model = CLDNN(dataset.shape(), dataset.num_classes, snr_min=args.min_snr, snr_max=args.max_snr)
    else:
        print("Unknown model name.")
        raise
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion
