
import logging
import numpy as np
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath("."))

import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import initialize_dataset

from core.train import train
from models import load_pretrained, save
from core.logging import init_logger
from core.arguments import get_args
from core.utils import set_random_seed, get_device

args = get_args(action="tune")

## set random seed
set_random_seed(args.seed)

## Create Loggers
init_logger(args.results_path, [args.save_name])
tb_writer = SummaryWriter(args.results_path, flush_secs=60 * 1)  # flush every 1 minutes
logger = logging.getLogger(args.save_name)

## get device
device = get_device(args)

## get model params
param_file = [f for f in os.listdir(args.results_path) if ".json" in f]
assert len(param_file) == 1
with open(os.path.join(args.results_path, param_file[0])) as f:
    params = json.load(f)
args.model = params['model_type']
args.input_length = params['input_shape'][-1]
args.label = params['label']

## Load datasets
train_set = initialize_dataset(args, type="SyntheticDataset", split='train', meta_filter=['modulation', 'snr', 'fo'])
msg = "Train set loaded containing {} examples of length {} samples, and {} output classes."
logger.info(msg.format(len(train_set), args.input_length, train_set.num_classes))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

val_set = initialize_dataset(args, type="SyntheticDataset", split='val', meta_filter=['modulation', 'snr', 'fo']) 
msg = "Validation set loaded containing {} examples of length {} samples, and {} output classes."
logger.info(msg.format(len(val_set), args.input_length, val_set.num_classes)) 
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

# ## load model, replace last layer
model, meta = load_pretrained(args.results_path, f_name=args.model)
model.reset_out(num_classes=train_set.num_classes)
print(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = nn.CrossEntropyLoss()

## train model 
cp_name = os.path.join(args.results_path, args.save_name + "_checkpoint.pt")
model, best_epoch = train(model, train_loader, val_loader, args.n_epochs, 
                        optimizer, criterion, logger, tb_writer, cp_name, 
                        device, args.reload_best)

## save final model and associated metadata
meta = {"input_shape":train_set.shape(),
        "num_classes":train_set.num_classes,
        "label_map":train_set.label_map,
        "best_epoch":best_epoch}
meta.update(vars(args))

save(model, meta, args.results_path, args.save_name)

## clean up
tb_writer.flush()
tb_writer.close()
