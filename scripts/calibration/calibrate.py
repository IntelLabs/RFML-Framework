import os
import sys
sys.path.append(os.path.abspath("."))

from argparse import ArgumentParser

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from data.synthetic import SyntheticDataset
from core.calibration import CalibratedModel
from core.utils import get_device, set_random_seed


def parse_args():
    p = ArgumentParser()
    p.add_argument('--lr', type=float, help="Learning rate")
    p.add_argument('--model-dir', required=True, help='Directory containing model')
    p.add_argument('--model-name', required=True, help="Name of model file without file extension")
    p.add_argument('--data-dir', required=True, help="Directory containing data")
    p.add_argument('--config', required=True, help="Data config")
    p.add_argument('--label', default='mod_scheme', choices=['mod_scheme', 'tx_id'], help='Class label (def: %(default)s)')
    p.add_argument('--n-eval', default=1000, help='Number of examples to evaluate (def: %(default)s)')
    p.add_argument('--batch-size', default=128, help='Batch size (def: %(default)s)')
    
    return p.parse_args()

device = get_device()
args = parse_args()

## get model params
with open(os.path.join(args.model_dir, args.model_name + '.json')) as f:
    params = json.load(f)
args.model = params['model_type']
args.input_length = params['input_shape'][-1]
args.label = params['label']

## load dataset
dataset = SyntheticDataset(path=args.data_dir, config=args.config, example_len=args.input_length, n_examples=args.n_eval, 
                    from_archive=False, meta_filter=['modulation', 'snr', 'fo'])
dataloader = DataLoader(dataset, batch_size=args.batch_size)

## load model
model = CalibratedModel(args.model_dir, dataset.num_classes, f_name=args.model_name, device=device)
model.get_set_temperature(dataloader, lr=args.lr)
print("Temperature: ", model.temperature.item())


