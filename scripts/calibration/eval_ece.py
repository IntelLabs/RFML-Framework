
import os
import sys
sys.path.append(os.path.abspath("."))

from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from data.synthetic import SyntheticDataset
from core.evaluate import get_logits
from core.calibration import get_metrics_torch
from models import load_pretrained
from core.utils import get_device, set_random_seed

def parse_args():
    p = ArgumentParser()
    p.add_argument('--model-dir', required=True, help='Directory containing model')
    p.add_argument('--data-dir', required=True, help="Directory containing data")
    p.add_argument('--model-name', required=True, help="Name of model file without file extension")
    p.add_argument('--config', required=True, help="Data config")
    p.add_argument('--label', default='mod_scheme', choices=['mod_scheme', 'tx_id'], help='Class label (def: %(default)s)')
    p.add_argument('--n-eval', default=1000, help='Number of examples to evaluate (def: %(default)s)')
    p.add_argument('--batch-size', default=128, help='Batch size (def: %(default)s)')
    
    return p.parse_args()

device = get_device()
args = parse_args()

## load model
model, meta = load_pretrained(args.model_dir, f_name=args.model_name)

## load dataset
dataset = SyntheticDataset(path=args.data_dir, config=args.config, example_len=meta['input_shape'][-1], n_examples=args.n_eval, 
                    from_archive=False, meta_filter=['modulation', 'snr', 'fo'])
dataloader = DataLoader(dataset, batch_size=args.batch_size)


logits, labels = get_logits(model, dataloader, device)
ece, mce = get_metrics_torch(logits, labels)
print("ECE:", ece)
print("MCE:", mce)