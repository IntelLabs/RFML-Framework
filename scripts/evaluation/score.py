## DEPRECATED

'''
Get accuracy, MITRE score, and/or accVsnr of trained model.

Usage:
python scripts/score.py --eval_path /store/nosnap/data/validate/ --results_path /store/nosnap/results/modclass_test/

'''

import argparse
import os
import sys
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.synthetic import SigDataset

from models import load_pretrained
from core.evaluate import get_acc
from core.evaluate import evaluate, snrVacc, mitre_score
from core.arguments import get_args
from core.utils import get_device, set_random_seed

args = get_args("eval")

device = get_device(args)

## load model, meta
model, meta = load_pretrained(args.results_path, f_name=args.model_name)

if args.n_eval == -1:
    args.n_eval = None

## load dataset
if args.eval_path[-6:] == ".sigmf":
    dataset = SigDataset(path=args.eval_path, config=args.config, example_len=meta['input_length'], n_examples=args.n_eval, from_archive=True)
else:
    dataset = SigDataset(path=args.eval_path, config=args.config, example_len=meta['input_length'], n_examples=args.n_eval, from_archive=False)
dataloader = DataLoader(dataset, batch_size=args.batch_size)
print("Label Mapping: " + str(dataset.label_map))

## get results
acc = get_acc(model, dataloader, device)

print("Model Accuracy: " + str(acc))
