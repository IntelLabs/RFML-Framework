import os
import sys
import pickle
sys.path.append(os.path.abspath("."))

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from data.captured import CapturedDataset
from core.evaluate import get_acc, multinomial_aggregate, get_cm, multinomial_enhance
from models import load_pretrained
from core.utils import get_device
from argparse import ArgumentParser
import json


def generate_cm(model_dir, data_inv, save_name, label, ex_len):
    model_file = os.path.join(model_dir, 'cldnn.pt')
    assert os.path.exists(model_file), f'Nothing at {model_file}'

    test_set = CapturedDataset(path=data_inv, example_len=ex_len, label=label, correct_cfo=True, n_examples=6000)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    device = get_device()

    assert os.path.exists(os.path.join(model_dir, 'cldnn.pt'))
    model, meta = load_pretrained(model_dir, 'cldnn')
    model.awgn.set_snr(snr_min=None, snr_max=None)

    cm = get_cm(model, test_loader, device, normalize='true')
    acc = get_acc(model, test_loader, device)
    cmm10 = multinomial_enhance(10, model, test_loader, device)
    cmm100 = multinomial_enhance(100, model, test_loader, device)

    print(cm)
    top5 = dict()
    for i in range(cm.shape[0]):
        top5[i]: list = [cm[i][i]] + list(reversed(np.sort(np.delete(cm[i], i)).tolist()))[:min(cm.shape[1], 5)]


    res = dict()
    res['model_file'] = model_file
    res['n_classes'] = test_set.num_classes
    res['accuracy'] = acc
    res['encoding'] = 'latin-1'
    res['conf_mat'] = pickle.dumps(cm).decode(res['encoding'])
    res['mna10'] = multinomial_aggregate(10, model, test_loader, device)
    res['mna100'] = multinomial_aggregate(100, model, test_loader, device)
    res['conf_mat_multinom_10'] = pickle.dumps(cmm10).decode(res['encoding'])
    res['conf_mat_multinom_100'] = pickle.dumps(cmm100).decode(res['encoding'])
    res['labels'] = test_set.label_map
    res['top_5'] = top5
    with open(f'{save_name}.json', 'w') as file:
        json.dump(res, file)

def parse_args():
    p = ArgumentParser()
    p.add_argument('--model-dir', required=True, help='Directory containing model')
    p.add_argument('--data-inv', required=True, help='Path to dataset inventory CSV')
    p.add_argument('--out', required=True, help='Save path of the confusion matrix')
    p.add_argument('--label', default='mod_scheme', choices=['mod_scheme', 'tx_id'], help='Class label (def: %(default)s)')
    p.add_argument('--example-len', default=256, help='Example length (def: %(default)s)')
    
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    assert os.path.exists(args.model_dir), 'Model directory does not exist'
    assert os.path.exists(args.data_inv), 'Dataset inventory does not exist'
    generate_cm(args.model_dir, args.data_inv, args.out, args.label, args.example_len)

