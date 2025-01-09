import os
import sys
sys.path.append(os.path.abspath("."))

import json
import math
import cmath

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import BaseRFDataset

class CapturedDataset(BaseRFDataset):
    def __init__(self, path, config=None, example_len=1024, n_examples=None, label="mod_scheme",
                meta_filter=None, correct_cfo=True, max_classes=60, mintx=None, maxtx=None):
        self.example_len = example_len
        if config is None:
            self.config = None
        else:
            with open(config) as f:
                self.config = json.load(f)
        self.correct_cfo = correct_cfo

        self.df = pd.read_csv(path, index_col=0)
        self.df = self.df.astype({"cf":int, "capture_start":int})
        _n_classes = len(self.df[label].unique())
        if (label == "tx_id") and (_n_classes > max_classes):
            curr_classes: np.ndarray = self.df[label].unique()
            curr_classes.sort()
            curr_classes = curr_classes[:max_classes]
            self.df = self.df[self.df["tx_id"].isin(curr_classes)]
            _n_classes = max_classes
        if mintx is not None:
            self.df = self.df[self.df["tx_id"] >= mintx]
            _n_classes = len(self.df[label].unique())
        if maxtx is not None:
            self.df = self.df[self.df["tx_id"] <= maxtx]
            _n_classes = len(self.df[label].unique())

        if (n_examples is not None) and (n_examples > 0) and (n_examples < len(self.df)):
            all_classes_present = False
            while not all_classes_present:
                self.df = self.df.sample(frac=1).reset_index(drop=True)
                if len(self.df.loc[:n_examples,label].squeeze().unique()) == _n_classes:
                    all_classes_present = True
                    self.df.drop(np.arange(n_examples, len(self.df)), inplace=True)
        elif (n_examples is not None) and (n_examples > len(self.df)):
            if maxtx is None and mintx is None:
                raise
        self.n_examples = len(self.df)

        if label in list(self.df.columns):
            self.label_col = label
            self._create_labels(self.label_col)


    def __getitem__(self, idx, sample_rate=250000):
        _df_row = self.df.iloc[idx]
        capture_start = _df_row["capture_start"]
        with open(_df_row["filename"] + ".sigmf-data", "rb") as _f:
            f_data = np.load(_f)
        iq = f_data[capture_start : (self.example_len + capture_start)]
        if self.correct_cfo:
            cfo = _df_row["cfo"]
            corr = [cmath.exp(-1j*2*math.pi*cfo*(t/sample_rate)) for t in range(len(iq))]
            iq = np.multiply(iq, corr)
        i = torch.FloatTensor(iq.real)
        q = torch.FloatTensor(iq.imag)
        iq = np.stack((i,q))
        snr = _df_row["snr"]
        label = _df_row['label_int']
        return {'data':iq[np.newaxis], 'label':label, 'snr':snr}

    def _filter_meta(self, inventory, config, params=["modulation", "snr", "cf"]):
        if "modulation" in params and len(config["modulation"]) != 0:
            inventory = inventory[inventory["modulation"].isin( config["modulation"])]

        if "snr" in params and len(config["snr"]) != 0:
            inventory = inventory[inventory["snr"] >= config["snr"][0]]
            inventory = inventory[inventory["snr"] <= config["snr"][1]]

        if "cf" in params and len(config["cf"]) != 0:
            inventory = inventory[inventory["cf"] >= (config["cf"][0] - 1e6)]
            inventory = inventory[inventory["cf"] <= (config["cf"][0] + 1e6)]

        if "tx_id" in params and len(config["tx_id"]) != 0:
            inventory = inventory[inventory["tx_id"].isin(config["tx_id"])]

        if "rx_id" in params and len(config["rx_id"]) != 0:
            inventory = inventory[inventory["rx_id"].isin(config["rx_id"])]

        if "rx_loc" in params and len(config["rx_loc"]) != 0:
            inventory = inventory[inventory["rx_loc"].isin(config["rx_loc"])]

        return inventory
        
        