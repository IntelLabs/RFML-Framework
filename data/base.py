import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class BaseRFDataset(Dataset):
    def __init__(self, path, config=None, example_len=1024, n_examples=None, label="mod_scheme", 
                from_archive=True, meta_filter=None):
        ''' create pandas df from sigmf files '''
        super(BaseRFDataset, self).__init__()

    def shape(self):
        return (1, 2, self.example_len)

    def clip(self, new_length=128):
        '''Take a dataset with example length = n, and clip to example length < n.
        Useful for testing networks with different input sizes without reloading datasets'''
        self.example_len = new_length

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        raise NotImplementedError

    def _create_labels(self, label_col, label_map=None):
        if label_map is None:
            labels = self.df[label_col].unique()
            labels.sort()
            self.num_classes = len(labels)
            if label_col == "tx_id":
                label_map_int = {int(labels[i]):int(i) for i in range(len(labels))}
            else:
                label_map_int = {labels[i]:int(i) for i in range(len(labels))}
            self.label_map = label_map_int
        else:
            self.label_map = label_map
            self.num_classes = len(self.label_map)

        labels_int = []
        for i in range(self.__len__()):
            labels_int.append(self.label_map[self.df.iloc[i][label_col]])
        self.df['label_int'] = labels_int

    def _filter_meta(self, meta, config, params=["modulation", "snr"]):
        raise NotImplementedError