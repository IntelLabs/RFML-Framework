## imports
import os
import tarfile
import json

import numpy as np
import pandas as pd

import torch
from data import BaseRFDataset

mapper = {"bpsk":"bpsk", "qpsk":"qpsk", "8psk":"8psk", "16psk":"16psk",
          "8qam":"8qam", "16qam":"16qam", "32qam":"32qam", "64qam":"64qam",
          "fsk5k":"fsk", "fsk75k":"fsk", "gfsk5k":"fsk", "gfsk75k":"fsk",
          "fmnb":"fm", "fmwb":"fm", "dsb":"am", "dsbsc":"am", "lsb":"am",
          "usb":"am", "awgn":"awgn"}

class SyntheticDataset(BaseRFDataset):
    def __init__(self, path, config=None, example_len=1024, n_examples=None, label="mod_scheme", 
                from_archive=True, meta_filter=['modulation', 'snr']):
        ''' create pandas df from sigmf files '''

        assert not ('snr' in meta_filter and 'esno' in meta_filter)

        self.example_len = example_len
        if config is None:
            self.config = None
        else:
            with open(config) as f:
                self.config = json.load(f)

        if from_archive:
            ## untar
            path_dir = path.split('.')[0]
            with tar.open(path, mode="r") as t:
                t.extractall(path_dir)
        else:
            path_dir = path
        
        if os.path.isfile(os.path.join(path_dir, 'files.txt')):
            with open(os.path.join(path_dir, 'files.txt')) as f: 
                files = [line.rstrip() for line in f]
        else:
            files = os.listdir(path)    # both data and meta files
        files = [os.path.join(path_dir, f.split(".")[0]) for f in files] # add path
        files = list(set(files))    # unique filenames only
        if n_examples is None:
            self.n_examples = len(files)
        elif n_examples == -1:
            self.n_examples = len(files)
        else:
            self.n_examples = n_examples
        np.random.shuffle(files)

        self.df = pd.DataFrame(columns=["capture_start", "capture_len", "snr", "fo", "channel_type", "mod_scheme", "filename"], index=np.arange(self.n_examples))
        idx = 0
        for i in range(len(files)):
            if idx == self.n_examples:
                break

            f = files[i]

            if f != path_dir:
                ## get meta
                with open(f + ".sigmf-meta") as _f:
                    f_meta = json.load(_f) 
                    f_meta = f_meta["annotations"][0]

                if self._filter_meta(f_meta, self.config, params=meta_filter):
                    self.df.loc[idx, "capture_start"] = f_meta["core:sample_start"]
                    self.df.loc[idx, "capture_len"] = f_meta["core:sample_count"]
                    self.df.loc[idx, "snr"] = f_meta["channel"]["snr"]
                    self.df.loc[idx, "fo"] = f_meta["channel"]["fo"]
                    self.df.loc[idx, "channel_type"] = f_meta["channel"]["type"]
                    self.df.loc[idx, "mod_scheme"] = f_meta["rfml_labels"]["modclass"]
                    self.df.loc[idx, "filename"] = files[i]
                    idx += 1
        
        self.df.dropna(inplace=True)
        self.n_examples = self.df.shape[0]
        
        if label in list(self.df.columns):
            self.label_col = label
            self._create_labels(self.label_col)
        else:
            print("Unknown label name.")
            raise
    
    def __getitem__(self, idx):
        _df_row = self.df.iloc[idx]

        capture_start = _df_row["capture_start"]

        with open(_df_row["filename"] + ".sigmf-data", "rb") as _f:
            f_data = np.load(_f)
        i = torch.FloatTensor(f_data[capture_start * 2 : (self.example_len + capture_start) * 2 : 2])
        q = torch.FloatTensor(f_data[(capture_start * 2) + 1 : ((self.example_len + capture_start) * 2) + 1 : 2])
        iq = np.stack((i, q))
        label = _df_row['label_int']
        snr = _df_row['snr']
        
        return {'data':iq[np.newaxis], 'label':label, 'snr':snr}

    def _filter_meta(self, meta, config, params=["modulation", "snr"]):
        if config is None:
            return True
        
        if ("modulation" in params) and (meta["rfml_labels"]["modclass"] not in config["modulation"]):
            return False
        
        if ("snr" in params):
            if (meta["channel"]["snr"] < config["channel"]["snr"][0]):
                return False
            elif (meta["channel"]["snr"] > config["channel"]["snr"][1]):
                return False
        
        if ("esno" in params):
            lin_freq = ["bpsk", "qpsk", "8psk", "16psk", "8qam", "16qam", "32qam", "64qam","fsk5k", "fsk75k", "gfsk5k", "gfsk75k"]
            if meta["rfml_labels"]["modclass"] in lin_freq:
                esno = meta["channel"]["snr"]*np.log2(meta["filter"]["sps"])
            else:
                esno = meta["channel"]["snr"]
            if esno < config["channel"]["esno"][0]:
                return False
            elif esno > config["channel"]["esno"][1]:
                return False

        if ("fo" in params):
            if (meta["channel"]["fo"] < (2.*np.pi*config["channel"]["fo"][0])):
                return False
            elif (meta["channel"]["fo"] > (2.*np.pi*config["channel"]["fo"][1])):
                return False
        
        return True
