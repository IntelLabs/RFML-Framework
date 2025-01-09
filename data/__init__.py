from .base import BaseRFDataset
from .synthetic import SyntheticDataset
from .captured import CapturedDataset

def initialize_dataset(args, type='SyntheticDataset', split="train", meta_filter=['modulation', 'snr', 'fo']):
    if split == "train":
        path = args.train_path
        n_examples = args.n_train
    elif split == "val":
        path = args.val_path
        n_examples = args.n_val
    elif split == "test":
        path = args.test_path
        n_examples = args.n_test

    if path[-6:] == ".sigmf":
        from_archive = True
    else:
        from_archive = False


    if type == "SyntheticDataset":
        dataset = SyntheticDataset(path=path, config=args.config, example_len=args.input_length, n_examples=n_examples, 
                from_archive=from_archive, meta_filter=meta_filter)
    elif type == "CapturedDataset":
        dataset = CapturedDataset(path=path, config=args.config, example_len=args.input_length, n_examples=n_examples, 
                meta_filter=meta_filter, correct_cfo=args.correct_cfo, label=args.label, max_classes=args.max_classes)
    else:
        raise ValueError
    return dataset
