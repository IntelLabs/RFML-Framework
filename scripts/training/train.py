
'''
Usage:
python scripts/train_cnn.py --train_path /store/nosnap/data/train/ --val_path /store/nosnap/data/validate/ --save_path /store/nosnap/results/modclass/temp/ --reload_best
'''
import logging
import os
import sys
sys.path.append(os.path.abspath("."))

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.utils import set_random_seed, get_device
from models import initialize_model
from data import initialize_dataset


from core.train import train
from models import save
from core.utils import create_paths
from core.logging import init_logger
from core.arguments import get_args

args = get_args(action="train")

## set random seed
set_random_seed(args.seed)

## Create file paths
args.save_path = create_paths(args)
args.min_snr = None
args.max_snr = None

## Create Loggers
init_logger(args.save_path, [args.model])
tb_writer = SummaryWriter(args.save_path, flush_secs=60 * 1)  # flush every 1 minutes
logger = logging.getLogger(args.model)

## get device
device = get_device(args)
logger.info("Running on " + device)

## Load datasets
train_set = initialize_dataset(args, type="SyntheticDataset", split='train')
msg = "Train set loaded containing {} examples of length {} samples, and {} output classes."
logger.info(msg.format(len(train_set), args.input_length, train_set.num_classes))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

val_set = initialize_dataset(args, type="SyntheticDataset", split='val') 
msg = "Validation set loaded containing {} examples of length {} samples, and {} output classes."
logger.info(msg.format(len(val_set), args.input_length, val_set.num_classes)) 
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

## Initialize model, optimizer, criterion
logger.info("Initializing model")
model, optimizer, criterion = initialize_model(args, train_set)

## train model 
logger.info("Training model")
cp_name = os.path.join(args.save_path, args.model + "_checkpoint.pt")
model, best_epoch = train(model, train_loader, val_loader, args.n_epochs, 
                        optimizer, criterion, logger, tb_writer, cp_name, 
                        device, args.reload_best, use_scheduler=True)

## save final model and associated metadata
meta = {"model_type":args.model,
        "input_shape":train_set.shape(),
        "num_classes":train_set.num_classes,
        "label_map":train_set.label_map,
        "best_epoch":best_epoch}
meta.update(vars(args))

save(model, meta, args.save_path, args.model)

## clean up
tb_writer.flush()
tb_writer.close()
