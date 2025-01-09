
import argparse

def get_args(action="train"):

    parser = argparse.ArgumentParser()

    if action == "train":
        parser.add_argument("--train_path", required=True, help="Location of training dataset.")
        parser.add_argument("--val_path", required=True, help="Location of validation dataset.")
        parser.add_argument("--config", type=str, default=None, help="Location of optional dataset config file.")
        parser.add_argument("--save_path", required=True, help="Location to store results.")
        parser.add_argument("--model", type=str, default="cldnn", help="Model name.")
        parser.add_argument("--input_length", default=512, type=int, 
                            help="Number of input samples per training example (default: %(default)s).")
        parser.add_argument("--correct_cfo", default=True, action="store_true", 
                            help="Correct center frequency offset. Only valid for captured dataset." )
        parser.add_argument("--label", default="mod_scheme", type=str, help="Label type: modscheme for AMC, tx_id for SEI")
        parser.add_argument("--n_train", default=-1, type=int, help="Training dataset size.")
        parser.add_argument("--n_val", default=-1, type=int, help="Validation dataset size.")
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: %(default)s).")
        parser.add_argument("--wd", type=float, default=0.0, help="Weight decay (default: %(default)s).")
        parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs (default: %(default)s).")
        parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: %(default)s).")
        parser.add_argument("--no_gpu", action="store_true", default=False, help="Run on CPU only.")
        parser.add_argument("--reload_best", action="store_true", default=True, 
                            help="Reload the model with lowest validation accuracy at the end of training.")
        parser.add_argument("--seed", type=int, default=0, help="Set random seed for reproducibility.")
        parser.add_argument("--min_snr", type=int, default=None, help="Lower bound of AWGN to add")
        parser.add_argument("--max_snr", type=int, default=None, help="Upper bound of AWGN to add")
        parser.add_argument("--cuda", type=int, default=-1, help="Manually specify GPU to train on")
        parser.add_argument("--max_classes", type=int, default=60, help="Maximum number of classes")
    elif action == "tune":
        parser.add_argument("--train_path", required=True, help="Location of training dataset.")
        parser.add_argument("--val_path", required=True, help="Location of validation dataset.")
        parser.add_argument("--config", type=str, default=None, help="Location of optional dataset config file.")
        parser.add_argument("--results_path", required=True, help="Location of results.")
        parser.add_argument("--save_name", required=True)
        parser.add_argument("--n_train", default=-1, type=int, help="Training dataset size.")
        parser.add_argument("--n_val", default=-1, type=int, help="Validation dataset size.")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: %(default)s).")
        parser.add_argument("--wd", type=float, default=0.0, help="Weight decay (default: %(default)s).")
        parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs (default: %(default)s).")
        parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: %(default)s).")
        parser.add_argument("--no_gpu", action="store_true", default=False, help="Run on CPU only.")
        parser.add_argument("--seed", type=int, default=-1, help="Set random seed for reproducibility.")
        parser.add_argument("--reload_best", action="store_true", default=False, 
                            help="Reload the model with lowest validation accuracy at the end of training.")
    elif action == "eval":
        parser.add_argument("--eval_path", required=True, help="Location of evaluation dataset.")
        parser.add_argument("--config", type=str, default=None, help="Location of optional dataset config file.")
        parser.add_argument("--n_eval", default=-1, type=int, help="Evaluation dataset size.")
        parser.add_argument("--results_path", required=True, help="Location of results.")
        parser.add_argument("--model_name", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: %(default)s).")
        parser.add_argument("--no_gpu", action="store_true", default=False, help="Run on CPU only.")
    else:
        print("Invalid action.")
        raise

    return parser.parse_args()
    
def get_sweep_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", required=True, help="Location of training dataset.")
    parser.add_argument("--val_path", required=True, help="Location of validation dataset.")
    parser.add_argument("--save_path", required=True, help="Location to store results.")
    parser.add_argument("--config", type=str, default=None, help="Location of optional dataset config file.")
    parser.add_argument("--n_train", default=-1, type=int, help="Training dataset size.")
    parser.add_argument("--n_val", default=-1, type=int, help="Validation dataset size.")
    parser.add_argument("--r", default=128, type=int)
    parser.add_argument("--d", default=1, type=float)
    parser.add_argument("--w", default=1, type=float)
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--dilation", default=1, type=int)
    parser.add_argument("--sgd", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: %(default)s).")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay (default: %(default)s).")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs (default: %(default)s).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: %(default)s).")
    parser.add_argument("--no_gpu", action="store_true", default=False, help="Run on CPU only.")
    parser.add_argument("--reload_best", action="store_true", default=False, 
                        help="Reload the model with lowest validation accuracy at the end of training.")
    parser.add_argument("--seed", type=int, default=-1, help="Set random seed for reproducibility.")

    return parser.parse_args()
