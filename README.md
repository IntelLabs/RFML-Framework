# RFML Framework

RFML Framework is an open-source library for performing various Radio Frequency Machine Learning (RFML) tasks. 
The repository is intuitively laid out in the following manner:
- `./core/` contains training, validation, and testing functions and general utilities such as the setting of random seeds, argument parsing, and logging std-out to txt files
- `./data/` contains PyTorch Dataset class wrappers for Synthetic and Captured raw RF datasets in SigMF format
- `./models/` contains PyTorch nn.Module wrappers for various model architectures and model I/O functions
- `./scripts/` contains example scripts for model training, evaluation, and calibration

Basic instructions for the use of this framework are given in the subsections below:
- [Setup](#setup)
- [Data](#data)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Evaluation](#evaluation)
- [Calibration](#calibration)

## Setup

This repository has been tested using Python 3.10 and the requires the packages listed in [requirements.txt](./requirements.txt). All required python packages can be installed via `pip` using the following command:

```
pip install -r requirements.txt
```

## Data

This codebase has been designed to ingest raw RF datasets in SigMF format. Compatible synthetic datasets (which make use of the SyntheticDataset class) can be generated using the [Synthetic Radio Frequency Data Generator](https://github.com/IntelLabs/Synthetic-Radio-Frequency-Data-Generator), or [downloaded from IEEE Dataport](https://ieee-dataport.org/open-access/transfer-learning-rf-domain-adaptation-%E2%80%93-synthetic-dataset). 
While there are no publicly available captured datasets that have been used within this framework, an example `CapturedDataset` class can be found in `./data/captured.py` that can be customized to your data. 
Any new dataset classes created for your data should inherit from the `BaseRFDataset` class found in `./data/base.py`, and can be integrated into the framework by adding it to the `initialize_dataset` function found in `./data/__init__.py`.
Additionally, the configuration files in `./configs/` can be used to filter examples from a larger dataset, but are not required.

## Model Architectures

Example CNN, CLDNN, and LSTM model architectures can be found in `./models/`, and example PyTorch layers can be found in `./models/layers/`. 
Any custom model class you design can be integrated into the framework by adding it to the `initialize_model` function found in `./models/__init__.py`. 
Model loading/saving functions can also be found in `./models/__init__.py`.

## Training

An example training script, `./scripts/training/train.py`, can be run using the following command:
```
python scripts/training/train.py --train_path /PATH/TO/TRAIN/DATA/ --val_path /PATH/TO/VALIDATION/DATA/ --save_path /PATH/TO/RESULTS/ --reload_best
```

Additional arguments can be provided at the command line, and can be found in `./core/arguments.py`.

An example model fine-tuning script, `./scripts/training/tune.py`, can be run using the following command:
```
python scripts/training/tune.py --train_path /PATH/TO/TRAIN/DATA/ --val_path /PATH/TO/VALIDATION/DATA/ --results_path /PATH/TO/PRETRAINED/MODEL/ --reload_best
```

## Evaluation

Functions are provided for the calculation of accuracy and the MITRE score, as well as for the creation of confusion matrices and SNR vs accuracy plots, and can be found in `./core/evaluate.py`. 
An example evaluation script calculating test accuracy can be found at `./scripts/evaluation/score.py`, and is run using the following command:
```
python scripts/evaluation/score.py --eval_path /PATH/TO/TEST/DATA/ --results_path /PATH/TO/RESULTS/
```

Additional arguments can be provided at the command line, and can be found in `./core/arguments.py`. 

## Calibration

Finally, this repository contains the functionality to perform post-training model calibration via Temperature Scaling and to calculate expected calibration error (ECE) and maximum calibration error (MCE) in `./core/calibration.py`.
An example calibration script can be found at `./scripts/calibration/calibrate.py`.


