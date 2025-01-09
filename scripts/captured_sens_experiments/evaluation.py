import sys
import os
sys.path.append(os.path.abspath("."))
from core.evaluate import predict, _get_acc, _get_cm, _multinomial_enhance, _multinomial_aggregate
from data.captured import CapturedDataset
from torch.utils.data import DataLoader
from models import load_pretrained
from core.utils import get_device
import numpy as np
import argparse
import logging
import pickle
import json

def get_f1_score(confusion_matrix: np.ndarray, i):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for j in range(confusion_matrix.shape[1]):
        if (i == j):
            TP += confusion_matrix[i, j]
            tmp = np.delete(confusion_matrix, i, 0)
            tmp = np.delete(tmp, j, 1)

            TN += np.sum(tmp)
        else:
            if (confusion_matrix[i, j] != 0):

                FN += confusion_matrix[i, j]
            if (confusion_matrix[j, i] != 0):

                FP += confusion_matrix[j, i]
    if TP == 0:
        recall = 0
        precision = 0
    else:
        recall = TP / (FN + TP)
        precision = TP / (TP + FP)
    if recall == 0 or precision == 0:
        return 0
    f1_score = 2 * 1/(1/recall + 1/precision)

    return f1_score

def get_macro_f1_score(confusion_matrix: np.ndarray):
    sum = 0
    for i in range(confusion_matrix.shape[0]):
        sum += get_f1_score(confusion_matrix, i)
    return sum / confusion_matrix.shape[0]

class Evaluator():
    def __init__(self, args) -> None:
        self.logger = logging.getLogger('evaluator')
        self.logger.info('Evaluator started')
        try:
            self.models = args.models
            self.eval = args.eval
            self.data = args.data
        except Exception as e:
            self.logger.error(e)

    def run(self):
        modeldirs = os.listdir(self.models)
        self.logger.debug(f'{len(modeldirs)} model directories')
        for modeldir in modeldirs:
            self.logger.info(f'Checking model {modeldir}')
            _modeldir = os.path.join(self.models, modeldir)
            modelfiles = os.listdir(_modeldir)

            # Check for necessary files
            if 'cldnn.pt' not in modelfiles:
                self.logger.warning(f'No model found in "{_modeldir}". Skipping.')
                continue
            if 'eval.json' not in modelfiles:
                self.logger.warning(f'No evaluation plan in "{_modeldir}". Skipping.')
                continue
            
            # Create directories
            _evaldir = os.path.join(self.eval, modeldir)
            if not os.path.exists(_evaldir):
                self.logger.info(f'Creating directory "{_evaldir}"')
                os.makedirs(_evaldir)

            resultpath = os.path.join(_evaldir, 'results.json')
            if os.path.exists(resultpath):
                self.logger.info(f'   Results already present, skipping model {modeldir}')
                continue

            
            evalpath = os.path.join(_modeldir, 'eval.json')
            try:
                with open(evalpath, 'r') as evalfile:
                    evalplan = json.load(evalfile)
            except Exception as e:
                self.logger.error(evalpath)
                self.logger.error(e)
                continue

            p_data_inv = None
            
            # Perform every test
            results = dict()
            results['tests'] = list()
            defaults: dict = evalplan['defaults']
            tests = evalplan['tests']
            for test in tests:
                # Initialize testing defaults
                for k, v in defaults.items():
                    if k not in test:
                        test[k] = v
                self.logger.info(f'   Working on test "{test["name"]}".')

                mintx = int(test['mintx']) if 'mintx' in test else None
                maxtx = int(test['maxtx']) if 'maxtx' in test else None
                    
                # Load dataset if not cached
                data_inv = os.path.join(self.data, test['dataset'])
                if not p_data_inv == data_inv or mintx is not None or maxtx is not None:
                    self.logger.info(f'   Loading new dataset "{data_inv}"')
                    test_set = CapturedDataset(path=data_inv, example_len=test['example_len'], label=test['label'], correct_cfo=True, n_examples=test['n_examples'], mintx=mintx, maxtx=maxtx)
                    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
                else:
                    self.logger.info(f'   Reusing cached dataset "{data_inv}"')

                
                # Load model with its specifications
                device = get_device()
                model_file = os.path.join(_modeldir, 'cldnn.pt')
                model, _ = load_pretrained(_modeldir, 'cldnn')
                if test['snr'] is None:
                    model.awgn.set_snr(snr_min=None, snr_max=None)
                else:
                    model.awgn.set_snr(snr_min=test['snr'], snr_max=test['snr']+1)
                
                self.logger.info(f'   Running test')
                predictions, truth, _ = predict(model, test_loader, device)

                self.logger.info(f'   Got results, performing analysis')
                acc = _get_acc(predictions, truth)
                cm = _get_cm(predictions, truth, normalize='true')
                f1 = get_macro_f1_score(cm)
                mna10 = _multinomial_aggregate(10, predictions, truth)
                cmm10 = _multinomial_enhance(10, predictions, truth)
                mnf110 = get_macro_f1_score(cmm10)
                mna100 = _multinomial_aggregate(100, predictions, truth)
                cmm100 = _multinomial_enhance(100, predictions, truth)
                mnf1100 = get_macro_f1_score(cmm100)

                top5 = dict()
                for i in range(cm.shape[0]):
                    top5[i]: list = [cm[i][i]] + list(reversed(np.sort(np.delete(cm[i], i)).tolist()))[:min(cm.shape[1], 5)]

                result = dict()
                result['name'] = test['name']
                result['model_file'] = model_file
                result['dataset'] = test['dataset']
                result['label'] = test['label']
                result['label_map'] = test_set.label_map
                result['n_classes'] = test_set.num_classes
                result['accuracy'] = acc
                result['f1'] = f1
                result['encoding'] = 'latin-1'
                result['conf_mat'] = pickle.dumps(cm).decode(result['encoding'])
                result['accuracy_mn10'] = mna10
                result['accuracy_mn100'] = mna100
                result['f1_mn10'] = mnf110
                result['f1_mn100'] = mnf1100
                result['conf_mat_multinom_10'] = pickle.dumps(cmm10).decode(result['encoding'])
                result['conf_mat_multinom_100'] = pickle.dumps(cmm100).decode(result['encoding'])
                result['labels'] = test_set.label_map
                result['top_5'] = top5
                results['tests'].append(result)
            
            with open(resultpath, 'w') as resultfile:
                json.dump(results, resultfile)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models', required=True, help='Top level directory containing directories for models')
    p.add_argument('--eval', required=True, help='Top level directory to store evaluation results')
    p.add_argument('--data', required=True, help='Top level directory containing all datasets')
    return p.parse_args()

def main(args):
    logging.basicConfig(level=logging.INFO)
    evaluator = Evaluator(args)
    evaluator.run()
    
    return 0

if __name__ == '__main__':
    sys.exit(main(parse_args()))