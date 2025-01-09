
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from models import load_pretrained

## SOURCES: 
## https://towardsdatascience.com/neural-network-calibration-using-pytorch-c44b7221a61
## https://colab.research.google.com/drive/1H_XlTbNvjxlAXMW5NuBDWhxF3F2Osg1F?usp=sharing
## https://github.com/gpleiss/temperature_scaling


class CalibratedModel(nn.Module):
    def __init__(self, model_path, temperature=None, f_name="cldnn", device='cuda'):
        super().__init__()
        self.model, self.meta = load_pretrained(model_path, f_name=f_name)
        print(self.model)
        if temperature is None:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.temperature = nn.Parameter(torch.ones(1)*temperature)
        self.model.to(device)
        self.num_classes = self.model.num_classes
        self.device = device
        self.model_chkpt = os.path.join(model_path, f_name)
        self.f_name = f_name

    def forward(self, X):
        logits = self.model(X)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        return torch.div(logits, self.temperature.to(self.device))

    def get_set_temperature(self, val_loader, lr=0.01, max_iter=1000, line_search_fn=None):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter, 
                                line_search_fn=line_search_fn)

        ## get logits and labels
        logits = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                x = batch["data"].float().to(self.device)
                y = batch["label"].to(self.device)
                
                logits.append(self.model(x))
                labels.append(y)

            logits = torch.cat(logits).to(self.device)
            labels = torch.cat(labels).to(self.device)

        ## get ECE/MCE before temperature scaling
        # softmax = nn.Softmax(dim=1)
        # preds = softmax(logits)
        # labels_oh = nn.functional.one_hot(labels, num_classes=self.num_classes)
        before_ece, before_mce = get_metrics_torch(logits, labels)
        print("Metrics before temperature: ", before_ece.item(), before_mce)

        def eval():
            optimizer.zero_grad()
            loss = criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        print("Temperature set to ", self.temperature.item())
        ## TODO: add temerature to meta

        ## get ECE/MCE after temperature scaling
        scaled_logits = self.temperature_scale(logits)
        # scaled_preds = softmax(scaled_logits)
        after_ece, after_mce = get_metrics_torch(scaled_logits, labels)
        print("Metrics after temperature: ", after_ece.item(), after_mce)
        print()

        ## TODO: save calibrated model
        # save(self.model, self.meta, self.model_chkpt, self.f_name + "_calibrated")


def get_metrics_torch(logits, labels, num_bins=10):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ## TODO: calculate MCE too
    ece = torch.zeros(1, device=logits.device)
    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            abs_conf_dif = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece +=  prop_in_bin * abs_conf_dif
            mce = max(mce, abs_conf_dif.item())

    return ece, mce


## DEPRECATED: USE TORCH VERSION
def calc_bins(preds, labels_oneh):
	# Assign each prediction to a bin
    # print(preds)
    # print(labels_oneh)
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    ## binned version of max output
    binned_max = np.digitize(np.max(preds, axis=1), bins)
    # print(binned_max)
    ## confidence of max output
    conf_max = np.max(preds, axis=1)
    # print(conf_max)
    ## index of predicted 
    pred_idx = np.argmax(preds, axis=1)
    # print(pred_idx)
    ## truth index
    label_idx = np.argmax(labels_oneh, axis=1)
    # print(label_idx)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        ## which max output values are in bin
        in_bin = [i == bin for i in binned_max]
        bin_sizes[bin] = np.sum(in_bin)
        if bin_sizes[bin] > 0:
            ## which max output values in bin are accurate
            bin_accs[bin] = np.array([(pred_idx[i] == label_idx[i]) and in_bin[i] for i in range(len(in_bin))]).sum() / bin_sizes[bin]
            ## how confident is the model output for the bin
            bin_confs[bin] = np.array([conf_max[i]*in_bin[i] for i in range(len(in_bin))]).sum() / bin_sizes[bin]

    return bins, bin_accs, bin_confs, bin_sizes

## DEPRECATED: USE TORCH VERSION
def get_metrics(preds, labels_oneh):
    ECE = 0
    MCE = 0
    bins, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)
    # print(bin_accs)
    # print(bin_confs)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE
