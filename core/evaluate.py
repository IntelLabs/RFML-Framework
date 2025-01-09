
import torch 
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix

def _get_cm(predictions, truth, normalize=None):
    return confusion_matrix(truth, predictions, normalize=normalize)

def get_cm(model, dataloader, device, normalize=None):
    predictions, truth, _ = predict(model, dataloader, device)
    return _get_cm(truth, predictions, normalize)

def _get_acc(predictions, truth):
    correct = np.sum(np.equal(predictions, truth))
    return correct/len(truth)

def get_acc(model, dataloader, device):
    predictions, truth, _ = predict(model, dataloader, device)
    return _get_acc(predictions, truth)

def _multinomial_aggregate(n, predictions, truth):
    correct = 0
    cm = _get_cm(truth, predictions, "true")
    for i in range(len(predictions)):
        t = truth[i]
        p_m = np.random.multinomial(n, cm[t, :])
        if np.argmax(p_m) == t:
            correct += 1
    return correct / len(truth)

def multinomial_aggregate(n, model, dataloader, device):
    predictions, truth, _ = predict(model, dataloader, device)
    return _multinomial_aggregate(n, predictions, truth)

def _multinomial_enhance(n, predictions, truth, normalize=True):
    cm = _get_cm(truth, predictions, "true")
    cmn = np.zeros((cm.shape[0], cm.shape[1]))
    trues = np.zeros(cm.shape[0])
    for i in range(len(predictions)):
        t = truth[i]
        p_m = np.random.multinomial(n, cm[t, :])
        cmn[t][np.argmax(p_m)] += 1
        trues[t] += 1

    if normalize:
        for i in range(cmn.shape[0]):
            cmn[i] = cmn[i] / trues[i]
    return cmn

def multinomial_enhance(n, model, dataloader, device, normalize=True):
    predictions, truth, _ = predict(model, dataloader, device)
    return _multinomial_enhance(n, predictions, truth, normalize)

def _snrVacc(predictions, truth, snr):
    snr_unique = list(set(snr))
    acc_list = np.zeros(len(snr_unique))
    for i, s_i in enumerate(snr_unique):
        idx = np.argwhere(snr == s_i)
        i_pred = np.take(predictions, idx)
        i_truth = np.take(truth, idx)
        tmp = _get_acc(i_pred, i_truth)
        acc_list[i] = tmp
    return snr_unique, acc_list

def snrVacc(model, dataloader, device):
    predictions, truth, snr = predict(model, dataloader, device)
    return _snrVacc(predictions, truth, snr)

def mitre_score(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    total = 0.
    model.to(device)
    model.eval()
    
    with torch.no_grad(): 
        for idx, batch in enumerate(dataloader):
            inputs = batch["data"].float().to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total += loss.item()
    
    avg = total / (idx + 1.)
    
    m_score = 100. / (1. + avg)
    return m_score

def evaluate(model, dataloader, device, normalize=None):
    predictions, truth, snr = predict(model, dataloader, device)
    return _get_acc(predictions, truth), _get_cm(predictions, truth, normalize), (snr, snrVacc)

def predict(model, dataloader, device):
    model.to(device)
    model.eval()

    predictions = []
    truth = []
    snr = []
    
    with torch.no_grad(): 
        for idx, batch in enumerate(dataloader):
            inputs = batch["data"].float().to(device)
            labels = batch["label"].numpy()
            snrs = batch["snr"].numpy()

            outputs = model(inputs)

            _, curr_pred = torch.max(outputs, 1)
            predictions.extend(curr_pred.cpu().numpy())
            truth.extend(labels)
            snr.extend(snrs)
    
    return predictions, truth, snr

def get_logits(model, dataloader, device):
    model.to(device)
    model.eval()

    logits = []
    labels = []
    
    with torch.no_grad(): 
        for idx, batch in enumerate(dataloader):
            _inputs = batch["data"].float().to(device)
            _labels = batch["label"].to(device)

            outputs = model(_inputs)

            logits.extend(outputs)
            labels.extend(_labels)

    logits = torch.stack(logits).to(device)
    labels = torch.stack(labels).to(device) 
    return logits, labels

def predict_wSoftmax(model, dataloader, num_classes, device):
    softmax = nn.Softmax(dim=1)
    model.to(device)
    model.eval()

    predictions = []
    truth = []
    snr = []
    
    with torch.no_grad(): 
        for idx, batch in enumerate(dataloader):
            inputs = batch["data"].float().to(device)
            labels = batch["label"]

            outputs = softmax(model(inputs))
            labels_oneh = nn.functional.one_hot(labels, num_classes=num_classes)

            # _, curr_pred = torch.max(outputs, 1)
            predictions.extend(outputs.cpu().numpy())
            truth.extend(labels_oneh.numpy())
    
    return np.array(predictions), np.array(truth)
