import torch
import numpy as np
from tqdm import tqdm

from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count+1e-8)


# evaluate
def accuracy(y_pred, y_actual, topk=(1, ), return_tensor=False):
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res


def evaluate_cls_acc(dataloader, model, dev, topk=(1,)):
    model[0].eval()
    model[1].eval()
    test_loss = AverageMeter()
    test_loss.reset()
    test_accuracy = AverageMeter()
    test_accuracy.reset()

    test_loss1 = AverageMeter()
    test_loss1.reset()
    test_accuracy1 = AverageMeter()
    test_accuracy1.reset()

    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='evaluating')):
            if type(sample) is dict:
                x = sample['data'].to(dev)
                y = sample['label'].to(dev)
            else:
                # x, y, _ = sample
                x, y,_,_ = sample
                x, y = x.to(dev), y.to(dev)
            output = model[0](x)
            logits = output['logits'] if type(output) is dict else output
            loss = torch.nn.functional.cross_entropy(logits, y)
            test_loss.update(loss.item(), x.size(0))
            acc = accuracy(logits, y, topk)
            test_accuracy.update(acc[0], x.size(0))

            output = model[1](x)
            logits = output['logits'] if type(output) is dict else output
            loss = torch.nn.functional.cross_entropy(logits, y)
            test_loss1.update(loss.item(), x.size(0))
            acc = accuracy(logits, y, topk)
            test_accuracy1.update(acc[0], x.size(0))

    return {'accuracy': [test_accuracy.avg, test_accuracy1.avg], 'loss': [test_loss.avg, test_loss1.avg]}


def evaluate_relabel_pr(given_labels, corrected_labels):
    precision = 0.0
    recall = 0.0
    # TODO: code for evaluation of relabeling (precision, recall)
    return {'relabel-precision': precision, 'relabel-recall': recall}

def detection_evaluate(prediction, ground_truth):
    # prediction and ground_truth are both indicator vectors containing 0 / 1
    precision_func = BinaryPrecision()
    precision = precision_func(prediction, ground_truth)
    recall_func = BinaryRecall()
    recall = recall_func(prediction, ground_truth)
    f1_func = BinaryF1Score()
    f1_score = f1_func(prediction, ground_truth)
    auroc_func = BinaryAUROC()
    auroc = auroc_func(prediction, ground_truth)
    return precision.item(), recall.item(), f1_score.item(), auroc.item()