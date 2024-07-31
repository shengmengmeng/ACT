# -*- coding: utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   @File        : eval.py.py
#   @Author      : Zeren Sun
#   @Created date: 2022/11/18 10:28
#   @Description :
#
# ================================================================
import torch
from tqdm import tqdm
from utils.utils import AverageMeter
# from torchmetrics import Precision, Recall, F1Score
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC


def accuracy(y_pred, y_actual, topk=(1, ), return_tensor=False):
    """
    Computes the precision@k for the specified values of k in this mini-batch
    :param y_pred   : tensor, shape -> (batch_size, n_classes)
    :param y_actual : tensor, shape -> (batch_size)
    :param topk     : tuple
    :param return_tensor : bool, whether to return a tensor or a scalar
    :return:
        list, each element is a tensor with shape torch.Size([])
    """
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res


def evaluate(dataloader, model, dev, topk=(1,), progress_bar=False):
    """

    :param dataloader:
    :param model:
    :param dev: devices, gpu or cpu
    :param topk: [tuple]          output the top topk accuracy
    :param progress_bar: [bool]   whether or not show progressbar
    :return:     [list[float]]    topk accuracy
    """
    model.eval()
    test_accuracy = AverageMeter()
    test_accuracy.reset()
    topk_accuracy = AverageMeter()
    topk_accuracy.reset()

    with torch.no_grad():
        pbar = tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc=f'EVALUATING') if progress_bar else dataloader
        for _, sample in enumerate(pbar):
            x = sample['data'].to(dev)
            y = sample['label'].to(dev)
            output = model(x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            acc = accuracy(logits, y, topk)
            test_accuracy.update(acc[0], x.size(0))
            if len(topk) > 1:
                topk_accuracy.update(acc[1], x.size(0))
    if len(topk) == 1:
        return test_accuracy.avg
    elif len(topk) == 2:
        return test_accuracy.avg, topk_accuracy.avg
    else:
        raise AssertionError(f'topk is set incorrectly (current topk is {topk})')


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
