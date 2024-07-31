import os
import sys
import argparse
import math
import time
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from loss import *
from utils.builder import *
from utils.plotter import *
from model.MLPHeader import MLPHead
from util import *
from utils.eval import *
from model.ResNet32 import resnet32
from model.SevenCNN import CNN
from data.Clothing1M import *
from utils.ema import EMA
import matplotlib
from torch.cuda.amp import autocast as autocast
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import datetime
from utils.logger import Logger


def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def build_logger(params):
    if params.ablation:
        logger_root = f'Ablation/{params.dataset}'
    else:
        logger_root = f'Results/{params.dataset}'
    logger_root = str(params.model) + logger_root
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    percentile = int(params.closeset_ratio * 100)
    noise_condition = f'symm_{percentile:2d}' if params.noise_type == 'symmetric' else f'asym_{percentile:2d}'
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if params.ablation:
        result_dir = os.path.join(logger_root, noise_condition, f'{params.log}-{logtime}')
    else:
        result_dir = os.path.join(logger_root, noise_condition, f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    # save_config(params, f'{result_dir}/params.cfg')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir


class ResNet(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True, activation='tanh', classifier='linear'):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.classfier_head = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.classfier_head, init_method='He')
        elif classifier.startswith('mlp'):
            sf = float(classifier.split('-')[1])
            self.classfier_head = MLPHead(self.feat_dim, mlp_scale_factor=sf, projection_size=num_classes,
                                          init_method='He', activation='relu')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')
        self.proba_head = torch.nn.Sequential(
            MLPHead(self.feat_dim, mlp_scale_factor=1, projection_size=3, init_method='He', activation=activation),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        x = self.neck(x).view(N, -1)
        logits = self.classfier_head(x)
        prob = self.proba_head(x)
        return {'logits': logits, 'prob': prob}


def robust_train(net, optimizer, trainloader, n_classes, config, train_loss_meter, train_accuracy_meter,
                 train_loss_meter1, train_accuracy_meter1, corrected_label_distributions, ema_label_distributions, weights):
    net[0].train()
    net[1].train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='ROBUST TRAINING')

    for it, sample in enumerate(pbar):
        indices = sample['index'].to(device)
        x, x_s = sample['data']
        x, x_s = x.to(device), x_s.to(device)
        y = sample['label'].to(device)
        with autocast():

            # robust train net0
            outputs0 = net[0](x)
            logits0 = outputs0['logits'] if type(outputs0) is dict else outputs0
            px0 = logits0.softmax(dim=1)
            score0, prediction0 = px0.max(dim=1)

            # normal train net1
            outputs1 = net[1](x)
            logits1 = outputs1['logits'] if type(outputs1) is dict else outputs1
            px1 = logits1.softmax(dim=1)
            score1, prediction1 = px1.max(dim=1)

            train_acc0 = accuracy(logits0, y, topk=(1,))
            train_acc1 = accuracy(logits1, y, topk=(1,))

            if train_acc0[0] !=0 and ((train_acc1[0]-train_acc0[0])/train_acc0[0] < config.tau):
                idx_selected = (((prediction0 == prediction1) & (prediction0 == y)) & (prediction1 == y)) | (( prediction0 != y ) & ( prediction1 == y ))| (( prediction0 == y) & ( prediction1 != y ))
            else:
                idx_selected = (((prediction0 == prediction1) & (prediction0 == y)) & (prediction1 == y)) | (( prediction0 == y) & ( prediction1 != y ))
            weights[indices[idx_selected]] += 1


            labels_ = torch.zeros(x.size(0), n_classes).cuda().scatter_(1, y.view(-1, 1), 1)
            label_ema = ema_label_distributions[indices.cpu(), :].clone().cuda()
            aph = 0.8
            label_ema = label_ema * aph + px0 * (1 - aph)

            weights_ = torch.FloatTensor(idx_selected).view(-1, 1).cuda() if isinstance(idx_selected, np.ndarray) else idx_selected.float().view(-1, 1)

            pseudo_label_l = labels_ * weights_ + label_ema * (1 - weights_)
            pesudo = pseudo_label_l.max(dim=1)[1]


            loss0 = (F.cross_entropy(logits0[idx_selected], y[idx_selected], reduction="none", label_smoothing= 0.35) * (
                        (weights[indices[idx_selected]]) / (epoch + 1))).mean()
            loss1 = F.cross_entropy(logits1, y, label_smoothing= 0.35)


            outputs_CR = net[0](x_s)
            logits_CR = outputs_CR['logits'] if type(outputs_CR) is dict else outputs_CR
            loss_CR = F.cross_entropy(logits_CR, pesudo, label_smoothing= 0.35)
            weight = 0.8
            loss0 = loss0 * weight + loss_CR * (1 - weight)

        optimizer[0].zero_grad()
        loss0.backward()
        optimizer[0].step()

        optimizer[1].zero_grad()
        loss1.backward()
        optimizer[1].step()

        corrected_label_distributions[indices] = pseudo_label_l.detach().clone().cpu().data
        ema_label_distributions[indices] = label_ema.detach().clone().cpu().data
        del label_ema

        train_accuracy_meter.update(train_acc0[0], x.size(0))
        train_loss_meter.update(loss0.detach().cpu().item(), x.size(0))
        train_accuracy_meter1.update(train_acc1[0], x.size(0))
        train_loss_meter1.update(loss1.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(
            f'TrainAcc_net_0: {train_accuracy_meter.avg:3.2f}%; TrainAcc_net_1: {train_accuracy_meter1.avg:3.2f}%; TrainLoss_net_0: {train_loss_meter.avg:3.2f}; TrainLoss_net_1: {train_loss_meter1.avg:3.2f}')

def warmup(net, optimizer, trainloader, train_loss_meter, train_accuracy_meter, train_loss_meter1,
           train_accuracy_meter1, config, model_ema, ema, weight):
    net[0].train()
    net[1].train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='WARMUP TRAINING')
    for it, sample in enumerate(pbar):
        x, _ = sample['data']
        indices = sample['index'].to(device)
        x = x.to(device)
        y = sample['label'].to(device)
        with autocast():
            # warmup net0
            outputs = net[0](x)
            logits = outputs['logits'] if type(outputs) is dict else outputs
            loss_ce = F.cross_entropy(logits, y, label_smoothing= 0.35)
            train_acc0 = accuracy(logits, y, topk=(1,))
            penalty = conf_penalty(logits)
            loss0 = loss_ce + penalty

            # warmup net1
            outputs = net[1](x)
            logits = outputs['logits'] if type(outputs) is dict else outputs
            loss1 = F.cross_entropy(logits, y, label_smoothing= 0.35)
            train_acc1 = accuracy(logits, y, topk=(1,))
            weight[indices] += 1

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        loss0.backward()
        loss1.backward()
        optimizer[0].step()
        optimizer[1].step()

        train_accuracy_meter.update(train_acc0[0], x.size(0))
        train_loss_meter.update(loss0.detach().cpu().item(), x.size(0))
        train_accuracy_meter1.update(train_acc1[0], x.size(0))
        train_loss_meter1.update(loss1.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(
            f'TrainAcc_net_0: {train_accuracy_meter.avg:3.2f}%; TrainAcc_net_1: {train_accuracy_meter1.avg:3.2f}%; TrainLoss_net_0: {train_loss_meter.avg:3.2f}; TrainLoss_net_1: {train_loss_meter1.avg:3.2f}')


def build_model(num_classes, config):
    if config.model == "CNN":
        net1 = CNN(input_channel=3, n_outputs=num_classes, activation='tanh')
        net2 = CNN(input_channel=3, n_outputs=num_classes, activation='tanh')
        # net2 = ResNet(arch="resnet18", num_classes=num_classes, pretrained=False)
    elif config.model == "Resnet50":
        net1 = ResNet(arch="resnet50", num_classes=num_classes, pretrained=True)
        net2 = ResNet(arch="resnet50", num_classes=num_classes, pretrained=True)
    elif config.model == "Resnet18":
        net1 = ResNet(arch="resnet18", num_classes=num_classes, pretrained=True)
        net2 = ResNet(arch="resnet18", num_classes=num_classes, pretrained=True)
    net1, net2 = net1.cuda(), net2.cuda()
    return net1, net2


def build_optimizer(net, params):
    if params.opt == 'adam':
        # return build_adam_optimizer(list(net[0].parameters()) + list(net[1].parameters()), params.lr,
        #                             params.weight_decay, amsgrad=False)
        return build_adam_optimizer(net[0].parameters(), params.lr, params.weight_decay,
                                    amsgrad=False), build_adam_optimizer(net[1].parameters(), params.lr1,
                                                                         params.weight_decay, amsgrad=False)
    elif params.opt == 'sgd':
        # return build_sgd_optimizer(list(net[0].parameters()) + list(net[1].parameters()), params.lr,
        #                            params.weight_decay, nesterov=True)
        return build_sgd_optimizer(net[0].parameters(), params.lr, params.weight_decay,
                                   nesterov=True), build_sgd_optimizer(
            net[1].parameters(), params.lr1, params.weight_decay, nesterov=True)

    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet!')


def build_loader(params):
    dataset_name = params.dataset

    if dataset_name.startswith('cif'):
        num_classes = int(100 * (1 - config.openset_ratio))
        transform = build_transform(rescale_size=32, crop_size=32)
        dataset = build_cifar100n_dataset("./data/cifar100",
                                          CLDataTransform(transform['cifar_train'],
                                                          transform['cifar_train_strong_aug']),
                                          transform['cifar_test'], noise_type=params.noise_type,
                                          openset_ratio=params.openset_ratio, closeset_ratio=params.closeset_ratio)
        trainloader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True)
        test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=4, pin_memory=False)

        num_samples = len(trainloader.dataset)
        return_dict = {'trainloader': trainloader, 'num_classes': num_classes, 'num_samples': num_samples,
                       'dataset': dataset_name}
        return_dict['test_loader'] = test_loader
    if dataset_name.startswith('web-'):
        class_ = {"web-aircraft": 100, "web-bird": 200, "web-car": 196}
        num_classes = class_[dataset_name]
        transform = build_transform(rescale_size=448, crop_size=448)
        dataset = build_webfg_dataset(os.path.join('Datasets', dataset_name),
                                      CLDataTransform(transform['train'], transform["train_strong_aug"]),
                                      transform['test'])
        trainloader = DataLoader(dataset["train"], batch_size=params.batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True)
        test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=4,
                                 pin_memory=False)
        num_samples = len(trainloader.dataset)
        return_dict = {'trainloader': trainloader, 'num_classes': num_classes, 'num_samples': num_samples,
                       'dataset': dataset_name}
        return_dict['test_loader'] = test_loader

    if dataset_name.startswith('food'):
        # class_ = {"web-aircraft": 100, "web-bird": 200, "web-car": 196}
        num_classes = 200
        transform = build_transform(rescale_size=256, crop_size=224)
        dataset = build_food101n_dataset('Datasets/food-101', CLDataTransform(transform['train'], transform["train_strong_aug"]), transform['test'])
        trainloader = DataLoader(dataset["train"], batch_size=params.batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True)
        test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=4,
                                 pin_memory=False)
        num_samples = len(trainloader.dataset)
        return_dict = {'trainloader': trainloader, 'num_classes': num_classes, 'num_samples': num_samples,
                       'dataset': dataset_name}
        return_dict['test_loader'] = test_loader

    return return_dict


def get_baseline_stats(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    test_acc_list = []
    test_acc_list2 = []
    valid_epoch = []
    # valid_epoch = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(1, min(11, len(lines) + 1)):
        line = lines[-idx].strip()
        epoch, test_acc = line.split(' | ')[0], line.split(' | ')[3]
        ep = int(epoch.split(': ')[1])
        valid_epoch.append(ep)
        # assert ep in valid_epoch, ep
        if '/' not in test_acc:
            test_acc_list.append(float(test_acc.split(': ')[1]))
        else:
            test_acc1, test_acc2 = map(lambda x: float(x), test_acc.split(': ')[1].lstrip('(').rstrip(')').split('/'))
            test_acc_list.append(test_acc1)
            test_acc_list2.append(test_acc2)
    if len(test_acc_list2) == 0:
        test_acc_list = np.array(test_acc_list)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f}, std: {test_acc_list.std():.2f}')
        print(f' {test_acc_list.mean():.2f}±{test_acc_list.std():.2f}')
        return {'mean': test_acc_list.mean(), 'std': test_acc_list.std(), 'valid_epoch': valid_epoch}
    else:
        test_acc_list = np.array(test_acc_list)
        test_acc_list2 = np.array(test_acc_list2)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f} , std: {test_acc_list.std():.2f}')
        print(f'mean: {test_acc_list2.mean():.2f} , std: {test_acc_list2.std():.2f}')
        print(
            f' {test_acc_list.mean():.2f}±{test_acc_list.std():.2f}  ,  {test_acc_list2.mean():.2f}±{test_acc_list2.std():.2f} ')
        return {'mean1': test_acc_list.mean(), 'std1': test_acc_list.std(),
                'mean2': test_acc_list2.mean(), 'std2': test_acc_list2.std(),
                'valid_epoch': valid_epoch}


def wrapup_training(result_dir, best_accuracy):
    stats = get_baseline_stats(f'{result_dir}/log.txt')
    with open(f'{result_dir}/result_stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
    os.rename(result_dir, f"{result_dir}-bestAcc_{best_accuracy:.4f}-lastAcc_{stats['mean']:.4f}")


def init_corrected_labels(num_samples, num_classes, trainloader, soft=True):
    corr_label_dist = torch.zeros((num_samples, num_classes)).float()
    with torch.no_grad():
        for it, sample in enumerate(trainloader):
            indices = sample['index']
            y = sample['label']
            y_dist = F.one_hot(y, num_classes).float()
            if soft: y_dist = F.softmax(y_dist*10, dim=1)
            assert y_dist.device.type == 'cpu'
            corr_label_dist[indices] = y_dist
    return corr_label_dist

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=str, default='cosine:20,5e-4,100')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--warmup-lr', type=float, default=0.001)
    parser.add_argument('--lr1', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save-weights', type=bool, default=False)

    parser.add_argument('--dataset', type=str, default='cifar100nc')
    parser.add_argument('--noise-type', type=str, default='symmetric')
    parser.add_argument('--closeset-ratio', type=float, default=0.2)
    parser.add_argument('--database', type=str, default='./dataset')
    parser.add_argument('--model', type=str, default='CNN')

    parser.add_argument('--ablation', type=bool, default=False)
    parser.add_argument('--method', type=str, default="ours")
    parser.add_argument('--tau', type=float, default=0.025)

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    config = parse_args()

    # assert config.dataset in ['cifar100nc', 'cifar80no']
    # assert config.noise_type in ['symmetric', 'asymmetric']
    # config.openset_ratio = 0.0 if config.dataset == 'cifar100nc' else 0.2

    init_seeds(config.seed)
    device = set_device(config.gpu)

    # bulid logger
    logger, result_dir = build_logger(config)
    logger.msg(str(config))

    # create dataloader
    loader_dict = build_loader(config)
    dataset_name, n_classes, n_samples = loader_dict['dataset'], loader_dict['num_classes'], loader_dict['num_samples']

    # create model
    model = build_model(n_classes, config)

    # create optimizer & lr_plan or lr_scheduler
    optim = build_optimizer(model, config)
    lr_plan = [build_lr_plan(config.lr, config.epochs, config.warmup_epochs, config.warmup_lr, decay=config.lr_decay),
               build_lr_plan(config.lr1, config.epochs, config.warmup_epochs, config.lr1, decay="baseline")]

    # for training
    best_accuracy, best_epoch = 0.0, None
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    train_loss_meter1 = AverageMeter()
    train_accuracy_meter1 = AverageMeter()

    weight = torch.zeros(n_samples,dtype=torch.int32).cuda()

    corrected_label_distributions = init_corrected_labels(n_samples, n_classes, loader_dict['trainloader'], soft=False)
    ema_label_distributions = copy.deepcopy(corrected_label_distributions)


    model_ema = copy.deepcopy(model[0])
    ema = EMA(model_ema, alpha=0.99)
    ema.apply_shadow(model_ema)

    for epoch in range(config.epochs):
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        train_loss_meter1.reset()
        train_accuracy_meter1.reset()
        adjust_lr(optim[0], lr_plan[0][epoch])
        adjust_lr(optim[1], lr_plan[1][epoch])
        input_loader = loader_dict['trainloader']
        if epoch < config.warmup_epochs and config.method not in ["small_loss", "co_teaching"]:
            warmup(model, optim, input_loader, train_loss_meter, train_accuracy_meter, train_loss_meter1,
                   train_accuracy_meter1, config, model_ema, ema, weight)
        else:
            robust_train(model, optim, input_loader,  n_classes,config, train_loss_meter, train_accuracy_meter, train_loss_meter1, train_accuracy_meter1, corrected_label_distributions, ema_label_distributions, weight)


        eval_result = evaluate_cls_acc(loader_dict['test_loader'], model, device)
        test_accuracy = eval_result['accuracy']

        if test_accuracy[0] > best_accuracy:
            best_accuracy = test_accuracy[0]
            best_epoch = epoch

        logger.info(
            f'>>  Epoch: {epoch} | Net [0] loss: {train_loss_meter.avg:.2f} | train acc: {train_accuracy_meter.avg:.2f} | test acc: {test_accuracy[0]:.2f} | Net [1] loss: {train_loss_meter1.avg:.2f} | train acc: {train_accuracy_meter1.avg:.2f} | test acc: {test_accuracy[1]:.2f} @ best epoch: {best_epoch}, best accuracy: {best_accuracy:.2f}.')
    wrapup_training(result_dir, best_accuracy)
