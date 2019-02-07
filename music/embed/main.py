# coding: utf-8

import os
import sys
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate

import hp
from embed import Embedding
from dataset import *
import warnings
warnings.filterwarnings("ignore")

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

BEST_LOSS = float('inf')

def calculate_accuracy(pred, labels):
    pred = pred.cpu().numpy()
    pred = np.argmax(pred, axis=-1)
    acc = np.sum(pred == labels)/ len(labels)
    return acc

def data_loader(args):
    dsets = {x: MyDataset(x) for x in ['train', 'val', 'test']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args['batch_size'], shuffle=True,collate_fn=collate_fn ,num_workers=args['workers']) for x in ['train', 'val', 'test']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))
    return dset_loaders, dset_sizes

def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def train_test(model, dset_loaders, criterion, epoch, phase, optimizer, args, use_gpu, save_path):
    global BEST_LOSS
    if phase == 'val' or phase == 'test':
        model.eval()
    if phase == 'train':
        model.train()
    if phase == 'train':
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, args['epochs']))
        print('Current Learning rate: {}'.format(showLR(optimizer)))

    running_loss, running_acc, running_all = 0., 0., 0.

    for batch_idx, (spec, labels) in enumerate(dset_loaders[phase]):
        
        if phase == 'train':
            optimizer.zero_grad()
        
        if use_gpu:
            spec = spec.cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)

        pred = model(spec)
        loss = criterion(pred, labels)
        
        if phase == 'train':
            loss.backward()
            optimizer.step()

        running_loss += float(loss) * spec.size(0)
        running_acc  += calculate_accuracy(pred, labels) * spec.size(0)
        running_all += len(spec)

        if batch_idx == 0:
            since = time.time()
        elif batch_idx % args['interval'] == 0 or (batch_idx == len(dset_loaders[phase])-1):
            print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                running_all,
                len(dset_loaders[phase].dataset),
                100. * batch_idx / (len(dset_loaders[phase])-1),
                running_loss / running_all,
                running_acc / running_all,
                time.time()-since,
                (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since)))
    print()
    print('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc: {:.4f}\n'.format(
        phase,
        epoch,
        running_loss / len(dset_loaders[phase].dataset),
        running_acc / len(dset_loaders[phase].dataset),
        ))

    if phase == 'train' and running_loss <= BEST_LOSS:
        torch.save(model.state_dict(), save_path+'/model.pt')
        BEST_LOSS = running_loss
    if phase == 'train':
        return model

def test_adam(args, use_gpu):

    save_path = hp.model_base_path

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    model = Embedding(hp.num_classes, 8)
    # reload model
    model = reload_model(model, args['path'])

    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)

    if use_gpu:
        model = model.cuda()
        criterion.cuda()

    dset_loaders, _ = data_loader(args)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(hp.epochs/5), gamma=0.5, last_epoch=hp.restore_epoch-1)

    if args['test']:
        for p in model.parameters():
            p.requires_grad = False
        train_test(model, dset_loaders, criterion, 0, 'val', optimizer, args, use_gpu, save_path)
        train_test(model, dset_loaders, criterion, 0, 'test', optimizer, args, use_gpu, save_path)
        return

    for epoch in range(hp.restore_epoch, args['epochs']):
        scheduler.step()
        model = train_test(model, dset_loaders, criterion, epoch, 'train', optimizer, args, use_gpu, save_path)
        for p in model.parameters():
            p.requires_grad = False
        train_test(model, dset_loaders, criterion, epoch, 'val', optimizer, args, use_gpu, save_path)
        for p in model.parameters():
            p.requires_grad = True


def main():
    # Settings
    args = {
        'path': hp.model_path,
        'lr': hp.lr,
        'batch_size': hp.batch_size,
        'epochs': hp.epochs,
        'workers': 4,
        'interval': hp.display_interval,
        'test': hp.test
    }

    use_gpu = torch.cuda.is_available()
    test_adam(args, use_gpu)


if __name__ == '__main__':
    main()

