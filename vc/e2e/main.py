# coding: utf-8

# TODO: synthisis

import os
import sys
import time
import random
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate

import hp
from model import E2E
from dataset import *
from lr_scheduler import *
from models.losses import *
from scripts.visual import plot_alignment
import warnings
warnings.filterwarnings("ignore")

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

BEST_LOSS = float('inf')

def data_loader(args):
    dsets = {x: MyDataset(x, hp.dataset) for x in ['train', 'val', 'test']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn ,num_workers=args.workers) for x in ['train', 'val', 'test']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))
    return dset_loaders, dset_sizes

def reload_model(model, logger, path=""):
    if not bool(path):
        logger.info('train from scratch')
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('*** model has been successfully loaded! ***')
        return model

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def train_test(model, dset_loaders, criterion, criterion_st, epoch, phase, optimizer, optimizer_st, args, logger, use_gpu, save_path):
    global BEST_LOSS
    if phase == 'val' or phase == 'test':
        model.eval()
    if phase == 'train':
        model.train()
    if phase == 'train':
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
        logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

    running_mel_loss, running_lin_loss, running_stop_loss, running_all = 0., 0., 0., 0.
    n_priority_freq = 384

    for batch_idx, data in enumerate(dset_loaders[phase]):
        ref = data[0]
        mag = data[1]
        mag_len = data[2]
        mel = data[3]
        mel_len = data[4]
        linear = data[5]
        stop = data[6]
        stop = stop.view(stop.size(0), stop.size(1) // hp.r, -1)
        
        del data

        optimizer.zero_grad()
        optimizer_st.zero_grad()
        
        if use_gpu:
            ref = ref.cuda(non_blocking=True)
            mag = mag.cuda(non_blocking=True)
            mel = mel.cuda(non_blocking=True)
            linear = linear.cuda(non_blocking=True)
            stop = stop.cuda(non_blocking=True)
            mag_len = mag_len.cuda(non_blocking=True)
            mel_len = mel_len.cuda(non_blocking=True)

        mask = sequence_mask(mag_len)

        mel_output, linear_output, alignments, stop_tokens = model(ref, mag, mel, mask)

        if phase == 'train':
            stop_loss = criterion_st(stop_tokens, stop)
            mel_loss = criterion(mel_output, mel, mel_len)
            linear_loss = 0.5 * criterion(linear_output, linear, mel_len)\
                + 0.5 * criterion(linear_output[:, :, :n_priority_freq],
                                linear[:, :, :n_priority_freq],
                                mel_len)
            loss = mel_loss + linear_loss

            # backpass and check the grad norm for spec losses
            loss.backward(retain_graph=True)
            optimizer.step()

            # backpass and check the grad norm for stop loss
            stop_loss.backward()
            optimizer_st.step()

        if hp.test:
            align_img = alignments[0].data.cpu().numpy()
            plot_alignment(align_img)

        running_mel_loss += float(mel_loss) * mag.size(0)
        running_lin_loss += float(linear_loss) * mag.size(0)
        running_stop_loss += float(stop_loss) * mag.size(0)
        running_all += len(mag)

        if batch_idx == 0:
            since = time.time()
        elif batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
            print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tMel Loss: {:.4f}\tMag Loss: {:.4f}\tStop Loss: {:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                running_all,
                len(dset_loaders[phase].dataset),
                100. * batch_idx / (len(dset_loaders[phase])-1),
                running_mel_loss / running_all,
                running_lin_loss / running_all,
                running_stop_loss / running_all,
                time.time()-since,
                (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since)))
    print()
    logger.info('{} Epoch:\t{:2}\tMel Loss: {:.4f}\tMag Loss: {:.4f}\tStop Loss: {:.4f}\n'.format(
        phase,
        epoch,
        running_mel_loss / len(dset_loaders[phase].dataset),
        running_lin_loss / len(dset_loaders[phase].dataset),
        running_stop_loss / len(dset_loaders[phase].dataset),
        ))

    if phase == 'train' and (running_mel_loss + running_lin_loss) <= BEST_LOSS:
        torch.save(model.state_dict(), save_path+'/model.pt')
        BEST_LOSS = (running_mel_loss + running_lin_loss)
    if phase == 'train':
        return model

def test_adam(args, use_gpu):

    save_path = hp.model_base_path

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # logging info
    if hp.test:
        filename = save_path + '/test' + '_' + args.mode+'_'+str(args.lr)+'.txt'
    else:
        filename = save_path + '/' + args.mode + '_' + str(args.lr) + '.txt'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    model = E2E()
    # reload model
    model = reload_model(model, logger, args.path)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_st = optim.Adam(model.decoder.stopnet.parameters(), lr=args.lr)
    
    criterion = L1LossMasked()
    criterion_st = nn.BCELoss()

    if use_gpu:
        model = model.cuda()
        criterion.cuda()
        criterion_st.cuda()

    dset_loaders, _ = data_loader(args)

    scheduler = AnnealLR(optimizer, warmup_steps=hp.warmup_steps, last_epoch=hp.restore_epoch)

    if args.test:
        for p in model.parameters():
            p.requires_grad = False
        train_test(model, dset_loaders, criterion, criterion_st, 0, 'val', optimizer, optimizer_st, args, logger, use_gpu, save_path)
        train_test(model, dset_loaders, criterion, criterion_st, 0, 'test', optimizer, optimizer_st, args, logger, use_gpu, save_path)
        return

    for epoch in range(hp.restore_epoch, args.epochs):
        scheduler.step()
        model = train_test(model, dset_loaders, criterion, criterion_st, epoch, 'train', optimizer, optimizer_st, args, logger, use_gpu, save_path)
        for p in model.parameters():
            p.requires_grad = False
        train_test(model, dset_loaders, criterion, criterion_st, epoch, 'val', optimizer, optimizer_st, args, logger, use_gpu, save_path)
        for p in model.parameters():
            p.requires_grad = True


def main():
    # Settings
    parser = argparse.ArgumentParser(description='Pytorch Video-only CTC Model')
    parser.add_argument('--path', default=hp.model_path, help='path to model')
    parser.add_argument('--mode', default='backendGRU', help='temporalConv, backendGRU, finetuneGRU')
    parser.add_argument('--every-frame', default=True, help='predicition based on every frame')
    parser.add_argument('--lr', default=hp.lr, help='initial learning rate')
    parser.add_argument('--batch-size', default=hp.batch_size, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('--workers', default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=hp.epochs, help='number of total epochs')
    parser.add_argument('--interval', default=hp.display_interval, help='display interval')
    parser.add_argument('--test', default=hp.test, help='perform on the test phase')
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    test_adam(args, use_gpu)


if __name__ == '__main__':
    main()

