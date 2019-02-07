# encoding: utf-8
import math
import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class AnnealLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=0.1, last_epoch=-1):
        self.warmup_steps = float(warmup_steps)
        super(AnnealLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [
            base_lr * self.warmup_steps**0.5 * min(
                step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.base_lrs
        ]