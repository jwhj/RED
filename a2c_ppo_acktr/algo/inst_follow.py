import sys
import os
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

MINITS_SCRIPTS = os.path.join(os.environ['MINIRTS_ROOT'], 'scripts/behavior_clone')
if MINITS_SCRIPTS not in sys.path:
    sys.path.append(MINITS_SCRIPTS)
import common_utils
from train_executor_with_unit_cont import train as train_
from dataset import BehaviorCloneDataset


class InstFollow:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimizer = optim.Adamax(
            self.model.parameters(), lr=2e-3, betas=(0.9, 0.999)
        )
        self.grad_clip = 0.5
        self.dataset = BehaviorCloneDataset(
            os.path.join(os.environ['MINIRTS_ROOT'], 'data/dataset/train.json'),
            11,
            50,
            25,
            inst_dict=self.model.inst_dict,
            word_based=True,
        )
        self.loader = DataLoader(self.dataset, 16, True, num_workers=20)
        self.data_iter = iter(self.loader)
        self.stat = common_utils.MultiCounter(None)

    def train(self, epoch: int):
        self.model.train()
        self.stat.start_timer()
        losses = []
        while True:
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.loader)
                batch = next(self.data_iter)
            batch = common_utils.to_device(batch, self.device)
            self.optimizer.zero_grad()
            loss, all_losses = self.model.compute_loss(batch, mean=False)
            loss = loss.mean()
            losses.append(loss.item())
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            for key, val in all_losses.items():
                v = val.mean().item()
                self.stat[key].feed(v)
            if len(losses) >= 5 and np.mean(losses[-5:]) < 2.3:
                break
        self.stat.summary(epoch)
        self.stat.reset()
