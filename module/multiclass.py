from copy import deepcopy
from typing import List, Optional
from tensorboardX import SummaryWriter
from torch.autograd import backward
from .ModuleBase import LZABase
from . import LinearRegression
from optim import *
from ScalerRecorder import ScalerRecorder
import torch

class MultiClass(LZABase):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, type:str='ovo') -> None:
        if type not in ['ovo', 'ova', 'softmax']:
            raise
        padding = False if type in ['ovo', 'ova'] else True
        self.type = type
        super().__init__(X, y, padding=padding)

    def train(self, epoches:int=100, batch_size:int=1, optim='SGD'):
        optim = globals()[optim]
        if self.type == 'ovo':
            self.train_ovo(epoches, batch_size, optim)
        elif self.type == 'ova':
            self.train_ova(epoches, optim)
        elif self.type == 'softmax':
            pass
        else:
            raise NotImplementedError
    
    def train_ova(self, epoch: int, optim=SGD):
        pass

    def train_ovo(self, epoches: int, batch_size: int, optim=SGD):
        self.optim = optim([self.w], lr=1e-3)

        def create_classifiers():
            classes = [{i, j} for i in range(self.classnum) for j in range(i+1, self.classnum)]
            reval = {}
            for pair in classes:
                l_pair = list(pair)
                key = ' '.join([str(c) for c in pair])
                x = self.X[(self.y == l_pair[0]) | (self.y == l_pair[1])]
                y = self.y[(self.y == l_pair[0]) | (self.y == l_pair[1])]
                local_to_origin = {i:l_pair[i] for i in range(2)}
                origin_to_local = {l_pair[i]:i for i in range(2)}
                clf = LinearRegression(x, y)
                reval[key] = {
                    'local2origin': local_to_origin,
                    'origin2local': origin_to_local,
                    'clf': clf
                }
            
            return reval

        self.classifiers = create_classifiers()
        for d in self.classifiers.values():
            d['clf'].train()
    
    def get_acc(self):
        def get_ovo_acc():
            c = deepcopy
            test_x, test_y = c(self.X), c(self.y)
            datanum = len(test_x)
            acc = 0
            for x, y in zip(test_x, test_y):
                res = {}
                for idx in self.classifiers:
                    if str(y.int().item()) not in idx:
                        continue
                    clf = self.classifiers[idx]
                    pred = clf['local2origin'][clf['clf'].predict(x).item()]
                    if pred not in res:
                        res[pred] = 1
                    else:
                        res[pred] += 1
                f = torch.argmax(torch.tensor(list(res.values())))
                if list(res.keys())[f.item()] == y.int().item():
                    acc += 1
            print(f'Acc for ovo method = {acc / datanum}')

        if self.type == 'ovo':
            get_ovo_acc()
        pass

    def get_loss(self, X, y):
        t = torch.exp(-y * torch.matmul(X, self.w))
        t = torch.log(1 + t)
        t = torch.mean(t)
        return t
    
    def get_backward(self, X, y):
        v = torch.exp(-y * torch.matmul(X, self.w))
        top = -(y * v).unsqueeze(-1) * X
        bot = 1 + v
        ret = torch.mean(top / (bot).unsqueeze(-1), dim=0)

        return ret