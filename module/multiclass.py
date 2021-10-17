from copy import deepcopy
from typing import List, Optional
from tensorboardX import SummaryWriter
from torch.autograd import backward
from torch.utils import data
from .ModuleBase import LZABase
from . import LinearRegression
from optim import *
from ScalerRecorder import ScalerRecorder
import torch
from copy import deepcopy
from tqdm import tqdm

class MultiClass(LZABase):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, type:str='ovo') -> None:
        if type not in ['ovo', 'ova', 'softmax']:
            raise
        self.padding = False if type in ['ovo', 'ova'] else True

        if type == 'softmax':
            self.writer = SummaryWriter()
            self.loss_recoder = ScalerRecorder(self.writer)

        self.type = type
        super().__init__(X, y)

    def train(self, epoches:int=1, batch_size:int=1, optim='SGD'):
        optim = globals()[optim]
        if self.type == 'ovo':
            self.train_ovo(epoches, batch_size, optim)
        elif self.type == 'ova':
            self.train_ova(epoches, optim)
        elif self.type == 'softmax':
            self.train_softmax(epoches, optim)
        else:
            raise NotImplementedError

    def train_softmax(self, epoch: int, optim=SGD, batch_size: int=256):
        c = deepcopy
        self.w_s = torch.stack([c(self.w) for _ in range(self.classnum)], dim=0).transpose(0, 1)

        self.optim = optim([self.w_s], lr=1e-6)

        if batch_size > self.data_num:
            batch_size = self.data_num

        for _ in range(epoch):
            pbar = tqdm(range(self.data_num // batch_size))
            for it in pbar:
                x_batch, y_batch = self.X[batch_size*it:batch_size*(it+1)], self.y[batch_size*it:batch_size*(it+1)]

                loss = self.get_loss(x_batch, y_batch)
                pbar.set_description(f'loss={round(loss.item(),3)}')
                self.loss_recoder(loss.item())

                backward_w = self.get_backward(x_batch, y_batch)

                self.optim.step([backward_w])
    
    def predict_softmax(self, X, padding=None):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if X.dim() != 2:
            raise
        
        if padding != None and self.padding:
            one = torch.tensor([1.]*len(X))
            X = torch.cat([one.unsqueeze(-1), X], dim=-1)

        
        y_hat = torch.matmul(X, self.w_s)
        reval = torch.argmax(y_hat, dim=-1)

        return reval

    
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
        
        def get_softmax_acc():
            c = deepcopy
            test_x, test_y = c(self.X), c(self.y)
            data_num = len(test_x)
            acc = 0
            for x, y in zip(test_x, test_y):
                v = self.predict_softmax(x)
                if v == y.int().item():
                    acc += 1
            
            print(f'Acc for softmax method = {acc / data_num}')


        if self.type == 'ovo':
            get_ovo_acc()
        elif self.type == 'softmax':
            get_softmax_acc()

    def get_loss(self, X, y):
        y_onehot = torch.eye(self.classnum)[y.long()]
        y_hat = torch.matmul(X, self.w_s)
        y_hat = torch.clamp(y_hat, 1e-5, 1)
        loss = -(y_onehot * torch.log(y_hat)).mean()
        if torch.isinf(loss) or torch.isnan(loss):
            raise ValueError
        return loss
    
    def get_backward(self, X, y):
        y_hat = torch.matmul(X, self.w_s)
        t = torch.bmm(X.unsqueeze(-1), y_hat.unsqueeze(1))
        for i in range(len(X)):
            t[i, :, y[i].long()] -= X[i]
        t = t.mean(dim = 0)

        return t