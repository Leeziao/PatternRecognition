from typing import Optional
from .ModuleBase import LZABase
from tensorboardX import SummaryWriter
from optim import *
from ScalerRecorder import ScalerRecorder
import torch

class Fisher(LZABase):
    def __init__(self, *args, **wargs) -> None:
        super().__init__(*args, **wargs)
        self.criterion = 0
        self.classes = torch.sort(torch.unique(self.y)).values

    def train(self):
        assert(len(self.classes) == 2)
        X, Var, miu = [], [], []
        for c in self.classes:
            cur_X = self.X[self.y == c]
            X.append(cur_X)
            Var.append(torch.matmul(cur_X.transpose(0, 1), cur_X))
            miu.append(torch.mean(cur_X, dim=0))

        self.w = torch.matmul(torch.inverse(sum(Var)), miu[0] - miu[1])
        self.criterion = sum(self.w * (sum(miu))) / 2

        print('criterion = ', self.criterion.item())

    def get_acc(self, w: Optional[torch.Tensor]=None):
        acc = 0
        w = w if w != None else self.w

        for x, y in zip(self.X, self.y):
            # FIXME: There is a minor bug, but I don't want to fix
            if torch.matmul(w, x) * y < self.criterion:
                acc += 1
        return acc / self.data_num