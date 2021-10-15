from tensorboardX import SummaryWriter
from torch.autograd import backward
from .ModuleBase import LZABase
from optim import *
from ScalerRecorder import ScalerRecorder
import torch

class LogisticRegression(LZABase):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.writer = SummaryWriter()
        self.loss_recoder = ScalerRecorder(self.writer)
        super().__init__(X, y)

    def train(self, epoches:int=100, batch_size:int=1, optim='SGD'):
        optim = globals()[optim]
        self.train_decent(epoches, batch_size, optim)

    def train_decent(self, epoches: int, batch_size: int, optim=SGD):
        self.optim = optim([self.w], lr=1e-3)

        for _ in range(epoches):
            loss = self.get_loss(self.X, self.y)
            self.loss_recoder(loss.item())

            backward_w = self.get_backward(self.X, self.y)

            print(f'w={self.w}, bp_w={backward_w}')
            self.optim.step([backward_w])
    
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