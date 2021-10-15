from tensorboardX import SummaryWriter
from .ModuleBase import LZABase
from optim import *
from ScalerRecorder import ScalerRecorder
import torch

class LinearRegression(LZABase):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, type:str='inverse') -> None:
        if not type in ['inverse', 'decent']:
            raise ValueError('Type not exist')
        
        self.type = type
        self.writer = SummaryWriter()
        self.loss_recoder = ScalerRecorder(self.writer)
        super().__init__(X, y)

    def train(self, epoches:int=100, batch_size:int=1, optim='SGD'):
        if self.type == 'inverse':
            self.train_inverse()
        elif self.type == 'decent':
            optim = globals()[optim]
            self.train_decent(epoches, batch_size, optim)

    def train_inverse(self):
        tmp = torch.inverse(torch.matmul(self.X.transpose(0, 1), self.X))
        tmp = torch.matmul(tmp, self.X.transpose(0, 1))
        self.w = torch.matmul(tmp, self.y.float())
    
    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if not x.dim() == 2: 
            raise ValueError
        if self.padding:
            data_num = len(x)
            padding = torch.tensor([1]*data_num).unsqueeze(-1)
            x = torch.cat([padding, x], dim=-1)

        v = torch.matmul(x, self.w)
        return (v > 0).int()

    def train_decent(self, epoches: int, batch_size: int, optim=SGD):
        self.optim = optim([self.w], lr=1e-3)

        for _ in range(epoches):
            tmp = torch.matmul(self.X, self.w) - self.y
            loss = torch.sum(tmp * tmp) / self.data_num
            print(loss)
            self.loss_recoder(loss.item())
            backward_w = 2 * torch.matmul(self.X.transpose(0,1),
                                        torch.matmul(self.X, self.w) - self.y) / self.data_num
            print(f'w={self.w}, bp_w={backward_w}')
            self.optim.step([backward_w])