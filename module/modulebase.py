from typing import Optional
import torch
import numpy as np
import matplotlib.pyplot as plt

class LZABase:
    r"""
    The Base Module for Methods
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, padding=True) -> None:
        self.padding = padding
        self.original_X = X
        if padding:
            X = torch.cat([torch.ones_like(X.transpose(0, 1)[:1]).transpose(0,1), X], dim=1)
        self.dim = X.shape[1]
        self.X = X
        self.y = y
        assert(X.shape[0] == y.shape[0])
        self.data_num = X.shape[0]
        self.classnum = len(torch.unique(y))
        self.w = torch.zeros_like(X[0])
        print('Number of Training Data:{}'.format(self.data_num))
    
    def train(self):
        raise NotImplementedError
    
    def plot(self):
        if self.dim != 3 and self.dim != 2:
            print(f'Dim = {self.dim}, choosing the first 2 dimension')
        print(f'w={self.w}')
        start_idx = 1 if self.padding else 0
        X = torch.index_select(self.X, -1, torch.tensor(list(range(start_idx, self.dim))))
        x0 = X.transpose(0, 1)[0]
        x1 = X.transpose(0, 1)[1]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for c in torch.unique(self.y):
            ax.scatter(x0[self.y == c], x1[self.y == c])

        w = self.w
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        xx = np.linspace(x_lim[0], x_lim[1])
        yy = np.linspace(y_lim[0], y_lim[1])
        xxx, yyy = np.meshgrid(xx, yy)
        xxx_yyy = np.stack((xxx.reshape(-1), yyy.reshape(-1)), axis=-1)

        if self.padding:
            y = [-w[0]/w[2] - w[1]/w[2] * x for x in x_lim]
        else:
            y = [- w[0]/w[1] * x for x in x_lim]

        # zzz = torch.matmul(torch.tensor(xxx_yyy), w)


        plt.plot(x_lim, y, 'm--')
        plt.title(f'Acc={self.get_acc()}')

        plt.show()
    
    def get_acc(self, w: Optional[torch.Tensor]=None):
        acc = 0
        w = w if w != None else self.w

        for x, y in zip(self.X, self.y):
            if torch.matmul(w, x) * y > 0:
                acc += 1
        return acc / self.data_num
    
    def evaluate(self, X, Y):
        X = torch.cat([torch.ones_like(X.transpose(0, 1)[:1]).transpose(0,1), X], dim=1)
        data_num = X.shape[0]
        acc = 0

        for x, y in zip(X, Y):
            if torch.matmul(self.w, x) * y > 0:
                acc += 1
        
        print(f'Evalution Acc={acc/data_num}')

    @property
    def w(self):
        return self._w 

    @w.setter
    def w(self, value):
        self._w = value
