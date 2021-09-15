from typing import Optional
import torch
import matplotlib.pyplot as plt

class LZABase:
    r"""
    The Base Module for Methods
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = torch.cat([torch.ones_like(X.transpose(0, 1)[:1]).transpose(0,1), X], dim=1)
        self.dim = X.shape[1]
        self.X = X
        self.y = y
        self.data_num = X.shape[0]
        self.w = torch.rand_like(X[0])
        self.w = torch.zeros_like(X[0])
        print(self.data_num)
    
    def train(self, update_steps: int=10):
        pass
    
    def plot(self):
        if self.dim != 3:
            print('Dim = 2 required')
            return
        X = torch.index_select(self.X, -1, torch.tensor(list(range(1, self.dim))))
        x0 = X.transpose(0, 1)[0]
        x1 = X.transpose(0, 1)[1]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x0[self.y == 1], x1[self.y == 1])
        ax.scatter(x0[self.y == -1], x1[self.y == -1])

        w = self.w
        x_lim = ax.get_xlim()
        y = [-w[0]/w[2] - w[1]/w[2] * x for x in x_lim]

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
