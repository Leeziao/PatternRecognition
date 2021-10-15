from typing import Optional
import torch
import matplotlib.pyplot as plt
from .ModuleBase import LZABase

class PLA(LZABase):
    def train(self, update_steps: int=10):
        for i in range(update_steps):
            x, y = self.X[i % self.data_num], self.y[i % self.data_num]
            self.w = self.w if torch.matmul(self.w, x)*y > 0 else self.w + x*y

class Pocket(LZABase):
    def train(self, update_steps: int=10, verbose=False):
        tmp_w = self.w
        Change = False
        for i in range(update_steps):
            item_idx = torch.randint(low=0, high=self.data_num, size=[1]).squeeze()
            x, y = self.X[item_idx], self.y[item_idx]
            if torch.matmul(tmp_w, x) * y <= 0:
                tmp_w = tmp_w + x*y
                Change = True
            if self.get_acc(tmp_w) > self.get_acc(self.w):
                self.w = tmp_w
            if verbose and Change:
                print(f'idx={item_idx}, x={x}, y={y}, tmp_w={tmp_w}, w={self.w}')
                Change = False


if __name__ == '__main__':
    X = torch.tensor([[1, 3],
                    [2, 3]])
    y = torch.tensor([1, -1])
    # p = PLA(X, y)
    p = Pocket(X, y)
    p.train(100)
    p.plot()
    print(p.w)

    # pla = PLA(X, y)
    # pla.train(10)
    # pla.plot()