from typing import Optional
import torch
import matplotlib.pyplot as plt
from .modulebase import LZABase

class PLA(LZABase):
    def train(self, update_steps: int=10):
        for i in range(update_steps):
            x, y = self.X[i % self.data_num], self.y[i % self.data_num]
            self.w = self.w if torch.matmul(self.w, x)*y > 0 else self.w + x*y

class Pocket(LZABase):
    def train(self, update_steps: int=10):
        tmp_w = self.w
        for i in range(update_steps):
            x, y = self.X[i % self.data_num], self.y[i % self.data_num]
            tmp_w = tmp_w if torch.matmul(tmp_w, x)*y > 0 else tmp_w + x*y
            if self.get_acc(tmp_w) > self.get_acc(self.w):
                self.w = tmp_w


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