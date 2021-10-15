from typing import Optional
from matplotlib import pyplot as plt
from matplotlib import ticker
from tensorboardX import SummaryWriter
from torch.autograd import backward
from .ModuleBase import LZABase
from optim import *
from ScalerRecorder import ScalerRecorder
from qpsolvers import cvxopt_solve_qp as qp_solver
from copy import deepcopy
import numpy as np
import torch

class SVM(LZABase):
    def __init__(self, *args, **wargs) -> None:
        self.type = 'dual'
        if 'type' in wargs:
            type = wargs.pop('type')
            assert(type in ['dual', 'primal', 'kernel'])
            self.type = type
        self.writer = SummaryWriter()
        self.loss_recoder = ScalerRecorder(self.writer)
        self.kernel = None
        self.w_helper = None

        super().__init__(*args, **wargs)

    def create_P(self):
        P = torch.matmul(self.y.unsqueeze(-1), self.y.unsqueeze(0)).float()
        for i in range(self.data_num):
            for j in range(self.data_num):
                P[i,j] *= self.kernel(self.original_X[i], self.original_X[j])
        
        return P
    
    def train(self, *args, **wargs):
        if self.type == 'dual':
            self.kernel = lambda x, y: (x*y).sum()
            self.train_dual(*args, **wargs)
        elif self.type == 'primal':
            epoch = 100 if 'epoch' not in wargs else wargs['epoch']
            optim = 'SGD' if 'optim' not in wargs else wargs['optim']
            optim = globals()[optim]
            self.train_decent(epoch, optim)
        elif self.type == 'kernel':
            if 'kernel' not in wargs:
                raise
            self.kernel = wargs.pop('kernel')
            self.train_dual(*args, **wargs)
    
    def train_decent(self, epoch, optim):
        self.optim = optim([self.w], lr=0.1)

        for _ in range(epoch):
            s = torch.matmul(self.X, self.w)
            tmp = s * self.y
            loss_value, loss_indices = torch.max(torch.stack([torch.zeros_like(tmp), 1-tmp]), dim=0)
            tot_loss = loss_value.sum()
            self.loss_recoder(tot_loss.item())

            backward_w = (- self.X * self.y.unsqueeze(-1) * loss_indices.unsqueeze(-1)).mean(dim=0)
            print(f'w={self.w}, bp_w={backward_w}')
            self.optim.step([backward_w])

    def train_dual(self):
        c = deepcopy

        A = c(self.y.float().unsqueeze(0)).numpy()
        b = torch.tensor([0.]).numpy()
        G = -torch.eye(self.data_num).float().numpy()
        h = torch.zeros(size=([self.data_num])).float().numpy()
        P = self.create_P().numpy()
        q = -torch.ones(size=([self.data_num])).float().numpy()

        qp_param = [P, q, G, h, A, b]
        qp_param = list(map(lambda x: np.array(x, dtype=np.float64), qp_param))

        alpha = qp_solver(*qp_param)
        alpha = torch.from_numpy(alpha).float()

        print("QP solution: x = {}".format(alpha))

        self.support = (alpha > 1e-5).nonzero(as_tuple=False)

        idx = torch.argmax(alpha)

        tmp_w = alpha * self.y
        self.w_helper = tmp_w
        self.b_helper = 0
        self.b_helper = self.y[idx] - self.evaluate(self.original_X[idx])

        # if self.type == 'dual':
        #     tmp_w = tmp_w.unsqueeze(-1) * self.original_X
        #     tmp_w = torch.sum(tmp_w, dim=0)
        #     b = self.y[idx] - (tmp_w * self.original_X[idx]).sum()
        #     self.w = torch.cat([b, tmp_w])
        
        print('Training Complete')
    
    def evaluate(self, x):
        assert(self.w_helper != None)
        assert(self.b_helper != None)

        K_res = torch.tensor([self.kernel(x_n, x) for x_n in self.original_X])
        K_res = K_res * self.w_helper
        res = sum(K_res) + self.b_helper

        return res
    
    def get_acc(self, w: Optional[torch.Tensor] = None):
        if self.type == 'primal':
            return super().get_acc(w=w)

        acc = 0
        for x, y in zip(self.original_X, self.y):
            if y * self.evaluate(x) > 0:
                acc += 1
        return acc / self.data_num
    
    def plot_contour(self):
        if self.dim != 3 and self.dim != 2:
            print('Dim = 2 required')
            return
        start_idx = 1 if self.padding else 0
        X = torch.index_select(self.X, -1, torch.tensor(list(range(start_idx, self.dim))))
        x0 = X.transpose(0, 1)[0]
        x1 = X.transpose(0, 1)[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(x0[self.y == 1], x1[self.y == 1], s=50)
        ax.scatter(x0[self.y == -1], x1[self.y == -1], s=50)

        x_range = ax.get_xlim()
        y_range = ax.get_ylim()

        x = np.linspace(start=x_range[0], stop=x_range[1], num=20)
        y = np.linspace(start=y_range[0], stop=y_range[1], num=20)

        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)

        for i in range(zz.shape[0]):
            for j in range(zz.shape[1]):
                zz[i,j] = self.evaluate(torch.tensor([xx[i,j], yy[i,j]]).float())
        
        CS = ax.contourf(xx, yy, zz, levels=10, alpha=.3, cmap='coolwarm', locator=ticker.LinearLocator())
        fig.colorbar(CS)
        # ax.clabel(CS, inline=True)

        CS = ax.contour(xx, yy, zz, linewidths=2, linestyles='dashed', levels=[-1, 0, 1], colors='k')
        ax.clabel(CS, inline=True)

        ax.scatter(x0[self.y == 1], x1[self.y == 1], s=50, c='r')
        ax.scatter(x0[self.y == -1], x1[self.y == -1], s=50, c='b')
        plt.plot(x0[self.support], x1[self.support], 'yo', markersize=12)

        plt.title(f'Acc={self.get_acc()}')

        plt.show()

