import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from ScalerRecorder import ScalerRecorder
from tqdm import tqdm


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.avg_pool2d(F.sigmoid(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.avg_pool2d(F.sigmoid(self.conv2(x)), 2)
        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return x


class LeNetMNIST:
    def __init__(self,
                X: torch.Tensor,
                y: torch.Tensor,
                X_test: torch.Tensor = None,
                y_test: torch.Tensor = None):
        self.X = X
        self.y = y
        self.X_test, self.y_test = None, None

        if len(X) != len(y):
            raise
        if X_test != None and y_test != None:
            assert(len(X_test) == len(y_test))
            self.X_test, self.y_test = X_test, y_test

        self.datanum = len(X)
        self.Model = LeNet()

        self.writer = SummaryWriter()
        self.loss_recorder = ScalerRecorder(self.writer)
        self.acc_recorder = ScalerRecorder(self.writer, 'Acc')

    def get_acc(self, X, y):
        with torch.no_grad():
            z = self.Model(X)
            y_pred = torch.argmax(z, dim=-1)
            acc = (y_pred == y).float().mean().item()
        print(f'Acc={acc}')
        return acc

    def train(self, epoches: int = 10, batch_size: int = 256):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.Model.parameters())

        for epoch in range(epoches):
            if batch_size > self.datanum:
                batch_size = self.datanum
            pbar = tqdm(range((self.datanum // batch_size)))
            for it in pbar:
                x_batch, y_batch = self.X[batch_size*it:batch_size *
                                          (it+1)], self.y[batch_size*it:batch_size*(it+1)]
                z_batch = self.Model(x_batch)
                loss = criterion(z_batch, y_batch)

                self.loss_recorder(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(f'loss={round(loss.item(), 3)}')

            acc = self.get_acc(self.X_test, self.y_test)
            self.acc_recorder(acc)
            print(f'Epoch {epoch}, Acc = {acc}')
            