from module.pla import PLA, Pocket
import torch
import timeit
import numpy as np

def data_generator(mean1, mean2):
    def helper(mean, shape, label):
        x = torch.randn(*shape)
        x += torch.tensor(mean)
        y = torch.LongTensor([label] * shape[0])

        return x, y

    x1, y1 = helper(mean1, [160, 2], 1)
    x2, y2 = helper(mean2, [160, 2], -1)
    x_train = torch.cat([x1, x2])
    y_train = torch.cat([y1, y2])

    packed = list(zip(x_train, y_train))
    np.random.shuffle(packed)
    x_train, y_train = list(zip(*packed))
    x_train = torch.stack(x_train)
    y_train = torch.stack(y_train)


    x1, y1 = helper(mean1, [40, 2], 1)
    x2, y2 = helper(mean2, [40, 2], -1)
    x_test = torch.cat([x1, x2])
    y_test = torch.cat([y1, y2])

    return x_train, y_train, x_test, y_test
