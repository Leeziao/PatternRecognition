from typing import Any
from .multiclass import *
import matplotlib.pyplot as plt

class MyMNIST(MultiClass):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, type: str = 'ovo') -> None:
        super().__init__(X, y, type=type)
        self.type = 'softmax'
    
    def get_mnist_acc(self, test_x, test_y):
        data_num = len(test_x)
        acc = 0
        for xx, yy in zip(test_x, test_y):
            pred_y = self.predict_softmax(xx, padding=True)
            if pred_y == yy.int().item():
                acc += 1
        
        print(f'Acc for MNIST is {acc / data_num}')

    def plot(self, X, y):
        fig, axes = plt.subplots(nrows=2, ncols=6)
        plot_num = len(axes.ravel())
        X, y = X[:plot_num], y[:plot_num]

        for xx, yy, ax in zip(X, y, axes.ravel()):
            pred_y = self.predict_softmax(xx, padding=True)
            ax.set_title(f'pred = {pred_y.item()}')
            ax.imshow(xx.reshape(28, 28))
            
        plt.show() 
