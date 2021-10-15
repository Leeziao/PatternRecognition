from data import data_generator
from module import *
from sklearn.datasets import load_iris

class solution:
    def __init__(self, solution_num) -> None:
        self.solution_num = solution_num

    def solve(self, *args, **wargs):
        s_name = 'solution' + str(self.solution_num)
        s = getattr(self, s_name)

        if None == s:
            print(f'{s_name} not exist')
            return

        s(*args, **wargs)

    def solution0(self):
        r"""
        Pocket for classification on given dataset, max_iter=20
        """
        X_train = torch.tensor([[0.2, 0.7],
                                [0.3, 0.3],
                                [0.4, 0.5],
                                [0.6, 0.5],
                                [0.1, 0.4],
                                [0.4, 0.6],
                                [0.6, 0.2],
                                [0.7, 0.4],
                                [0.8, 0.6],
                                [0.7, 0.5]])
        y_train = torch.tensor([1]*5+[-1]*5)

        Method = globals()['Pocket']

        p = Method(X_train, y_train)
        p.train(400, verbose=True)
        p.plot()

    def solution1(self, method_type, mean1=[-5, 0], mean2=[0, 5]):
        r"""
        Solution for lecture 1 - PLA and Pocket Algorithm
        args:
            type = 'PLA' |'Pocket
            mean1=[-5, 0]
            mean2=[0, 5]
        """
        if method_type not in ['PLA', 'Pocket']:
            raise ValueError

        x_train, y_train, x_test, y_test = data_generator(mean1, mean2)

        method_name = method_type
        Method = globals()[method_name]

        p = Method(x_train, y_train)
        p.train(100)
        print(f'Running {method_name}')
        p.evaluate(x_test, y_test)
        p.plot()

    def solution2(self, method_type, mean1=[-5, 0], mean2=[0, 5], optim='SGD'):
        r"""
        Solution for lecture 2 - Inverse and Gradient Decent for MSE
        args:
            method_type = 'inverse' | 'decent'
            mean1 = [-5, 0]
            mean2 = [0, 5]
        """

        if method_type not in ['inverse', 'decent']:
            raise ValueError

        x_train, y_train, x_test, y_test = data_generator(mean1, mean2)
        method_name = 'LinearRegression'

        Method = globals()[method_name]

        p = Method(x_train, y_train, method_type)
        p.train(optim=optim)
        p.plot()

    def solution3(self):
        r"""
        Solution for lecture 2 - Inverse and Gradient Decent for MSE
            on given dataset
        """
        x_train = torch.tensor([[0.2, 0.7],
                                [0.3, 0.3],
                                [0.4, 0.5],
                                [0.6, 0.5],
                                [0.1, 0.4],
                                [0.4, 0.6],
                                [0.6, 0.2],
                                [0.7, 0.4],
                                [0.8, 0.6], 
                                [0.7, 0.5]])
        y_train = torch.tensor([1]*5 + [-1]*5)
        method_name = 'LinearRegression'

        Method = globals()[method_name]

        p = Method(x_train, y_train, 'inverse')
        p.train()
        p.plot()

    def solution4(self, mean1=[-5, 0], mean2=[0, 5]):
        r"""
        Solution for lecture 3 - Fisher Discriminate Analysis
        """
        x_train, y_train, x_test, y_test = data_generator(mean1, mean2)
        method_name = 'Fisher'

        Method = globals()[method_name]

        p = Method(x_train, y_train, padding=False)
        p.train()
        p.plot()

    def solution5(self, mean1=[-5, 0], mean2=[0, 5]):
        r"""
        Solution for lecture 3 - Fisher Discriminate Analysis on given dataset
        """
        x_train = torch.tensor([[5, 37],
                                [7, 30],
                                [10, 35],
                                [11.5, 40],
                                [14, 38],
                                [12, 31],
                                [35, 21.5],
                                [39, 21.7],
                                [34, 16], 
                                [37, 17]])
        y_train = torch.tensor([1]*6 + [-1]*4)
        method_name = 'Fisher'

        Method = globals()[method_name]

        p = Method(x_train, y_train, padding=False)
        p.train()
        p.plot()

    def solution6(self, type='dual', mean1=[-5, 0], mean2=[0, 5]):
        r"""
        Solution for lecture 4 - Support Vector Machine for dual and primal solution on given dataset
            args: 
                type = 'dual' | 'primal'
        """
        assert(type in ['dual', 'primal'])

        x_train = torch.tensor([[3, 0],
                                [0, 4],
                                [0, 0]]).float()
        y_train = torch.tensor([1] * 2 + [-1] * 1)
        method_name = 'SVM'

        Method = globals()[method_name]

        p = Method(x_train, y_train, padding=True, type=type)
        p.train()
        p.plot_contour()
        # p.plot()

    def solution7(self, kernel='gauss', mean1=[-5, 0], mean2=[0, 5]):
        r"""
        Solution for lecture 4 - Support Vector Machine
        args:
            kernel: 'gauss' | '2' | '4'
        """
        assert(kernel in ['gauss', '2', '4'])

        x_train, y_train, x_test, y_test = data_generator(mean1, mean2, train_num=160)
        # x_train = torch.tensor([[3, 0],
        #                         [0, 4],
        #                         [0, 0]]).float()
        # y_train = torch.tensor([1] * 2 + [-1] * 1)
        method_name = 'SVM'

        Method = globals()[method_name]

        p = Method(x_train, y_train, padding=True, type='kernel')

        def kernel_func_2(x, y):
            inner_prod = (x*y).sum()
            return 1 + inner_prod + inner_prod*inner_prod

        def kernel_func_4(x, y):
            inner_prod = (x*y).sum()
            return 1 + inner_prod + 0.01 * inner_prod ** 4
            
        def kernel_func_gauss(x, y):
            return torch.exp(-0.1*((x-y) * (x-y)).sum())
            # return torch.exp(-1*((x-y) * (x-y)).sum()) # 这样不行

        kernel_dict = {'gauss': kernel_func_gauss,
                        '2': kernel_func_2,  
                        '4':kernel_func_4}

        p.train(kernel=kernel_dict[kernel])
        # p.train(kernel=lambda x,y: (x*y).sum())
        p.plot_contour()

    def solution8(self, mean1=[-5, 0], mean2=[0, 5]):
        r"""
        Solution for lecture 3 - Logistic Regression
        """
        x_train, y_train, x_test, y_test = data_generator(mean1, mean2)
        method_name = 'LogisticRegression'

        Method = globals()[method_name]

        p = Method(x_train, y_train)
        p.train(epoches=300, optim='Adagrad')
        p.plot()

    def solution9(self, mean1=[-5, 0], mean2=[0, 5]):
        r"""
        Solution for lecture 3 - Logistic Regression
        """
        d = load_iris()
        x_train, y_train = torch.tensor(d['data'], dtype=torch.float32), torch.tensor(d['target'], dtype=torch.float32)
        method_name = 'MultiClass'

        Method = globals()[method_name]

        p = Method(x_train, y_train)
        p.train(epoches=300, optim='Adagrad')
        p.get_acc()
