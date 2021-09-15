from data import data_generator
from module.pla import PLA, Pocket

mean1 = [-5, 0]
mean2 = [0, 5]

x_train, y_train, x_test, y_test = data_generator(mean1, mean2)

method_name = 'Pocket'
method_name = 'PLA'
Method = globals()[method_name]

p = Method(x_train, y_train)
# p = Pocket(x_train, y_train)
p.train(100)
print(f'Running {method_name}')
p.evaluate(x_test, y_test)
p.plot()
