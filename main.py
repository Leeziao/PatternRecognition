from data import data_generator
from module import PLA, Pocket, LinearRegression
from solution import solution

import sys
sys.path.append('.')

s = solution(9)
s.solve()
# s.solve(kernel='gauss')
