import numpy as np
import torch

sb = torch.tensor([[339437.0000, 592107.5625],
        [554612.6250, 967455.9375]])
print(sb)
sbn = torch.inverse(sb)
print(torch.matmul(sbn, sb))
print("="*20)
sb = sb.numpy()
print(sb)
sb = np.array([[339437.0000, 592107.5625],
        [554612.6250, 967455.9375]])
print(sb)
sbn = np.linalg.inv(sb)
print(sbn@sb)

# Decorator

def my_inverse(X):
	rank = torch.matrix_rank(X)
	if rank < X.shape[0]:
		raise ValueError("Not full rank")
	return torch.inverse(X)




#%% Malloc - 

def watch_malloc(func):
	def ret_func(*args, **wargs):
		print('Call Decorator')
		func(*args, **wargs)
	return ret_func

@watch_malloc
def malloc(size):
	print('Call Malloc')
	pass



# %%
