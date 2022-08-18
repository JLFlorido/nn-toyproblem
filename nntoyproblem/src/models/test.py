import torch
# import torch.nn as nn
import numpy as np

# Nx, Ny = [5, 5]
# C0 = torch.rand(Nx, Ny, dtype=torch.double).reshape(1, 1, Nx, Ny)
# C = C0
# laplacian = torch.tensor([[[[0, 1., 0],
#                            [1, -4, 1],
#                            [0, 1, 0]]]], dtype=torch.double)
# print(laplacian)

#--------------------------------------
# x = torch.linspace(-10,10,10)
# y = torch.linspace(-10,10,10)
# z = torch.linspace(-10,10,10)
# Force = (x + 2*y + (2/3)*z)
# print(Force)

#---------------------------------------------
W1 = torch.randn(2,8)
print(W1)