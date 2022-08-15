import torch
x = torch.linspace(-10,10,5) 
y = 2*x
z = y - x/2
print(z)

print(torch.is_tensor(z))