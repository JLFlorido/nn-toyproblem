# -*- coding: utf-8 -*-
import torch
import math

#create class describing own nn module
class Polynomial3(torch.nn.Module):
    def __init__(self):

        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self,x):

        return (self.a) + (self.b * x) + (self.c * x**2) + (self.d * x**3)

    def string(self):

        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

#Instantiate the class defined above
model = Polynomial3()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)



for t in range(2000):

    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = criterion(y_pred, y)
    
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    #model.zero_grad()
    optimizer.zero_grad()   #Why use this over the previous line?
    loss.backward()
    optimizer.step()

# You can access the first layer of `model` like accessing the first item of a list
print(f'Result:{model.string()}')