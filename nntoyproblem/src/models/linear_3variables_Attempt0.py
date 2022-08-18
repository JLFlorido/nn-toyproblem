import torch
import math

class ODE1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #Define parameters/weights
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))   
    def forward(self, x, y, z):
        #Functional and Relu are the basic ones
        Force = self.a * x + self.b * y + self.c * z
        return Force
    def string(self):
        return f'First Order ODE is approximated by:\nForce = {self.a.item()} x + {self.b.item()} y + {self.c.item()} z'

#Real solution
x = torch.linspace(-10,10,1000)
y = torch.linspace(-10,10,1000)
z = torch.linspace(-10,10,1000)
Force = (x + 2*y + (2/3)*z)

#Instantiate model
model = ODE1()

#loss function and optimizer
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)


#Main training loop
for t in range(2000):
    #forward pass first
    Force_Pred = model(x,y,z)
    # Compute and print loss.
    loss = loss_fn(Force_Pred, Force)
    
    if t % 100 == 99:
        print(t, loss.item())

    #The step I don't understand
    optimizer.zero_grad()   #Alternatively: model.zero_grad()
    #Backward pass
    loss.backward()
    #Optimizer update
    optimizer.step()

print(f'\n{model.string()}')