import torch
import math
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

#In this second attempt, I will try and implement BC. That practice will help for moving onto burger's (I hope).
#On another hand, nn might be too simple and might have to look into using sequential with
#built in modules (Linear, relu) to create a multi-layer network. However, all examples I've seen that do that are 
#for image recognition and the like.

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
    def bc_u_0(x):  #In raissi these are just functions that help relate the bc to its value. so it will be smth like x=0 and you
                     #use it in the code downstream to generate the correct data for the bc. how this is implemented in the model...

#Real solution
x = torch.linspace(-10,10,1000)
y = torch.linspace(-10,10,1000)
z = torch.linspace(-10,10,1000)
Force = (x + 2*y + (2/3)*z)

#Instantiate model
model = ODE1()

#loss function and optimizer
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)   #From Sacchetti 2022


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