import torch
import math

#18 Aug update:
#Following feedback will not attempt to solve a linear problem, but a nonlinear one. 
#Will first use linear approximator, then add higher powers and see how it improves.
#It takes more and more iterations to converge as it gets more complex but gets closer to real solution.
#Next will look at specifying layers.

class ODE1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #Define parameters/weights
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))   
        self.d = torch.nn.Parameter(torch.randn(()))  
        self.e = torch.nn.Parameter(torch.randn(()))
        self.f = torch.nn.Parameter(torch.randn(()))
        self.g = torch.nn.Parameter(torch.randn(()))
        self.h = torch.nn.Parameter(torch.randn(()))
        self.i = torch.nn.Parameter(torch.randn(()))
        self.j = torch.nn.Parameter(torch.randn(()))

    def forward(self, x, y, z):
        #Functional and Relu are the basic ones
        Force = (self.a * x + self.e * x ** 2 + self.h * x ** 3 +
        self.b * y + self.f * y ** 2 + self.i * y ** 3 +
        self.c * z + self.g * z ** 2 + self.j * z ** 3 +
        self.d)
        return Force
    def string(self):
        return f'First Order ODE is approximated by:\nF = {self.a.item()} x + {self.e.item()} x^2 + {self.h.item()} x^3 + {self.b.item()} b + {self.f.item()} b^2 + {self.i.item()} b^3 + {self.c.item()} c + {self.g.item()} c^2 + {self.j.item()} c^3 + {self.d.item()}'

#Real solution
x = torch.linspace(-math.pi,math.pi,100)
y = torch.linspace(-math.pi,math.pi,100)
z = torch.linspace(-1,1,100)

Force = (torch.sin(x) + 2*torch.cos(y) + z**2)

#Instantiate model
model = ODE1()

#loss function and optimizer
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)


#Main training loop
for t in range(100000):
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