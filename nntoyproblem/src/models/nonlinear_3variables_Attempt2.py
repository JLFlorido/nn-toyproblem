import torch
import torch.nn.functional as F
from torch import nn
import math

#18 August, second try:
#Trying different optimizers/loss functiosn.
#Going to try to implement an FC network to solve the same problem and compare results
#Can't compare without looking at weights. Could look at predictive capability but rn it's a bit too much.

class JNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Define network layers
        self.layers = nn.Sequential(
            nn.Linear(100,12),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(12,10),
            nn.ReLU(),
            nn.Linear(10,100)
        )

    def forward(self, x):
        #Forward pass
        x = torch.sigmoid(self.layers(x))
        return x

#Instantiate model
model = JNet()

#Input
x = torch.linspace(-math.pi,math.pi,100)

Force_groundtruth = torch.cos(x)

#loss function and optimizer
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


#Main training loop
for iter in range(5000):
#     #forward pass first
    Force_Pred = model(x)
#     # Compute and print loss.
    loss = loss_fn(Force_Pred, Force_groundtruth)   #differnece between ground truth and estimate from forward pass
    
    if iter % 100 == 99:           #This simply prints the loss every 100 iterations
        print(iter, loss.item())

#     #The step I don't understand
    optimizer.zero_grad()   #Alternatively: model.zero_grad()
#     #Backward pass
    loss.backward()         #Uses chain rule to obtain gradients
#     #Optimizer update
    optimizer.step()        #Updates weights using gradient of loss