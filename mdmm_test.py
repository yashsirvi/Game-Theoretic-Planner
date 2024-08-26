import mdmm
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

NUM_ITERATIONS = 100 

class fx(nn.Module):
    def __init__(self):
        super(fx, self).__init__()
        self.x = nn.Parameter(torch.tensor(0.))
        self.y = nn.Parameter(torch.tensor(0.))
        
    def forward(self):
        return self.x**2 + self.y**2
        


model = fx()
constraint = mdmm.EqConstraint(lambda: model.x + model.y - 1, 0)
mdmm_module = mdmm.MDMM([constraint])
opt = mdmm_module.make_optimizer(model.parameters(), optimizer=Adam, lr=0.1)

from time import time as t
prev = t()
for i in range(NUM_ITERATIONS):
    loss = model()
    mdmm_return = mdmm_module(loss)
    opt.zero_grad()
    mdmm_return.value.backward()
    opt.step()
    # loss.backward()
    print(model.x,"\n",model.y)
final = t()
print("Time taken: ", final-prev)