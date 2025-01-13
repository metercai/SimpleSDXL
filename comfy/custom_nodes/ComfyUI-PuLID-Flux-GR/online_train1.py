# supervised by a global average embedding, which is a biased estimation of the true embedding
# use projection to enable a complex decoding
# makes no big difference than mean so far, the decoding may not work 🤦‍

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
import random

class Transform(nn.Module):
    def __init__(self, n=2, token_size=32, input_dim=2048):
        super().__init__()
        
        self.n=n
        self.dim= input_dim*token_size
        self.token_size=token_size
        self.input_dim=input_dim
        
        self.weight = nn.Parameter(torch.ones(self.n,1),requires_grad=True)
        
        self.projections = nn.ModuleList([nn.Sequential(
            nn.Linear(self.dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.dim)
        ) for _ in range(self.n)])
        
    def encode(self, x):
        x = x.view(-1, self.dim)
        x = self.weight*x
        return x
    
    def decode(self, x):
        out=[]
        for i in range(self.n):
            t = self.projections[i](x[i])
            out.append(t)
        x = torch.stack(out, dim=0)
        x=x.view(self.n,self.token_size,self.input_dim)
        x=torch.mean(x,dim=0)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

def online_train(cond, device="cuda:1",step=1000):
    old_device=cond.device
    dtype=cond.dtype
    cond = cond.clone().to(device,torch.float32)
    cond.requires_grad=False
    torch.set_grad_enabled(True)
    
    print("online training, initializing model...")
    n=cond.shape[0]
    model=Transform(n=n)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.MSELoss()
    model.to(device)
    model.train()
        
    y=torch.mean(cond,dim=0)
    
    random.seed(42)
    bar=tqdm(range(step))
    for s in bar:
        optimizer.zero_grad()
        attack_weight=[random.uniform(0.5,1.5) for _ in range(n)]
        attack_weight=torch.tensor(attack_weight)[:,None,None].to(device)
        x=attack_weight*cond
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        bar.set_postfix(loss=loss.item())
        
    weight=model.weight
    cond=weight[:,:,None]*cond
    print(weight)
    
    print("online training, ending...")
    del model
    del optimizer
    
    cond=torch.mean(cond,dim=0).unsqueeze(0)
    return cond.to(old_device,dtype=dtype)