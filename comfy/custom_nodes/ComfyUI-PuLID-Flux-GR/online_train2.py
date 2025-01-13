# self-supervised learning, one of the embedding acts as the target, the other as the support
# works nicely

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
        self.token_size=token_size
        
        self.weight = nn.Parameter(torch.ones(self.n,self.token_size),requires_grad=True)
        
    def encode(self, x):
        x = torch.einsum('bij,bi->ij', x, self.weight)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        return x
    
def criterion(output, target, token_sample_rate=0.25):
    t=target-output
    t=torch.norm(t,dim=1)
    s=random.sample(range(t.shape[0]),int(token_sample_rate*t.shape[0]))
    return torch.mean(t[s])

def online_train(cond, device="cuda:1",step=1000):
    old_device=cond.device
    dtype=cond.dtype
    cond = cond.clone().to(device,torch.float32)
    # cond.requires_grad=False
    # torch.set_grad_enabled(True)
    
    y=cond[0,:,:]
    cond=cond[1:,:,:]
    
    print("online training, initializing model...")
    n=cond.shape[0]
    model=Transform(n=n)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.to(device)
    model.train()
    
    random.seed(42)
    bar=tqdm(range(step))
    for s in bar:
        optimizer.zero_grad()
        x=cond
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        bar.set_postfix(loss=loss.item())
        
    weight=model.weight
    print(weight)
    cond=weight[:,:,None]*cond+y[None,:,:]*(1.0/n)
    
    print("online training, ending...")
    del model
    del optimizer
    
    cond=torch.mean(cond,dim=0).unsqueeze(0)
    return cond.to(old_device,dtype=dtype)