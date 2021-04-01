import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, H, W, LS, BS):
        super(VAE, self).__init__()
        self.H = H
        self.W = W
        self.ls = LS
        self.bs = BS
        # encoder part
        self.cv1 = nn.Conv2d(3, 40, 3, 2, 1)
        self.fc1 = nn.Linear(H//2*W//2*40, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc31 = nn.Linear(64, self.ls)
        self.fc32 = nn.Linear(64, self.ls)
        # decoder part
        self.fc4 = nn.Linear(self.ls, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, H//2*W//2*40)
        self.cvL = nn.ConvTranspose2d( 40, 3, 3, 2, 1, 1)
        
    def encoder(self, x):
        h = F.relu(self.cv1(x))
        h = h.view(self.bs, -1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = h.view(self.bs, 40, self.H//2,self.W//2)
        h = torch.sigmoid(self.cvL(h))
        return h
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var