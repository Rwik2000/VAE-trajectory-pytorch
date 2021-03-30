import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, H, W, latent_size, batch_size):
        super(VAE, self).__init__()
        self.H = H
        self.W = W
        self.batch_size = batch_size
        # encoder part
        self.cv1 = nn.Conv2d(3, 128, 3, 2, 1)
        self.cv2 = nn.Conv2d(128, 64 , 3, 2, 1)
        self.cv3 = nn.Conv2d(64, 32 , 3, 2, 1)
        self.fc1 = nn.Linear(H//8*W//8*32, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc31 = nn.Linear(64, latent_size)
        self.fc32 = nn.Linear(64, latent_size)
        # decoder part
        self.fc4 = nn.Linear(latent_size, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc7 = nn.Linear(128, H//8*W//8*32)
        self.cvd1 = nn.ConvTranspose2d(32, 64, 3, 2, 1, 1)
        self.cvd2 = nn.ConvTranspose2d(64, 128, 3, 2, 1, 1)
        self.cvd3 = nn.ConvTranspose2d( 128, 3, 3, 2, 1, 1)
        
    def encoder(self, x):
        h = F.relu(self.cv1(x))
        h = F.relu(self.cv2(h))
        h = F.relu(self.cv3(h))
        h = h.view(self.batch_size, -1)
        h = F.relu(self.fc1(h))
        # h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        # h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = h.view(self.batch_size, 32, self.H//8,self.W//8)
        h = F.relu(self.cvd1(h))
        h = F.relu(self.cvd2(h))
        h = torch.sigmoid(self.cvd3(h))
        return h
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
