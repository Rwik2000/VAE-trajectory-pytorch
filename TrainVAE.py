import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import cv2
    # import torch
gc.collect()
torch.cuda.empty_cache()
train_images = np.load("../Numpy_Dataset/train_data_im_torch.npy")
test_images = np.load("../Numpy_Dataset/test_data_im_torch.npy")
train_images = train_images/255
test_images = test_images/255
_,_,H,W = train_images.shape
# print(H,W)
BATCH_SIZE = 32
LATENT_SIZE = 25
imageTensor = torch.Tensor(train_images)
dataset = TensorDataset(imageTensor, imageTensor)
dataset = DataLoader(dataset, batch_size=BATCH_SIZE)

class VAE(nn.Module):
    def __init__(self, H, W):
        super(VAE, self).__init__()
        self.H = H
        self.W = W
        # encoder part
        self.cv1 = nn.Conv2d(3, 128, 3, 2, 1)
        self.cv2 = nn.Conv2d(128, 64 , 3, 2, 1)
        self.cv3 = nn.Conv2d(64, 32 , 3, 2, 1)
        self.fc1 = nn.Linear(H//8*W//8*32, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc31 = nn.Linear(64, LATENT_SIZE)
        self.fc32 = nn.Linear(64, LATENT_SIZE)
        # decoder part
        self.fc4 = nn.Linear(LATENT_SIZE, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc7 = nn.Linear(128, H//8*W//8*32)
        self.cvd1 = nn.ConvTranspose2d(32, 64, 3, 2, 1, 1)
        self.cvd2 = nn.ConvTranspose2d(64, 128, 3, 2, 1, 1)
        self.cvd3 = nn.ConvTranspose2d( 128, 3, 3, 2, 1, 1)
        
    def encoder(self, x):
        h = F.relu(self.cv1(x))
        h = F.relu(self.cv2(h))
        h = F.relu(self.cv3(h))
        h = h.view(BATCH_SIZE, -1)
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
        h = h.view(BATCH_SIZE, 32, self.H//8,self.W//8)
        h = F.relu(self.cvd1(h))
        h = F.relu(self.cvd2(h))
        h = torch.sigmoid(self.cvd3(h))
        return h
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

# build model
vae = VAE(H,W)
# vae = torch.load("vae_model_30.pt")
pytorch_total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
# print(pytorch_total_params)
if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters(), lr = 0.001)
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.mse_loss(recon_x.view(-1, H*W*3), x.view(-1, H*W*3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataset):
        if(data.shape[0] == BATCH_SIZE):
            data = data.cuda()
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = vae(data)
            # print(recon_batch.shape)
            loss = loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataset.dataset),
                    100. * batch_idx / len(dataset), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataset.dataset)))

    if epoch%10==0:
        torch.save(vae, "./models/vae_model_"+str(epoch)+".pt")
        # exit()

for epoch in range(0, 101):
    train(epoch)
