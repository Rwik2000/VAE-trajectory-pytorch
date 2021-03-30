import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from net_VAE import VAE

train_images = np.load("../Numpy_Dataset/train_data_im_torch.npy")
test_images = np.load("../Numpy_Dataset/test_data_im_torch.npy")
train_images = train_images/255
test_images = test_images/255
_,_,H,W = train_images.shape

BATCH_SIZE = 32
LATENT_SIZE = 25
imageTensor = torch.Tensor(train_images)
dataset = TensorDataset(imageTensor, imageTensor)
dataset = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

vae = VAE(H,W, LATENT_SIZE, BATCH_SIZE)
pytorch_total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
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
