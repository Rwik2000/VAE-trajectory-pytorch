
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net_enc2path import encodeToTraj
from torch.utils.data import TensorDataset, DataLoader
from net_enc2path import encodeToTraj

loadEnc = np.load("../Numpy_Dataset/train_data_encodings_torch.npy")
# loadEnc.transpose(0, 2 ,1)
load_track = np.load("../Numpy_Dataset/train_data_tr_torch.npy")
loadEnc_torch = torch.Tensor(loadEnc)
load_track_torch = torch.Tensor(load_track)



# exit()
BATCH_SIZE = 32
LATENT_SIZE = 25
dataset = TensorDataset(loadEnc_torch, load_track_torch)
dataset = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def loss_function(output, recons):
    loss = F.mse_loss(recons, output, reduction='sum')
    return loss
model = encodeToTraj(LATENT_SIZE)


if torch.cuda.is_available():
    model.cuda()
    
optimizer = optim.Adam(model.parameters(), lr = 0.00001)
def train(epoch):
    model.train()
    train_loss  = 0
    for batch_idx , (data, output) in enumerate(dataset):
        if(data.shape[0] == BATCH_SIZE):
            data = data.cuda()
            output = output.cuda()
            recons = model(data)

            loss = loss_function(recons, output)
            loss.backward()
            train_loss+=loss.item()
            optimizer.step()

            if batch_idx%100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(dataset.dataset),
                        100. * batch_idx / len(dataset), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataset.dataset)))
    if epoch%20==0:
        torch.save(model, "./models/trajec_model_"+str(epoch)+".pt")

for epoch in range(0,101):
    train(epoch)

