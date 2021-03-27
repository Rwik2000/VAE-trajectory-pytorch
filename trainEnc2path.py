from numpy.core.defchararray import mod
# from Train import LATENT_SIZE
import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

loadEnc = np.load("../Numpy_Dataset/train_data_encodings_torch.npy")
# loadEnc.transpose(0, 2 ,1)
load_track = np.load("../Numpy_Dataset/train_data_tr_torch.npy")
loadEnc_torch = torch.Tensor(loadEnc)
load_track_torch = torch.Tensor(load_track)



# exit()
BATCH_SIZE = 32
LATENT_SIZE = 25
dataset = TensorDataset(loadEnc_torch, load_track_torch)
dataset = DataLoader(dataset, batch_size=BATCH_SIZE)

class encodeToTraj(nn.Module):
    def __init__(self, latent_size):
        super(encodeToTraj, self).__init__()
        # self.cvd1 = nn.Conv1d(1, 32, 3, 1, 1)
        self.l1 = nn.Linear(LATENT_SIZE, 64)
        # self.d1 = nn.Dropout(0.7)
        self.l2 = nn.Linear(64, 512 )
        self.l3 = nn.Linear(512, 1024)
        self.l4 = nn.Linear(1024, 256)
        self.l5 = nn.Linear(256, 68)
    
    def forward(self, x):
        # print(x.shape)
        # h = F.relu(self.cvd1(x))
        # h = h.view(BATCH_SIZE, -1)
        h = F.relu(self.l1(x))
        # h= F.relu(self.d1(h))
        # print(h.shape)
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = F.sigmoid(self.l5(h))

        return h
def loss_function(output, recons):
    loss = F.mse_loss(recons, output, reduction='sum')
    return loss
model = encodeToTraj(LATENT_SIZE)

'''
uncomment while testing
'''
model = torch.load("./models/trajec_model_100.pt")
# if torch.cuda.is_available():
#     model.cuda()
# optimizer = optim.Adam(model.parameters(), lr = 0.00001)
# def train(epoch):
#     model.train()
#     train_loss  = 0
#     for batch_idx , (data, output) in enumerate(dataset):
#         if(data.shape[0] == BATCH_SIZE):
#             data = data.cuda()
#             output = output.cuda()
#             recons = model(data)
#             # if epoch==50:
#             #     fin = recons[0].reshape(2,34)
#             #     fin = fin.detach().cpu().numpy()
#             #     print(fin.reshape(2,-1).T)
#             #     print(output[0])
#             #     exit()
#             loss = loss_function(recons, output)
#             loss.backward()
#             train_loss+=loss.item()
#             optimizer.step()

#             if batch_idx%100 == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                         epoch, batch_idx * len(data), len(dataset.dataset),
#                         100. * batch_idx / len(dataset), loss.item() / len(data)))
#     print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataset.dataset)))
#     if epoch%20==0:
#         torch.save(model, "./models/trajec_model_"+str(epoch)+".pt")

# for epoch in range(0,101):
#     train(epoch)

'''
testing encoder to path
'''
def test():
    # input_img = input_img.unsqueeze(0)
    for i in range(1000):
        num =i
        inp = torch.Tensor(loadEnc[num])
        outp = load_track[num]
        inp = inp.cuda()
        inp = inp.unsqueeze(0)
        fin_image = np.zeros((600, 800))

        # print(inp)
        k = model(inp)
        k = k.detach().cpu().numpy()
        k = k.reshape(2,-1).T
        outp = outp.reshape(2,-1).T
        k = (k*[800, 600]).astype(int)
        outp = (outp*[800, 600]).astype(int)
        # print(np.uint8(k*[800, 600]))\
        for i in range(len(outp)-1):
            fin_image = cv2.line(fin_image, tuple(k[i]), tuple(k[i+1]), (255,255,255), 3)
            fin_image = cv2.line(fin_image, tuple(outp[i]), tuple(outp[i+1]), (255,255,255), 3)
        
        cv2.imshow("x.png", fin_image)
        cv2.waitKey(1000)
    # print(k)



test()
