# from Train import train
import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import TensorDataset, DataLoader
import time

gc.collect()
torch.cuda.empty_cache()
train_images = np.load("../Numpy_Dataset/train_data_im_torch.npy")
train_images = train_images/255
_,_,H,W = train_images.shape
BATCH_SIZE = 1
LATENT_SIZE = 50
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

vae = torch.load("vae_model_10.pt")

def SaveEncodings():
    encodings = []
    for i in range(len(train_images)):
        print(i)
        inp = torch.Tensor(train_images[i])
        inp = inp.unsqueeze(0)
        inp = inp.cuda()
        m, var = vae.encoder(inp)
        z = vae.sampling(m,var)
        z = z.cpu().detach().numpy()
        encodings.append(z[0])
    print(encodings[0])
    np.save("../Numpy_Dataset/train_data_encodings_torch.npy", np.array(encodings))
    print("Saved!")

SaveEncodings()
# def getEncoding(image, getOutImg = 0):
#     image = cv2.resize(image, (200, 152))
#     image = image.transpose((2,0,1))
#     image = image/255
#     input_img = torch.Tensor(image)
#     input_img = input_img.unsqueeze(0)
#     inp = input_img.cuda()
#     st = time.time()
#     m,var = vae.encoder(inp)
#     # print(time.time()-st)
#     z = vae.sampling(m, var)
#     # print(time.time()-st)
#     img = vae.decoder(z)
#     print(time.time()-st)
#     img = img.detach().cpu().numpy()
#     img = img[0].transpose((1,2,0))
#     img = img*255
#     if getOutImg:
#         cv2.imshow("img", img)
#         cv2.imshow("img3", image.transpose(1,2,0))
#         cv2.waitKey(0)

#     return z

# img = cv2.imread("../images/VAE_img_11807.jpg")
# imgs = [img]
# for i in range(len(imgs)):
#     enc = getEncoding(img, 1)